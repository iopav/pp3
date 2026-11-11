"""GPU port of native BDNN, including candidate-count, sorting and indexing quirks."""
import numpy as np
import math
import torch


class BDNNGPU:
    def __init__(self, inputs, classes, settings, device):
        self.settings, self.device = settings, torch.device(device)
        self.alpha, self.beta = settings['alpha'], settings['beta']
        self.W = torch.empty(inputs, 0, device=device)
        self.U = torch.empty(0, 0, device=device)
        self.Beta = torch.empty(0, classes, device=device)
        self.L, self.counter = 0, None

    @torch.no_grad()
    def hidden(self, x, w, u):
        x, w, u = x.float(), w.float(), u.float()
        syn = x.new_zeros((len(x), w.shape[1]))
        mem, out = syn.clone(), syn.clone()
        history = []
        inputs = torch.einsum('abc,cd->abd', x, w)
        for i in range(x.shape[1]):
            drive = inputs[:, i] + torch.einsum('ab,bc->ac', out, u)
            out = (mem - 1 > 0).float()
            syn, mem = self.alpha * syn + drive, (self.beta * mem + syn) * (1 - out)
            history.append(out)
        spikes = torch.stack(history, 1)
        return spikes.sum(1) / x.shape[1], spikes

    def getH(self, x):
        return self.hidden(x, self.W, self.U)

    def inequalityEq(self, eq, gk, r):
        a = (1 - r) * eq.T
        return (eq.T @ gk).square() / (gk.T @ gk) - a @ eq

    @torch.no_grad()
    def sc_Search(self, x, target, candidates=None):
        size, count = self.settings['pool_size'], self.settings['growth_step']
        scores, accepted_w, accepted_u = [], [], []
        nC = 0
        for j, scale in enumerate(self.settings['lambdas']):
            scale = torch.tensor(scale, dtype=torch.float32, device=self.device)
            if candidates is None:
                # Preserve native NumPy random stream, then perform arithmetic
                # on GPU. Sampling and discrete control are host operations.
                uniform = np.random.rand(x.shape[2], size)
                cached = np.random.get_state()[3]
                normal = np.random.randn(size, size)
                if self.counter is not None:
                    pairs = (max(0, normal.size - cached) + 1) // 2
                    self.counter.add_manual('native_randn_transform', math.ceil(pairs * (28 / math.pi + 6)),
                        rule='numpy_legacy_polar_expected_cost_estimate_rng_engine_excluded')
                w = scale * (2 * torch.as_tensor(uniform, device=self.device) - 1)
                u = scale * (2 * torch.as_tensor(normal, device=self.device) - 1)
            else:
                w, u = candidates[j]
            h, _ = self.hidden(x, w, u)
            for value in self.settings['r']:
                r = torch.tensor(value, dtype=torch.float32, device=self.device)
                for t in range(size):
                    ksi = torch.zeros((1, target.shape[1]), dtype=torch.float64, device=self.device)
                    for c in range(target.shape[1]):
                        ksi[0, c] = self.inequalityEq(target[:, c:c+1], h[:, t:t+1], r).reshape(())
                    score = ksi.sum(0).reshape(-1, 1)
                    if bool(ksi.min() > 0):
                        scores.append(score)
                        accepted_w.append(w[:, t:t+1])
                        accepted_u.append(u[:, t:t+1])
                # Native len(C) is class count, not candidate count.
                nC = target.shape[1] if scores else 0
                if nC >= count:
                    break
            if nC >= count:
                break
        if nC < count:
            return None
        matrix = torch.cat(scores, 1)
        # Native [::-1] reverses class rows, keeping ascending candidate order.
        # Keep NumPy's native tie ordering. Only this discrete control index
        # selection stays on host; scores and all model arithmetic are CUDA.
        order = matrix.detach().cpu().numpy().argsort(axis=1)[::-1].copy()
        indices = torch.as_tensor(order[0, :count], device=self.device)
        w, u = torch.cat(accepted_w, 1), torch.cat(accepted_u, 1)
        return w[:, indices], u[indices][:, indices]

    def addNodes(self, w, u):
        if self.L == 0:
            self.W, self.U = w, u
        else:
            self.W = torch.cat((self.W, w), 1).float()
            # Keep the original operator; it only allocates/copies two blocks.
            # PP3 has no block_diag rule, so annotate it locally without editing
            # the shared counter or counting its copies as arithmetic.
            counter = self.counter
            previous = counter.paused if counter else False
            if counter:
                counter.paused = True
            try:
                self.U = torch.block_diag(self.U, u)
            finally:
                if counter:
                    counter.paused = previous
            if counter:
                counter.add_manual('block_diag', 0, rule='block_diagonal_allocation_and_copy_only')
        self.L += 1

    def computeBeta(self, h, target):
        counter = self.counter
        active = counter is not None and counter.enabled and not counter.paused
        previous = counter.paused if counter else False
        if counter:
            counter.paused = True
        try:
            # NumPy linalg uses double-precision LAPACK internally for float32.
            inverse = torch.linalg.pinv(h.double(), atol=0., rtol=1e-15).to(h.dtype)
        finally:
            if counter:
                counter.paused = previous
        if active:
            m, n = h.shape
            q, p = min(m, n), max(m, n)
            cost = 0 if not q else 4*p*q*q + 8*q**3 + 1 + q + q*m + 2*n*q*m
            counter.add_manual('cuda_pinv', cost,
                rule='real_pinv_thin_svd_4pq2_8q3_plus_reconstruction_estimate')
        self.Beta = inverse @ target

    @staticmethod
    def softmax(output):
        exp = torch.exp(output - output.max(1, keepdim=True).values)
        return exp / exp.sum(1, keepdim=True)

    def upgradeSCN(self, x, target):
        h, spikes = self.getH(x)
        self.computeBeta(h, target)
        drive = torch.einsum('abc,cd->abd', spikes.float(), self.Beta.float())
        filt = x.new_zeros((len(x), target.shape[1]))
        output, history = filt.clone(), []
        for t in range(x.shape[1]):
            filt, output = self.alpha * filt + drive[:, t], self.beta * output + filt
            history.append(output)
        result = self.softmax(torch.stack(history, 1).sum(1))
        self.COST = (target - result).square().mean(0).mean()
        return result, self.COST

    def getOutput(self, x):
        h, _ = self.getH(x)
        return self.softmax(torch.einsum('ab,bc->ac', h.float(), self.Beta.float()))

    def getLabel(self, x):
        output = self.getOutput(x)
        return torch.nn.functional.one_hot(output.argmax(1), output.shape[1]).float()

    def state_dict(self):
        return dict(W=self.W, U=self.U, Beta=self.Beta)

    def parameter_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.state_dict().values())
