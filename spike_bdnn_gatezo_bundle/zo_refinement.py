import math

import numpy as np
import torch


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize(vector, eps=1e-12):
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector
    return vector / norm


def _select_subset(X, T, sample_size):
    if sample_size is None or sample_size <= 0 or sample_size >= X.shape[0]:
        return X, T
    indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
    return X[indices], T[indices]


def _cross_entropy_from_logits(logits, T):
    labels = np.argmax(T, axis=1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True) + 1e-12)
    return float(-np.mean(log_probs[np.arange(labels.shape[0]), labels]))


def _softmax_from_logits(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)


def _current_loss_and_beta(model, X, T):
    H, _ = model.getH(X)
    beta = np.linalg.pinv(H) @ T
    logits = H @ beta
    loss = _cross_entropy_from_logits(logits, T)
    return loss, beta


def _current_state(model, X, T):
    H, _ = model.getH(X)
    beta = np.linalg.pinv(H) @ T
    logits = H @ beta
    loss = _cross_entropy_from_logits(logits, T)
    probs = _softmax_from_logits(logits)
    return loss, beta, probs


def _pack_new_params(model, start, end):
    W = _as_numpy(model.W)
    U = _as_numpy(model.U)
    parts = [W[:, start:end].reshape(-1)]
    if getattr(model, "zo_optimize_u", True):
        parts.append(U[start:end, start:end].reshape(-1))
    return np.concatenate(parts).astype(np.float64)


def _assign_new_params(model, theta, start, end):
    W = _as_numpy(model.W).copy()
    U = _as_numpy(model.U).copy()
    new_nodes = end - start
    w_size = W[:, start:end].size

    W[:, start:end] = theta[:w_size].reshape(W[:, start:end].shape)
    if getattr(model, "zo_optimize_u", True):
        U[start:end, start:end] = theta[w_size:].reshape(new_nodes, new_nodes)

    model.W = torch.Tensor(W)
    model.U = torch.Tensor(U)


def _input_features(X):
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 3:
        return X_arr.mean(axis=1)
    if X_arr.ndim == 2:
        return X_arr
    return X_arr.reshape(X_arr.shape[0], -1)


def _residual_guided_direction(model, X, T, theta, start, end):
    direction = np.zeros_like(theta)
    if end <= start:
        return direction

    W = _as_numpy(model.W)
    w_shape = W[:, start:end].shape
    w_size = W[:, start:end].size

    _, beta, probs = _current_state(model, X, T)
    beta_new = np.asarray(beta[start:end, :], dtype=np.float64)
    if beta_new.size == 0:
        return direction

    residual = np.asarray(probs - T, dtype=np.float64)
    hard_score = np.linalg.norm(T - probs, axis=1)
    top_ratio = float(getattr(model, "zo_residual_top_ratio", 0.3))
    top_ratio = min(max(top_ratio, 0.0), 1.0)

    if 0.0 < top_ratio < 1.0:
        count = max(1, int(math.ceil(X.shape[0] * top_ratio)))
        selected = np.argsort(hard_score)[-count:]
    else:
        selected = np.arange(X.shape[0])

    X_feat = _input_features(X)[selected]
    if X_feat.shape[1] != w_shape[0]:
        return direction

    hidden_error = residual[selected] @ beta_new.T
    weights = hard_score[selected].reshape(-1, 1)
    weights = weights / (np.mean(weights) + 1e-12)

    # Descent-like direction for W_new. It is only used as a perturbation prior;
    # the actual update is still accepted through zero-order loss evaluations.
    w_direction = -(X_feat.T @ (weights * hidden_error)) / max(1, selected.size)
    direction[:w_size] = w_direction.reshape(-1)
    return _normalize(direction)


def _make_direction(model, X, T, base_theta, start, end, history_direction, alpha):
    noise = _normalize(np.random.randn(base_theta.size))

    mode = str(getattr(model, "zo_direction_mode", "random")).lower()
    if mode in ("residual", "residual_guided", "guided"):
        guided = _residual_guided_direction(model, X, T, base_theta, start, end)
        mix = float(getattr(model, "zo_residual_mix", 0.5))
        mix = min(max(mix, 0.0), 1.0)
        if np.linalg.norm(guided) > 1e-12 and mix > 0.0:
            noise = _normalize(mix * guided + math.sqrt(max(0.0, 1.0 - mix ** 2)) * noise)

    if alpha > 0.0 and np.linalg.norm(history_direction) > 1e-12:
        noise = alpha * _normalize(history_direction) + math.sqrt(1.0 - alpha ** 2) * noise
        noise = _normalize(noise)

    return noise


def _iter_minibatches(X, T, batch_size):
    if batch_size is None or batch_size <= 0 or batch_size >= X.shape[0]:
        yield X, T
        return
    indices = np.random.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], T[batch_indices]


def _zo_step(model, X_eval, T_eval, start, end, history_direction, alpha, lr, mu, directions, accept_tol, verbose, require_improvement=True):
    base_theta = _pack_new_params(model, start, end)
    base_loss, base_beta = _current_loss_and_beta(model, X_eval, T_eval)
    grad_estimate = np.zeros_like(base_theta)

    for _ in range(directions):
        _assign_new_params(model, base_theta, start, end)
        z = _make_direction(model, X_eval, T_eval, base_theta, start, end, history_direction, alpha)

        _assign_new_params(model, base_theta + mu * z, start, end)
        loss_plus, _ = _current_loss_and_beta(model, X_eval, T_eval)

        _assign_new_params(model, base_theta - mu * z, start, end)
        loss_minus, _ = _current_loss_and_beta(model, X_eval, T_eval)

        grad_estimate += ((loss_plus - loss_minus) / (2.0 * mu)) * z

    grad_estimate /= directions
    candidate_theta = base_theta - lr * grad_estimate
    _assign_new_params(model, candidate_theta, start, end)
    candidate_loss, candidate_beta = _current_loss_and_beta(model, X_eval, T_eval)

    if (not require_improvement) or candidate_loss <= base_loss - accept_tol:
        if verbose:
            print("ZO step accepted, CE loss {:.6f}".format(candidate_loss))
        return candidate_theta.copy(), candidate_beta, candidate_loss, candidate_theta - base_theta, True

    _assign_new_params(model, base_theta, start, end)
    if verbose:
        print("ZO step rejected, CE loss {:.6f}".format(candidate_loss))
    return base_theta.copy(), base_beta, base_loss, history_direction * 0.5, False


def zero_order_refine_new_nodes(model, X, T, start, end):
    steps = int(getattr(model, "zo_steps", 0))
    if steps <= 0 or end <= start:
        return None

    theta = _pack_new_params(model, start, end)
    if theta.size == 0:
        return None

    lr = float(getattr(model, "zo_lr", 1e-2))
    mu = float(getattr(model, "zo_mu", 1e-3))
    alpha = float(getattr(model, "zo_alpha", 0.0))
    directions = max(1, int(getattr(model, "zo_directions", 1)))
    accept_tol = float(getattr(model, "zo_accept_tol", 0.0))
    verbose = bool(getattr(model, "zo_verbose", False))
    converge = bool(getattr(model, "zo_converge", False))

    alpha = min(max(alpha, 0.0), 1.0)
    history_direction = np.zeros_like(theta)
    best_theta = theta.copy()

    if converge:
        batch_size = getattr(model, "zo_batch_size", None)
        if batch_size is not None:
            batch_size = int(batch_size)
        max_epochs = max(1, int(getattr(model, "zo_max_epochs", 1)))
        best_loss, best_beta = _current_loss_and_beta(model, X, T)

        for epoch in range(max_epochs):
            for X_batch, T_batch in _iter_minibatches(X, T, batch_size):
                step_theta, step_beta, step_loss, history_direction, accepted = _zo_step(
                    model, X_batch, T_batch, start, end, history_direction,
                    alpha, lr, mu, directions, accept_tol, verbose,
                    require_improvement=False
                )
                best_theta = step_theta.copy()
                best_loss = step_loss
                best_beta = step_beta
            if verbose:
                print("ZO epoch {} finished, last mini-batch CE loss {:.6f}".format(epoch + 1, best_loss))
    else:
        X_eval, T_eval = _select_subset(X, T, getattr(model, "zo_sample_size", None))
        best_loss, best_beta = _current_loss_and_beta(model, X_eval, T_eval)

        for step in range(steps):
            step_theta, step_beta, step_loss, history_direction, accepted = _zo_step(
                model, X_eval, T_eval, start, end, history_direction,
                alpha, lr, mu, directions, accept_tol, verbose
            )
            if accepted and step_loss <= best_loss - accept_tol:
                best_theta = step_theta.copy()
                best_loss = step_loss
                best_beta = step_beta
                if verbose:
                    print("ZO step {} accepted, CE loss {:.6f}".format(step + 1, best_loss))

    _assign_new_params(model, best_theta, start, end)
    model.Beta = best_beta
    return best_loss
