
import numpy as np
import torch.nn as nn
import numpy.matlib
from math import sqrt
from scipy.special import expit
from sklearn import metrics
import torch
from updatelist import *
from sklearn.metrics import mean_squared_error
from snnandfunction import spike_fn
from standardmatrxi import softmax_standardization
import numpy_flops
# make output of SCN6 is limitted in (0,1) with softmax
#plot  neurons
import time
import matplotlib.pyplot as plt
#！！！！！！！！can not use !!!!!!!!!!!
# because>
#        O = softmax(O)

#       Error = mean_squared_error(T, O)
class bdnn1(torch.nn.Module):
    name = 'Stochastic Configuration Networks'
    version = '1.0 beta'
    # Basic parameters （networks structure）
    L = 0  # hidden node number / start with 0
    W = []  # input weight matrix
    b = []  # hidden layer bias vector
    U = []
    Beta = []  # output weight vector

    # constructor
    def __init__(self,
                 L_max=100000,
                 T_max=100,
                 tol=1e-4,
                 Lambdas=[0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200],
                 r=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999],
                 nB=1,
                 verbose=50,
                 alpha=0.13,
                 beta=1
                 ):
        self.alpha = alpha
        self.beta = beta
        if isinstance(verbose, int):
            self.verbose = verbose
        if isinstance(L_max, int):
            self.L_max = L_max
            #if L_max > 5000:
                #self.verbose = 500  # does not need too many output
        if isinstance(T_max, int):
            self.T_max = T_max
        if isinstance(tol, float):
            self.tol = tol
        if isinstance(Lambdas, list):
            self.Lambdas = torch.Tensor(np.array(Lambdas))
        if isinstance(r, list):
            self.r = torch.Tensor(np.array(r))
        if isinstance(nB, int):
            self.nB = nB

    def printProperties(self):
        print('   Name:{}'.format(self.name))
        print('version:{}'.format(self.version))
        print('      L:{}'.format(self.L))
        print('      W:{}'.format(self.W.shape))
        print('      U:{}'.format(self.U.shape))
        #print('      b:{}'.format(self.b.shape))
        print('   Beta:{}'.format(self.Beta.shape))
        print('      r:{}'.format(self.r))
        print('    tol:{}'.format(self.tol))
        print('Lambdas:{}'.format(self.Lambdas))
        print('  L_max:{}'.format(self.L_max))
        print('  T_max:{}'.format(self.T_max))
        print('     nB:{}'.format(self.nB))
        print('verbose:{}'.format(self.verbose))
        print('   COST:{}'.format(self.COST))

        # inequality equation return the ksi

    def _flops_counter(self):
        runtime = getattr(self, '_pp3_runtime', None)
        return runtime.counter if runtime is not None else None

    def _accuracy(self, target, output):
        return numpy_flops.metric(self._flops_counter(), 'accuracy_score',
                                  metrics.accuracy_score, target, output)

    def inequalityEq(self, eq, gk, r_L):
        counter = self._flops_counter()
        a = numpy_flops.multiply(counter, (1 - r_L), eq.conj().T)
        # The scalar subtraction is dispatched for a Torch r_L. Mixed
        # Tensor/ndarray multiplication checks actual dispatch ownership.
        if not isinstance(r_L, torch.Tensor) and numpy_flops.real_arrays(counter, 'inequality_scale', eq):
            numpy_flops.record(counter, 'inequality_scale', 1)
        ksi = (numpy_flops.matmul(counter, eq.conj().T, gk) ** 2) / \
              numpy_flops.matmul(counter, gk.conj().T, gk) - \
              numpy_flops.matmul(counter, np.array(a), eq)
        if numpy_flops.real_arrays(counter, 'inequality_elementwise', eq, gk):
            numpy_flops.record(counter, 'inequality_elementwise', 3 * ksi.size)

        return ksi

    # Search for {WB,bB} of nB nodes
    def sc_Search(self, X, E0):
        # X is  in int type when we chiced the X(tensor) to np.array(X)
        # 0: continue; 1: stop;
        # return a good node /or stop training by set Flag = 1
        Flag = 0
        WB = []
        bB = []
        UB = []
        #tau_mem = 10e-3
        #tau_syn = 5e-3
        # get sample and feature number
        d = X.shape[2]
        l = X.shape[1]
        m = E0.shape[1]
        #print('In sc<-reseach m is :'+str(m))
        #print('In sc_Research, shape of E0 is: ' + str(E0.shape))
        # Linear search for better nodes
        # container of kesi
        C = []
        for Lambda in self.Lambdas:

            # Generate candidates T_max vectors of w and b for selection
            # alpha = 0.2
            #
            # beta = 1
            # WW is d-by-T_max
            WT = numpy_flops.multiply(self._flops_counter(), Lambda, (2 * np.random.rand(d, self.T_max) - 1))
            # Only NumPy 2*x-1 here; multiply() accounts for Lambda separately.
            numpy_flops.record(self._flops_counter(), 'candidate_uniform_affine',
                               2 * d * self.T_max)

            # bb is 1-by-T_max
            #bT = Lambda * (2 * np.random.rand(1, self.T_max) - 1)

            # UT is Tmax, Tmax
            UT = numpy_flops.multiply(self._flops_counter(), Lambda,
                                     (2 * numpy_flops.randn(self._flops_counter(), self.T_max, self.T_max) - 1))
            numpy_flops.record(self._flops_counter(), 'candidate_normal_affine',
                               2 * self.T_max**2)


            HT = np.zeros((X.shape[0], self.T_max))

            h_prev = np.zeros((X.shape[0],self.T_max))#ng you have batch_size and hidden_size defined
            syn = torch.zeros((X.shape[0],self.T_max))
            mem = torch.zeros((X.shape[0],self.T_max))
            mem_rec = []
            spk_rec = []
            out = torch.zeros((X.shape[0],self.T_max))
            #print((type(WT)))
            h1_from_inputs = torch.einsum("abc,cd->abd", (torch.tensor(X, dtype=torch.float32), torch.tensor(WT, dtype=torch.float32)))

            # Loop through each time step
            for i in range(l):
                h1 = h1_from_inputs[:, i, :] + torch.einsum("ab,bc->ac", (torch.tensor(out, dtype=torch.float32), torch.tensor(UT, dtype=torch.float32)))
                out = spike_fn(mem -1)
                rst = out.detach()
                new_syn = self.alpha * syn + h1
                new_mem = (self.beta * mem + syn) * (1.0 - rst)
                mem_rec.append(mem)
                spk_rec.append(out)
                mem = new_mem
                syn = new_syn

            spk_rec = torch.stack(spk_rec, dim=1)

            HT = torch.sum(spk_rec, 1)
            HT = HT.numpy()
                #HT = np.add(HT, HT_i)

            # Now HT contains the outputs for all time steps
            HT = HT / l
            numpy_flops.record(self._flops_counter(), 'candidate_time_average', HT.size)
            #HT = expit(HT)

            for r_L in self.r:
                # calculate the Ksi value
                # searching one by one
                for t in range(0, self.T_max):
                    # print('t in range Tmax is: ' + str(t))
                    # Calculate H_t
                    H_t = HT[:, t]
                    # Calculate kesi_1, ... kesi_m
                    ksi_m = np.zeros((1, m), dtype=np.float64)

                    for i_m in range(0, m):
                        eq = E0[:, i_m].reshape(-1, 1)
                        gk = H_t.reshape(-1, 1)
                        ksi_m[0, i_m] = self.inequalityEq(eq, gk, r_L)


                    Ksi_t = np.sum(ksi_m, 0).reshape(-1, 1)  # values of ksi_m is equail to Ksi_t
                    numpy_flops.record(self._flops_counter(), 'candidate_score_sum', ksi_m.size - Ksi_t.size)

                    # in sc_research function, ksi_m is(1, 10) and in sc_research function, Ksi_t is(10, 1)
                    # print('for t in T_max, the Ksi_t is: ' + str(Ksi_t))
                    # [[15.06438256],[14.31606498],[14.83900515],[14.84635216],[14.28682784], [13.78974091]
                    #  [13.78974091],[14.1900178 ], [13.69453301],[15.22712635],[14.5966143 ]]
                    if np.min(ksi_m) > 0:
                        if type(C) == list:
                            C = Ksi_t
                        else:
                            C = np.concatenate([C, Ksi_t], axis=1)
                            # print('in sc_reaearch function, the size of C is' + str(C.shape))
                            # with the increasement of t in Tmax, the siz eof C is increased by steps
                            # like (10,1)->(10,100)
                        if type(WB) == list:
                            WB = WT[:, t].reshape(-1, 1)
                        else:
                            WB = np.concatenate(
                                (WB, WT[:, t].reshape(-1, 1)), axis=1)
                            # print('in sc_reaearch function, the size of WB is' + str(WB.shape))
                            # (64,1)->(64,100)
                            # similar to C
                        if type(UB) == list:
                            UB = UT[:, t].reshape(-1, 1)
                        else:
                            UB = np.concatenate((UB, UT[:, t].reshape(-1, 1)), axis=1)

                        #if type(bB) == list:
                            #bB = bT[:, t].reshape(-1, 1)

                        #else:
                            #bB = np.concatenate(
                             #   (bB, bT[:, t].reshape(-1, 1)), axis=1)
                            # print('in sc_reaearch function, the size of bB is' + str(bB.shape))
                            # (1,1)->(1,100)
                nC = len(C)
                # print('for t in T-max,')
                if nC >= self.nB:
                    break  # r loop
                else:
                    continue
            # end r
            if nC >= self.nB:
                break  # lambda loop
            else:
                continue
        # end lambda
        # Return the good node / or stop the training
        if nC >= self.nB:
            # print('In sc_research function self.nB is ' + str(self.nB)): always be 10
            I = C.argsort(axis=1)[::-1]
            I_nb = I[0, 0:self.nB]
            #print('in sc-reserch function, the I_nb is' + str(I_nb))# for different L, the size of I_nb is same and values are different
            WB = WB[:, I_nb]
            # print('in sc-reserch function, the size of WB is' + str(WB.shape)): always be (64,10)
            #bB = bB[:, I_nb]
            UB = UB[I_nb, :][:, I_nb]

            #print('in sc-research, the shape of numpy UB is : '+ str(UB.shape))

            # print('in sc-reserch function, the size of bB is' + str(bB.shape))
            # In sc_research function the size of WB and bB always be (64, 10)(1, 10)

        # discard w b
        if nC == 0 or nC < self.nB:
            Flag = 1

        # In sc_research function the size of WB and bB always be (64, 10)(1, 10)

        return [WB, UB, Flag]

    # Add nodes to the previous model
    def addNodes(self, w_L,  u_L):
        #print('Type of W is: ' + str(type(self.W)))
        if type(self.W) == list:
            self.W = w_L
        else:

            self.W = torch.Tensor(np.concatenate((self.W, w_L), axis=1))
            #print('In function addNodes the size of self.W is' + str(self.W.shape))

        if isinstance(self.U, list):
            self.U = u_L
        else:
            self.U = torch.block_diag(torch.tensor(self.U), torch.tensor(u_L))
            #self.U = torch.Tensor(np.concatenate((self.U, u_L), axis=1))
            # Now self.U should be the matrix you described

        self.L = self.L + 1
        #print('in addNodes the L is ' + str(self.L))

    # ComputeBeta
    def computeBeta(self, H, T):
        #print('the size of H is: ' + str(torch.Tensor(H).size()) + 'the size of T is: ' + str(torch.Tensor(T).size()))

        counter = self._flops_counter()
        Beta = numpy_flops.matmul(counter, numpy_flops.pinv(counter, H), T)
        #print(Beta)
        self.Beta = Beta

    # Output Matrix of hidden layer
    def getH(self, X):
        # print('In activation, the size of W is '+ str((torch.Tensor(self.W)).size()))#[64, 10]
        # print('In activation, the size of b is '+ str((torch.Tensor(self.b)).size()))#[1, 10]
        l = X.shape[1]
        H = np.zeros((X.shape[0], self.W.shape[1]))
        syn = torch.zeros((X.shape[0],self.W.shape[1]))
        mem = torch.zeros((X.shape[0],self.W.shape[1]))
        mem_rec = []
        spk_rec = []
        out = torch.zeros((X.shape[0],self.W.shape[1]))
        h1_from_inputs = torch.einsum("abc,cd->abd", (torch.Tensor(X).float(), torch.Tensor(self.W).float()))
        # Check the type of self.U
        #print("Type of self.U:", type(self.U))
        # Generate candidates T_max vectors of w and b for selection

        # alpha = 0.25
        # beta = 1

        # Check the shape of self.U
        #if isinstance(self.U, np.ndarray):
        #    print("Shape of self.U:", self.U.shape)
        #else:
        #    print("self.U is not a NumPy array.")

        h_prev = np.zeros((X.shape[0], self.U.shape[1]))
        #print('in get H, self.U size is: ' + str(self.U.shape))
        #state = np.zeros((X.shape[0], self.U.shape[1]))
        for i in range(l):
            h1 = h1_from_inputs[:, i, :] + torch.einsum("ab,bc->ac", (torch.tensor(out).float(), torch.tensor(self.U).float()))
            out = spike_fn(mem-1)
            rst = out.detach()
            new_syn = self.alpha * syn + h1
            new_mem = (self.beta * mem + syn)*(1 - rst)
            mem_rec.append(mem)
            spk_rec.append(out)
            mem = new_mem
            syn = new_syn
        spk_rec = torch.stack(spk_rec, dim=1)

        HT = torch.sum(spk_rec, 1)
        H = HT.numpy()
        # HT = np.add(HT, HT_i)
        H = H / l
        numpy_flops.record(self._flops_counter(), 'hidden_time_average', H.size)
        # the size of H is: torch.Size([1610, 10]))
        return H,spk_rec

    # Compute the Beta, Output, ErrorVector and Cost
    def upgradeSCN(self, X, T):
        H, spk_rec = self.getH(X)
        self.computeBeta(H, T)
        h2 = torch.einsum("abc,cd->abd", (torch.Tensor(spk_rec).float(), torch.Tensor(self.Beta).float()))
        flt = torch.zeros((X.shape[0], T.shape[1]))
        out = torch.zeros((X.shape[0], T.shape[1]))
        # out_rec = [out]
        out_rec = []
        # Generate candidates T_max vectors of w and b for selection
        # alpha = 0.25
        #
        # beta = 1
        for t in range(X.shape[1]):
            new_flt = self.alpha * flt + h2[:, t]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)
            # print('size of out rec is : '+str(len(out_rec)))

        out_rec = torch.stack(out_rec, dim=1)
        O = torch.sum(out_rec, 1)
        O = O.numpy()
        O = numpy_flops.softmax(self._flops_counter(), softmax_standardization, O)

        Error = numpy_flops.metric(self._flops_counter(), 'mean_squared_error',
                                   mean_squared_error, T, O)

        self.COST = Error
        return (O, Error)

    # get output
    def getOutput(self, X):
        H,spk_rec  = self.getH(X)
        O = torch.einsum("ab,bc->ac", (torch.Tensor(H).float(), torch.Tensor(self.Beta).float()))
        #h2 = torch.einsum("abc,cd->abd", (torch.Tensor(spk_rec).float(), torch.Tensor(self.Beta).float()))
        # flt = torch.zeros((X.shape[0], self.Beta.shape[1]))
        # out = torch.zeros((X.shape[0], self.Beta.shape[1]))
        # # out_rec = [out]
        # out_rec = []
        # for t in range(X.shape[1]):
        #     new_flt = self.alpha * flt + h2[:, t]
        #     new_out = self.beta * out + flt
        #
        #     flt = new_flt
        #     out = new_out
        #
        #     out_rec.append(out)
        # out_rec = torch.stack(out_rec, dim=1)
        #O = torch.sum(out_rec, 1)

        O = O.numpy()
        O = numpy_flops.softmax(self._flops_counter(), softmax_standardization, O)


        #print('in function getOutput, the size of O is: ' + str(O.shape))
        return O

    # Regression

    # get Label
    def getLabel(self, X):
        O = self.getOutput(X)
        (N, p) = O.shape
        ON = np.zeros((N, p))
        ind = np.argmax(O, axis=1)
        if p > 1:
            for i in range(0, N):
                ON[i, ind[i]] = 1
        else:
            for i in range(0, N):
                if O(i) > 0.50:
                    ON[i] = 1
        return ON

    # get accuracy
    def getAccuracy(self, X, T):
        O = self.getLabel(X)
        # Compute the confusion matrix
        conf_matrix = metrics.confusion_matrix(np.argmax(T, axis=1), np.argmax(O, axis=1))

        # Convert the confusion matrix values to percentages
        conf_matrix_percent = conf_matrix / np.sum(conf_matrix) * 100
        # Integer confusion-matrix sum is excluded; division and scaling float.
        numpy_flops.record(self._flops_counter(), 'confusion_percent', 2*conf_matrix.size)
        #rate = metrics.confusion_matrix(T.argmax(axis=1), O.argmax(axis=1))
        return (O, conf_matrix_percent)

    # Classification
    def classification(self, X, T, X2, T2):
        runtime = getattr(self, '_pp3_runtime', None)
        measured_round = 0
        #print('Y_train.size()' + str(T.size()))
        # X_train = torch.flatten(X_train, start_dim=1)
        #print('X_train.size()' + str(X.size()))
        X = np.array(X)
        T = np.array(T)
        E = T
        ErrorList = []
        RateList = []
        RateList2 = []
        timeList = []
        rate_r0 = [0]
        Error = 30
        error = 100
        rate = 0
        Rate = 0
        rate2 = 0
        Rate_0 =0
        Rate_new = 1/15
        training_time = 0

        while (self.L < self.L_max) and (Error > self.tol) and ((Rate_new - Rate) >= 0.001) :
            measured_round += 1
            if runtime:
                runtime.begin('train', measured_round, excluded='Python scalar loop/control arithmetic')
             #  and (torch.abs(torch.tensor(Rate_new) - torch.tensor(Rate)) >= 0.00)):
            #print(self.verbose)

            start = time.time()
            if (self.L != 0) and self.L % self.verbose == 0:

                Rate = Rate_new
                #print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f}\r'.format(
                    #self.L, Error, rate))
                print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f} \tACC-test:{:.4f} \ttime-loop:{:.4f}\r'.format(self.L, Error, rate, rate2,training_time))
                Rate_new = max(rate_r0)
                Rate_0 = rate

                rate_r0 = []
            # Search for candidate node / Hidden Parameters
            (w_L,  u_L, Flag) = self.sc_Search(X, E)
            if Flag == 1:
                # could not find enough node
                if runtime:
                    runtime.end()
                    runtime.epoch_row(measured_round, unit='growth_round_failed')
                break
            self.addNodes(w_L,  u_L)
            # print('Rate_new is'+str(Rate_new))
            # print('Rate is '+ str(Rate))
            # Calculate Beta/ Update all
            (otemp, Error) = self.upgradeSCN(X, T)
            O = self.getLabel(X)
            if runtime:
                runtime.end(self._accuracy(T, O), Error)
                runtime.begin('test', measured_round)
            O2 = self.getLabel(X2)

            rate = self._accuracy(T, O) # print(rate)
            rate_r0.append(rate)
            rate2 = self._accuracy(T2, O2)
            if runtime:
                runtime.end(rate2)
                runtime.epoch_row(measured_round, unit='growth_round')
            end = time.time()
            training_time = (end - start) / 3600
            # log
            ErrorList = update_list(ErrorList, Error, w_L.shape[1])
            RateList = update_list(RateList, rate, w_L.shape[1])
            RateList2 = update_list(RateList2, rate2, w_L.shape[1])
            timeList = update_list(timeList, training_time, w_L.shape[1])
        # Ratetest = torch.tensor(RateList2)
        #     if rate2 >= 0.975 and rate >= 0.975:
        #         break
        # Find the maximum value and its index using PyTorch functions
        Ratetest = torch.tensor(RateList2)
        if runtime:
            runtime.begin('final_refit')
        max_value = torch.max(Ratetest)
        index_of_max_value = torch.argmax(Ratetest)
        print('highest testing accuracy is: '+ str(max_value))
        suitW = self.W[:, 0:index_of_max_value]
        suitU = self.U[0:index_of_max_value, 0:index_of_max_value]
        self.W = suitW
        self.U = suitU
        newH,_ = self.getH(X)
        self.computeBeta(newH, T)
        print('index of max value is: ' + str(index_of_max_value))
        print('Suitable W.shape is: ' + str(suitW.shape))
        print('Suitable U.shape is: ' + str(suitU.shape))


        print('End Searching ...')
        print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f} \tACC-test:{:.4f}\r'.format(self.L, Error, rate, rate2))
        print('***************************************')
        self.printProperties()
        O2 = self.getLabel(X2)
        conf_matrix = metrics.confusion_matrix(np.argmax(T2, axis=1), np.argmax(O2, axis=1))
        rate2 = self._accuracy(T2, O2)
        print('测试阶段准确率', rate2,'测试混淆矩阵是', conf_matrix)

        if runtime:
            runtime.end(rate2)
        return ErrorList, RateList, RateList2, timeList, suitW, suitU, self.Beta
    def test_model(self, X2,T2, suitW,suitU,suitB):
        suitW = np.load(suitW)
        suitU = np.load(suitU)
        suitB = np.load(suitB)
        self.W, self.U, self.Beta = suitW, suitU, suitB
        O2 = self.getLabel(X2)
        conf_matrix = metrics.confusion_matrix(np.argmax(T2, axis=1), np.argmax(O2, axis=1))
        rate2 = self._accuracy(T2, O2)
        return conf_matrix, rate2

