# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from math import sqrt
from scipy.special import expit
from sklearn import metrics
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from snnandfunction import spike_fn
from standardmatrxi import softmax_standardization
# plot  neurons
import time
from updatelist import *
class bdnn2(nn.Module):
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
                 nB=3,
                 verbose=10,
                 alpha=0.25,
                 beta=1
                 ):
        self.alpha = alpha
        self.beta = beta
        if isinstance(verbose, int):
            self.verbose = verbose
        if isinstance(L_max, int):
            self.L_max = L_max
            # if L_max > 5000:
            # self.verbose = 500  # does not need too many output
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
        # print('      b:{}'.format(self.b.shape))
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

    def inequalityEq(self, eq, gk, r_L):
        a = (((1 - r_L)) * eq.conj().T)
        ksi = ((eq.conj().T @ gk) ** 2) / (gk.conj().T @ gk) - \
              np.array(a) @ eq

        return ksi

    # Search for {WB,bB} of nB nodes
    def sc_Search(self, X, E0):

        Flag = 0
        WB = []
        bB = []
        UB = []
        d = X.shape[2]
        l = X.shape[1]
        m = E0.shape[1]

        C = []
        for Lambda in self.Lambdas:
            alpha = 0.25
            beta = 1
            # WW is d-by-T_max
            WT = Lambda * (2 * np.random.rand(d, self.T_max) - 1)
            UT = Lambda * (2 * np.random.randn(self.T_max, self.T_max) - 1)

            HT = np.zeros((X.shape[0], self.T_max))

            h_prev = np.zeros((X.shape[0], self.T_max))  # ng you have batch_size and hidden_size defined
            syn = torch.zeros((X.shape[0], self.T_max))
            mem = torch.zeros((X.shape[0], self.T_max))
            mem_rec = []
            spk_rec = []
            out = torch.zeros((X.shape[0], self.T_max))

            h1_from_inputs = torch.einsum("abc,cd->abd", (torch.tensor(X,dtype=torch.float32).float(), torch.tensor(WT,dtype=torch.float32).float()))

            # Loop through each time step
            for i in range(l):
                h1 = h1_from_inputs[:, i, :] + torch.einsum("ab,bc->ac",
                                                            (torch.tensor(out,dtype=torch.float32).float(), torch.tensor(UT,dtype=torch.float32).float()))
                out = spike_fn(mem - 1)
                rst = out.detach()
                new_syn = alpha * syn + h1
                new_mem = (beta * mem + syn) * (1.0 - rst)
                mem_rec.append(mem)
                spk_rec.append(out)
                mem = new_mem
                syn = new_syn

            spk_rec = torch.stack(spk_rec, dim=1)

            HT = torch.sum(spk_rec, 1)
            HT = HT.numpy()
            HT = HT / l


            for r_L in self.r:
                for t in range(0, self.T_max):
                    H_t = HT[:, t]
                    ksi_m = np.zeros((1, m), dtype=np.float64)

                    for i_m in range(0, m):
                        eq = E0[:, i_m].reshape(-1, 1)
                        gk = H_t.reshape(-1, 1)
                        ksi_m[0, i_m] = self.inequalityEq(eq, gk, r_L)

                    Ksi_t = np.sum(ksi_m, 0).reshape(-1, 1)  # values of ksi_m is equail to Ksi_t
                    if np.min(ksi_m) > 0:
                        if type(C) == list:
                            C = Ksi_t
                        else:
                            C = np.concatenate([C, Ksi_t], axis=1)

                        if type(WB) == list:
                            WB = WT[:, t].reshape(-1, 1)
                        else:
                            WB = np.concatenate(
                                (WB, WT[:, t].reshape(-1, 1)), axis=1)

                        if type(UB) == list:
                            UB = UT[:, t].reshape(-1, 1)
                        else:
                            UB = np.concatenate((UB, UT[:, t].reshape(-1, 1)), axis=1)

                nC = len(C)
                if nC >= self.nB:
                    break  # r loop
                else:
                    continue
            # end r
            if nC >= self.nB:
                break  # lambda loop
            else:
                continue

        if nC >= self.nB:

            I = C.argsort(axis=1)[::-1]
            I_nb = I[0, 0:self.nB]

            WB = WB[:, I_nb]
            UB = UB[I_nb, :][:, I_nb]

        if nC == 0 or nC < self.nB:
            Flag = 1

        print('Flag is' + str(Flag))
        return [WB, UB, Flag]

    # Add nodes to the previous model
    def addNodes(self, w_L, u_L):

        if type(self.W) == list:
            self.W = w_L
        else:

            self.W = torch.Tensor(np.concatenate((self.W, w_L), axis=1))
            print('In function addNodes the size of self.W is' + str(self.W.shape))

        if isinstance(self.U, list):
            self.U = u_L
        else:

            self.U = torch.block_diag(torch.tensor(self.U), torch.tensor(u_L))

        self.L = self.L + 1


    # ComputeBeta
    def computeBeta(self, H, T):

        Beta = np.linalg.pinv(H) @ T

        self.Beta = Beta

        return self.Beta


    # Output Matrix of hidden layer
    def getH(self, X):
        l = X.shape[1]
        syn = torch.zeros((X.shape[0], self.W.shape[1]))
        mem = torch.zeros((X.shape[0], self.W.shape[1]))
        mem_rec = []
        spk_rec = []
        out = torch.zeros((X.shape[0], self.W.shape[1]))
        h1_from_inputs = torch.einsum("abc,cd->abd", (torch.tensor(X,dtype=torch.float32).float(), torch.tensor(self.W,dtype=torch.float32).float()))
        alpha = 0.25
        beta = 1
        for i in range(l):
            h1 = h1_from_inputs[:, i, :] + torch.einsum("ab,bc->ac",
                                                        (torch.tensor(out,dtype=torch.float32).float(), torch.tensor(self.U,dtype=torch.float32).float()))
            out = spike_fn(mem - 1)
            rst = out.detach()
            new_syn = alpha * syn + h1
            new_mem = (beta * mem + syn) * (1 - rst)
            mem_rec.append(mem)
            spk_rec.append(out)
            mem = new_mem
            syn = new_syn
        spk_rec = torch.stack(spk_rec, dim=1)
        HT = torch.sum(spk_rec, 1)
        H = HT.numpy()
        has_negative = np.any(H < 0)
        print("有小于0的元素:", has_negative)
        H = H / l
        return H, spk_rec

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
        O = softmax_standardization(O)


        #print('in function getOutput, the size of O is: ' + str(O.shape))
        return O
    # Compute the Beta, Output, ErrorVector and Cost
    def upgradeSCN(self, X, T):
        H, spk_rec = self.getH(X)
        self.Beta = self.computeBeta(H, T)
        h2 = torch.einsum("abc,cd->abd", (torch.Tensor(spk_rec).float(), torch.Tensor(self.Beta).float()))
        flt = torch.zeros((X.shape[0], T.shape[1]))
        out = torch.zeros((X.shape[0], T.shape[1]))
        out_rec = []
        # Generate candidates T_max vectors of w and b for selection
        alpha = 0.25

        beta = 1
        for t in range(X.shape[1]):
            new_flt = alpha * flt + h2[:, t]
            new_out = beta * out + flt

            flt = new_flt
            out = new_out
            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)

        O = torch.sum(out_rec, 1)
        O = O.numpy()
        O = softmax_standardization(O)

        Error = mean_squared_error(T, O)

        self.COST = Error
        return (O, Error, self.Beta)

    def getLabel(self, X,Beta):
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
    def getAccuracy(self, X, T,Beta):
        O = self.getLabel(X,Beta)
        # Compute the confusion matrix
        conf_matrix = metrics.confusion_matrix(np.argmax(T, axis=1), np.argmax(O, axis=1))

        # Convert the confusion matrix values to percentages
        conf_matrix_percent = conf_matrix / np.sum(conf_matrix) * 100
        # rate = metrics.confusion_matrix(T.argmax(axis=1), O.argmax(axis=1))
        return (O, conf_matrix_percent)

    # Classification
    def classification(self, X, T,  X2, T2, w_path, u_path):

        X = np.array(X)
        T = np.array(T)

        ErrorList = []
        RateList = []
        RateList2 = []
        timeList = []
        rate_r0 = [0]
        rate_r0_te = [0]
        Error = 30
        error = 100
        rate = 0
        Rate = 0
        Rate_te = 0
        rate2 = 0
        Rate_new_te = 1 / 5
        Rate_new = 1 / 5
        training_time = 0
        oriW = np.load(w_path)
        oriU = np.load(u_path)
        self.W = oriW
        self.U = oriU
        a = self.U.shape[0]

        H,_ = self.getH(X)
        m = T.shape[1]
        r_L_max = self.r[-1]
        ksi_list = []
        filtered_indices = []
        for i in range(0, H.shape[1]):
            H_t = H[:, i]
            ksi_m = np.zeros((1, m), dtype=np.float64)
            for i_m in range(0, m):
                eq = T[:, i_m].reshape(-1, 1)
                gk = H_t.reshape(-1, 1)
                ksi_m[0, i_m] = self.inequalityEq(eq, gk, r_L_max)
            Ksi_i = np.sum(ksi_m, 0).reshape(-1, 1)#The shape of Ksi_i is (10,1)

            if np.min(ksi_m) > 0:
                Ksi_i_i = Ksi_i
                ksi_list.append(Ksi_i_i)
                filtered_indices.append(i)

                # print('filtered_indices is', filtered_indices)
        if len(ksi_list) > 0:
            ksi_rest = np.hstack(ksi_list)
        print('active neurons of pre-trained model', (filtered_indices))
        self.W = self.W[:, np.array(filtered_indices)]
        self.U = self.U[np.ix_(filtered_indices, filtered_indices)]

        E = T
        print('Shape of original W and U are: ' + str(self.W.shape) + str(self.U.shape))
        while (self.L < self.L_max) and (Error > self.tol) and ((Rate_new - Rate) >= 0.001) :
            print('L is'+str(self.L))
            print(self.verbose)

            start = time.time()
            if (self.L != 0) and self.L % self.verbose == 0:
                Rate = Rate_new
                Rate_te = Rate_new_te
                # print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f}\r'.format(
                # self.L, Error, rate))
                print(
                    '#L:{}\tRMSE:{:.4f} \tACC:{:.4f} \tACC-test:{:.4f} \ttime-loop:{:.4f}\r'.format(self.L, Error, rate,
                                                                                                    rate2,
                                                                                                    training_time))
                Rate_new = max(rate_r0)
                Rate_new_te = max(rate_r0_te)
                print('Rate_new_te is' + str(Rate_new_te))
                print('Rate is ' + str(Rate_te))
                rate_r0 = []
                rate_r0_te = []
            # Search for candidate node / Hidden Parameters

            (w_L, u_L, Flag) = self.sc_Search(X, E)
            print('***********************************')
            print('new rate, the X_shape is ' + str(X.shape) + 'E shape is ' + str(E.shape) + '/' + 'T shape is ' + str(
                T.shape))
            if Flag == 1:
                # could not find enough node
                break
            self.addNodes(w_L, u_L)

            # Calculate Beta/ Update all
            (otemp, Error,Beta) = self.upgradeSCN(X, T)
            O = self.getLabel(X,Beta)
            O2 = self.getLabel(X2,Beta)

            rate = metrics.accuracy_score(T, O)  # print(rate)
            rate_r0.append(rate)
            rate2 = metrics.accuracy_score(T2, O2)
            rate_r0_te.append(rate2)

            # print('rate2 is'+str(rate2.size()))
            end = time.time()
            training_time = (end - start) / 3600
            print('training time' + str(training_time))
            # log
            # 使用示例
            ErrorList = update_list(ErrorList, Error, w_L.shape[1])
            RateList = update_list(RateList, rate, w_L.shape[1])
            RateList2 = update_list(RateList2, rate2, w_L.shape[1])
            timeList = update_list(timeList, training_time, w_L.shape[1])

        Ratetest = torch.tensor(RateList2)

        # Find the maximum value and its index using PyTorch functions
        print(Ratetest.size())
        max_value = torch.max(Ratetest)
        index_of_max_value = torch.argmax(Ratetest)
        print('max test accuracy is:' + str(max_value))
        suitW = self.W[:, 0:index_of_max_value + a]
        suitU = self.U[0:index_of_max_value + a, 0:index_of_max_value + a]
        self.W, self.U = suitW, suitU
        (otemp, Error, Beta) = self.upgradeSCN(X, T)
        suitBeta = self.Beta
        print('shape of suitBeta and suitW are', suitBeta.shape, self.W.shape)
        print('index of max value is: ' + str(index_of_max_value))
        print('Suitable W.shape is: ' + str(suitW.shape))
        print('Suitable U.shape is: ' + str(suitU.shape))

        print('End Searching ...')
        print('#L:{}\tRMSE:{:.4f} \tACC:{:.4f} \tACC-test:{:.4f}\r'.format(self.L, Error, rate, rate2))
        print('***************************************')
        self.printProperties()


        return ErrorList, RateList, RateList2, timeList, suitW, suitU, suitBeta
