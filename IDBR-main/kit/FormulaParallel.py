import math
from math import log10, exp, log2
import numpy as np

class FormulaParallel:


    @staticmethod
    def KL(P, Q):
        
        V = P / Q
        W = np.log2(V)
        
        # 这一行可以是KL散度范围为 0~1
        W[W>1] = 1
        Z = P * W
        V = np.sum(Z, axis=1).reshape(-1, 1)
        V = FormulaParallel.check(V)
        return V


    # 高斯分布
    @staticmethod
    def GD(kl, va):
        
        V =  1.0 / (np.sqrt(2.0 * np.pi) * va) * np.exp( - kl * kl / 2.0 / va / va )
        V = FormulaParallel.check(V)
        return V


    @staticmethod
    def check(V):

        V[V <= 0] = 1e-6
        # for i in range(len(V)):
        #     if V[i][0] <= 0:
        #         V[i][0] = 1e-6

        return V
