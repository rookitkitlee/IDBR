import numpy as np

class Param:

    def __init__(self, R,  m, n, o, p, vs) -> None:

        self.R = R
   
        self.m = m
        self.n = n
        self.o = o
        self.p = p

        self.vecter_size = vs

        self.RI = np.zeros(self.R.shape)
        self.RI[self.R > 0] = 1
        self.RN = self.RI.sum()
        self.RS = self.R.sum(axis=1)


        RIup = np.append(np.zeros([self.m, self.m]), self.RI, axis=1)
        self.BigRI = np.append(RIup, np.zeros([self.n, self.m + self.n]), axis=0)
 

        self.M = None
        self.N = None

        self.TrainU = None
        self.TrainV = None
        self.TrainR = None

        self.TestU = None
        self.TestV = None
        self.TestR = None

        self.MI = None
        self.MN = None

        self.NI = None
        self.NN = None

        self.CR = None
        self.CM = None
        self.CN = None

        self.Mix = None
        self.MixCombination = None

        self.MixReal = None
        self.Mixes = []
    
        self.Pu = None
        self.Pv = None
        self.P = None

        self.MixUV = []
        self.MixVU = []
        self.MixUU = []
        self.MixVV = []

        self.VMixUV = None
        self.VMixVU = None
        self.VMixUU = None
        self.VMixVV = None

        self.mm = None
        self.nn = None
        self.mn = None
        self.nm = None

        self.xbm = None

    
    def show(self):

        print("--------------------------------------")
        print(self.distribution_user[0][0:4])
        print(self.distribution_user[500][0:4])
        print(self.distribution_user[1000][0:4])