from bean.Param import Param
import numpy as np

class HandleStepMixOrder:


    @staticmethod
    def normalization(data):
        return data / np.sum(data, axis=1).reshape(-1,1) 


    @staticmethod
    def execute(p:Param):
       
        print('HandleStepMixOrder -----')
       
        Pmm = np.zeros([p.m, p.m])
        Pnn = np.zeros([p.n, p.n])
        Pmn = np.zeros([p.m, p.n])
        Pnm = np.zeros([p.n, p.m])

        #
        RNu2v1 = HandleStepMixOrder.normalization(p.R)          # m * n
        RNv2u1 = HandleStepMixOrder.normalization(p.R.T) 
        Pmn = Pmn + RNu2v1 * 1
        Pnm = Pnm + RNv2u1 * 1

        # 2
        MM2 = np.matmul(RNu2v1, RNv2u1)  # m * m
        NN2 = np.matmul(RNv2u1, RNu2v1)  # n * n
        Pmm = Pmm + MM2 * 1
        Pnn = Pnn + NN2 * 1

        # 3
        RNu2v3 = np.matmul(MM2, RNu2v1)
        RNv2u3 = np.matmul(NN2, RNv2u1)
        Pmn = Pmn + RNu2v3 * 1
        Pnm = Pnm + RNv2u3 * 1

        # 4
        MM4 = np.matmul(MM2, MM2)
        NN4 = np.matmul(NN2, NN2)
        Pmm = Pmm + MM4 * 1
        Pnn = Pnn + NN4 * 1

        # # 5
        # RNu2v5 = np.matmul(MM4, RNu2v1)
        # RNv2u5 = np.matmul(NN4, RNv2u1)
        # Pmn = Pmn + RNu2v5 * 1
        # Pnm = Pnm + RNv2u5 * 1

        # # 6
        # MM6 = np.matmul(MM4, MM2)
        # NN6 = np.matmul(NN4, NN2)
        # Pmm = Pmm + MM6
        # Pnn = Pnn + NN6

        
        p.mm = Pmm
        p.nn = Pnn
        p.mn = Pmn
        p.nm = Pnm

        return p


    @staticmethod
    def execute2(p:Param):
       
        print('HandleStepMixOrder -----')
       
        Pmm = np.zeros([p.m, p.m])
        Pnn = np.zeros([p.n, p.n])
        Pmn = np.zeros([p.m, p.n])
        Pnm = np.zeros([p.n, p.m])

        #
        RNu2v1 = HandleStepMixOrder.normalization(p.R)          # m * n
        RNv2u1 = HandleStepMixOrder.normalization(p.R.T) 
        Pmn = Pmn + RNu2v1 * 1
        Pnm = Pnm + RNv2u1 * 1

        # 2
        MM2 = np.matmul(RNu2v1, RNv2u1)  # m * m
        NN2 = np.matmul(RNv2u1, RNu2v1)  # n * n
        Pmm = Pmm + MM2 * 1
        Pnn = Pnn + NN2 * 1

        # 3
        RNu2v3 = np.matmul(MM2, RNu2v1)
        RNv2u3 = np.matmul(NN2, RNv2u1)
        Pmn = Pmn + RNu2v3 * 1
        Pnm = Pnm + RNv2u3 * 1

        # 4
        MM4 = np.matmul(MM2, MM2)
        NN4 = np.matmul(NN2, NN2)
        Pmm = Pmm + MM4 * 1
        Pnn = Pnn + NN4 * 1

        # # 5
        # RNu2v5 = np.matmul(MM4, RNu2v1)
        # RNv2u5 = np.matmul(NN4, RNv2u1)
        # Pmn = Pmn + RNu2v5 * 1
        # Pnm = Pnm + RNv2u5 * 1

        # # 6
        # MM6 = np.matmul(MM4, MM2)
        # NN6 = np.matmul(NN4, NN2)
        # Pmm = Pmm + MM6
        # Pnn = Pnn + NN6

        
        p.mm = Pmm
        p.nn = Pnn
        p.mn = Pmn
        p.nm = Pnm

        return p

    @staticmethod
    def execute3(p:Param):
       
        print('HandleStepMixOrder -----')

        MM2 = HandleStepMixOrder.normalization(np.matmul(p.R, p.R.T))
        MM4 = np.matmul(MM2, MM2)
        MM = MM2 + MM4

        NN2 = HandleStepMixOrder.normalization(np.matmul(p.R.T, p.R))
        NN4 = np.matmul(NN2, NN2)
        NN = NN2 + NN4

        p.mm = MM
        p.nn = NN

        return p


    @staticmethod
    def execute4(p:Param):
       
        print('HandleStepMixOrder -----')

        MM2 = HandleStepMixOrder.normalization(np.matmul(p.R, p.R.T))
        MM4 = np.matmul(MM2, MM2)
        MM6 = np.matmul(MM4, MM2)
        MM8 = np.matmul(MM6, MM2)
        MM10 = np.matmul(MM8, MM2)
        MM = MM2 + MM4 + MM6 + MM8 + MM10

        NN2 = HandleStepMixOrder.normalization(np.matmul(p.R.T, p.R))
        NN4 = np.matmul(NN2, NN2)
        NN6 = np.matmul(NN4, NN2)
        NN8 = np.matmul(NN6, NN2)
        NN10 = np.matmul(NN8, NN2)
        NN = NN2 + NN4 + NN6 + NN8 + NN10

        p.mm = MM
        p.nn = NN

        return p
