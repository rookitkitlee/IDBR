import numpy as np
from bean.Param import Param


class HandleStepGenerateParam:

    # R 交互矩阵
    # o 用户的聚类中心个数
    # p 项目的聚类中心个数
    @staticmethod
    def execute(R, TrainU, TrainV, TrainR, TestU, TestV, TestR, o, p, vs):

        print('HandleStepGenerateParam')

        # m 用户的个数
        m = len(R)
        # n 项目的个数
        n = len(R[0])


        param = Param(R, m, n, o, p, vs)

        param.TrainU = TrainU
        param.TrainV = TrainV
        param.TrainR = TrainR
        param.TestU = TestU
        param.TestV = TestV
        param.TestR = TestR

        RP = np.zeros([m, n])
        RP[R > 0] = 1

        param.Pu = np.sum(RP, axis=1) / np.sum(RP)
        param.Pv = np.sum(RP, axis=0) / np.sum(RP)
        param.P = np.append(param.Pu, param.Pv)

        # print('!!!!!!!!!!!!!!!!!')
        # print(param.Pu)
        # print(param.Pv)
        # print(param.P)

        return param