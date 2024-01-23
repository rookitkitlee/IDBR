import numpy as np
from bean.Param import Param
from handle.HandleStepLoadData import HandleStepLoadData
from handle.HandleStepGenerateParam import HandleStepGenerateParam
from handle.HandleStepTrainCFRec import HandleStepTrainCFRec
from handle.HandleStepMixOrder import HandleStepMixOrder


import random
random.seed(1)
import os
import sys


R, TrainU, TrainV, TrainR, TestU, TestV, TestR = HandleStepLoadData.read_dblp()
p = HandleStepGenerateParam.execute(R, TrainU, TrainV, TrainR, TestU, TestV, TestR, 64, 64, 128)
p = HandleStepMixOrder.execute3(p)
p = HandleStepTrainCFRec.execute(p, 20001, "DBLP")





