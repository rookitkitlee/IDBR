import numpy as np
from bean.Param import Param
from handle.HandleStepLoadData import HandleStepLoadData
from handle.HandleStepGenerateParam import HandleStepGenerateParam
from handle.HandleStepTrainCFLink import HandleStepTrainCFLink
from handle.HandleStepMixOrder import HandleStepMixOrder
from handle.HandleStepSample import HandleStepSample
from handle.HandleStepMixOrder import HandleStepMixOrder

import random
random.seed(1)
import os
import sys


R, TrainU, TrainV, TrainR, TestU, TestV, TestR, ue, ie = HandleStepLoadData.read_wiki()
LKTrain, LKTest = HandleStepLoadData.read_wiki_lk(ue, ie)
p = HandleStepGenerateParam.execute(R, TrainU, TrainV, TrainR, TestU, TestV, TestR, 64, 64, 128)
p = HandleStepMixOrder.execute3(p)
p = HandleStepTrainCFLink.execute(p, 10001, LKTrain, LKTest, "WIKI")






