from collections import UserList
from pyexpat import model
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
from torch import nn
import math
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np
import operator
import heapq
import time
import pandas as pd
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support
from sklearn import metrics


def link_prediction(uvec, vvec, LKTrain, LKTest):# 链路预测

    TrainX = []
    TrainY = []
    TestX = []
    TestY = []

    for (u, v, y) in LKTrain:

        if u == -1 or v == -1:
            continue

        if u == -1:
            ur = [0 for _ in range(128)]
        else:
            ur = uvec[u]

        if v == -1:
            vr = [0 for _ in range(128)]
        else:
            vr = vvec[v]

        x = ur + vr
        # x = uvec[u] + vvec[v]
        TrainX.append(x)
        TrainY.append(y)

    for (u, v, y) in LKTest:

        if u == -1 or v == -1:
            continue

        if u == -1:
            ur = [0 for _ in range(128)]
        else:
            ur = uvec[u]

        if v == -1:
            vr = [0 for _ in range(128)]
        else:
            vr = vvec[v]

        x = ur + vr
        # x = uvec[u] + vvec[v]
        TestX.append(x)
        TestY.append(y)

    TrainX = pd.DataFrame(TrainX)
    TestX = pd.DataFrame(TestX)

    TrainX = TrainX.fillna(TrainX.mean())
    TestX = TestX.fillna(TestX.mean())

    lg = LogisticRegression(penalty='l2',C=0.001)
    lg.fit(TrainX, TrainY)
    lg_y_pred_test = lg.predict_proba(TestX)[:,1]
    fpr,tpr,thresholds = metrics.roc_curve(TestY, lg_y_pred_test)
    average_precision = average_precision_score(TestY, lg_y_pred_test)

    return metrics.auc(fpr,tpr), average_precision




