from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from torch import nn
import math
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np
from torch.autograd import Variable
from bean.Param import Param
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture 

class CFRec(nn.Module):

    def __init__(self, p: Param, device):
        super(CFRec, self).__init__()

        self.p = p
        self.device = device

        self.u = nn.Parameter(self.creatRandomParam(p.m, p.vecter_size), requires_grad=True).to(self.device)
        self.v = nn.Parameter(self.creatRandomParam(p.n, p.vecter_size), requires_grad=True).to(self.device)  

        self.m_t = torch.LongTensor(self.p.TrainU).to(self.device)
        self.n_t = torch.LongTensor(self.p.TrainV).to(self.device)

        # self.uc = None
        # self.vc = None

        self.u2c = None
        self.v2c = None

        self.MM = None
        self.NN = None

        self.MM2 = None
        self.NN2 = None

    def creatRandomParam(self, x, y):
        vu = np.random.random([x, y])
        vu = preprocessing.normalize(vu, norm='l2')# 再转成tensor
        vu = torch.from_numpy(vu).to(self.device)
        return vu


    def derive(self):
        return self.u.cpu().detach().numpy(), self.v.cpu().detach().numpy()

    def getLikeHoodMatrixRate(self):
        outputs = torch.mm(self.u, self.v.t())
        return outputs.cpu().detach().numpy()

    def normalization_torch(self, data):
        # return data / data.sum(dim=1).reshape(-1, 1)
        return data / torch.sum(data, 1).unsqueeze(-1)

    def normalization_numpy(self, data):
        return data / np.sum(data, axis=1).reshape(-1,1) 
    

    def generate_2_loss(self, o, t):

        os = torch.sigmoid(o)
        ols = F.log_softmax(os.float(), dim=-1)
        ts = F.softmax(t, dim=-1)

        kl = F.kl_div(ols.float(), ts.float(), reduction='sum')
        return kl


    def generate_3_loss(self, o, y, t):

        ON = self.normalization_torch(o)
        ONR = torch.mm(y, ON).float()
        return F.kl_div(F.log_softmax(ONR, dim=-1).float(), F.softmax(t, dim=-1).float(), reduction='sum')


    def generate_4_loss(self, o, t):

        ols = F.log_softmax(o.float(), dim=-1)
        ts = F.softmax(t, dim=-1)

        kl = F.kl_div(ols.float(), ts.float(), reduction='sum')
        return kl


    def step1_init(self):

        self.MM = torch.from_numpy(self.p.mm).float().to(self.device)
        self.NN = torch.from_numpy(self.p.nn).float().to(self.device)


    def step1_loss(self):

        # loss1
        pu = self.u.index_select(0, self.m_t).float()
        pv = self.v.index_select(0, self.n_t).float()
        outputs = torch.mul(pu, pv)
        outputs = torch.sum(outputs, dim=1)
        loss1 = - torch.mean(F.logsigmoid(outputs))


        u_u = torch.mm(self.u, self.u.t()).float() # m * m
        v_v = torch.mm(self.v, self.v.t()).float() # n * n

        kl_mm = self.generate_2_loss(u_u, self.MM)
        kl_nn = self.generate_2_loss(v_v, self.NN)
        
        # return loss1

        return loss1 + kl_mm + kl_nn 



    def step2_init(self):

        print('component start')

        u = self.u.cpu().detach().numpy()
        v = self.v.cpu().detach().numpy()

        mu = GaussianMixture(n_components=self.p.o)
        mv = GaussianMixture(n_components=self.p.p)
        mu.fit(u)
        mv.fit(v)

        u2uc = mu.predict_proba(u)  # m * uc
        v2vc = mv.predict_proba(v)  # n * vc
        r = self.normalization_numpy(self.p.R) # m * n

        u2vc = np.matmul(r, v2vc) # m * uc
        v2uc = np.matmul(r.T, u2uc) # n * vc

        MM2 = np.matmul(self.normalization_numpy(u2vc), self.normalization_numpy(u2vc.T))
        MM4 = np.matmul(MM2, MM2)
        MM = MM2 + MM4

        NN2 = np.matmul(self.normalization_numpy(v2uc), self.normalization_numpy(v2uc.T))
        NN4 = np.matmul(NN2, NN2)
        NN = NN2 + NN4

        self.MM = torch.from_numpy(self.p.mm).float().to(self.device)
        self.NN = torch.from_numpy(self.p.nn).float().to(self.device)
 
        self.MM2 = torch.from_numpy(MM).float().to(self.device)
        self.NN2 = torch.from_numpy(NN).float().to(self.device)

        self.uc = torch.from_numpy(mu.means_).to(self.device)
        self.vc = torch.from_numpy(mv.means_).to(self.device)

        print('component end')

    def step2_loss(self):


        # loss1
        pu = self.u.index_select(0, self.m_t).float()
        pv = self.v.index_select(0, self.n_t).float()
        outputs = torch.mul(pu, pv)
        outputs = torch.sum(outputs, dim=1)
        loss1 = - torch.mean(F.logsigmoid(outputs))


        u_u = torch.mm(self.u, self.u.t()).float() # m * m
        v_v = torch.mm(self.v, self.v.t()).float() # n * n
        kl_mm = self.generate_2_loss(u_u, self.MM)
        kl_nn = self.generate_2_loss(v_v, self.NN)

        kl_mm2 = self.generate_2_loss(u_u, self.MM2)
        kl_nn2 = self.generate_2_loss(v_v, self.NN2)

        # ------------------------------------------


        u_v = torch.mm(self.u, self.v.t()).float() # m * m
        v_u = torch.mm(self.v, self.u.t()).float() # n * m

        u_uc = torch.mm(self.u, self.uc.t()).float() # m * 150
        v_vc = torch.mm(self.v, self.vc.t()).float() # n * 150
        u_vc = torch.mm(self.u, self.vc.t()).float() # m * 150
        v_uc = torch.mm(self.v, self.uc.t()).float() # n * 150

        u_vc_v = torch.mm(u_vc, self.normalization_torch(v_vc).t()) # m * n
        v_uc_u = torch.mm(v_uc, self.normalization_torch(u_uc).t()) # n * m

        kl_t_mn = self.generate_4_loss(u_vc_v, u_v)
        kl_t_nm = self.generate_4_loss(v_uc_u, v_u)



        return loss1  + kl_mm + kl_nn  + 0.1 * kl_mm2 + 0.1 * kl_nn2 + 0.001 * kl_t_mn + 0.001 * kl_t_nm 

        # return loss1  
        # + 0.1 * kl_mm2 + 0.1 * kl_nn2
        
