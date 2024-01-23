from bean.Param import Param
from handle.CFRec import CFRec
import torch
import torch.optim as optim
import torch.nn.functional as F
import kit.EvaluatingIndicator as EV
import datetime
from time import strftime
from kit.IOKit import IOKit
import json

class HandleStepTrainCFRec:

    @staticmethod
    def execute(p:Param, poch, datadir):

        print('HandleStepTrainCFRec')
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = CFRec(p, device).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, nesterov=True)


        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d %H:%M:%S")
        log = open("Data/"+ datadir +"/log"+str(date)+".txt", mode='w')
        print("Data/"+ datadir +"/log"+str(date)+".txt")
        log.write('start \n')
        log.flush()


        net.step1_init()
        for i in range(2001):

            optimizer.zero_grad()
            loss = net.step1_loss() 
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print('step1 poch: ' +str(i)+ ' loss: ' + str(loss))
                log.write('step1 poch: ' +str(i)+ ' loss: ' + str(loss) + '\n')

                f1, map, mrr, mndcg = EV.top_N_by_matrixRate(p.TestU, p.TestV, p.TestR, net.getLikeHoodMatrixRate(),  10)        
                print("f1:", f1, "mndcg:", mndcg, "map:", map, "mrr:", mrr)
                log.write("f1:" + str(f1) + " mndcg:" + str(mndcg) + " map:" + str(map) +  " mrr:"+str(mrr) + '\n')
                log.flush()


        
        for i in range(poch):

            if i % 2000 == 0:
                net.step2_init()


            optimizer.zero_grad()
            loss = net.step2_loss() 
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print('step2 poch: ' +str(i)+ ' loss: ' + str(loss))
                log.write('step2 poch: ' +str(i)+ ' loss: ' + str(loss) + '\n')

                f1, map, mrr, mndcg = EV.top_N_by_matrixRate(p.TestU, p.TestV, p.TestR, net.getLikeHoodMatrixRate(),  10)        
                print("f1:", f1, "mndcg:", mndcg, "map:", map, "mrr:", mrr)
                log.write("f1:" + str(f1) + " mndcg:" + str(mndcg) + " map:" + str(map) +  " mrr:"+str(mrr) + '\n')
                log.flush()


        log.close()
        return p 