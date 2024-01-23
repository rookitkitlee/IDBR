from bean.Param import Param
from handle.CFLink import CFLink
import torch
import torch.optim as optim
import torch.nn.functional as F
import kit.EvaluatingIndicator as EV
import datetime
from time import strftime
from kit.EVLink import link_prediction

class HandleStepTrainCFLink:

    @staticmethod
    def execute(p:Param, poch, LKTrain, LKTest, datadir):

        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        net = CFLink(p, device).to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, nesterov=True)

        date = datetime.datetime.now()
        date = date.strftime("%Y-%m-%d %H:%M:%S")
        log = open("Data/"+datadir+"/log"+str(date)+".txt", mode='w')
        print("Data/"+datadir+"/log"+str(date)+".txt")
        log.write('start \n')
        log.write('all \n')
        log.flush()


        net.step1_init()
        for i in range(201):

            optimizer.zero_grad()
            loss = net.step1_loss() 
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('step1 poch: ' +str(i)+ ' loss: ' + str(loss))
                log.write('step1 poch: ' +str(i)+ ' loss: ' + str(loss) + '\n')

                uvec, vvec = net.getEmbedding()
                s1,s2 = link_prediction(uvec, vvec, LKTrain, LKTest)
                print("AUC-ROC = ",s1," AUC-PR = ",s2)
                log.write('step1 likehood \n')
                log.write("AUC-ROC:" + str(s1) + " AUC-PR:" + str(s2)  + '\n')
                log.flush()

                # f1, map, mrr, mndcg = EV.top_N_by_matrixRate(p.TestU, p.TestV, p.TestR, net.getLikeHoodMatrixRate(), 10)        
                # print('step1 likehood')
                # print("f1:", f1, "mndcg:", mndcg, "map:", map, "mrr:", mrr)
                # log.write('step1 likehood \n')
                # log.write("f1:" + str(f1) + " mndcg:" + str(mndcg) + " map:" + str(map) +  " mrr:"+str(mrr) + '\n')
                # log.flush()

        
        for i in range(poch):

            if i % 500 == 0:
                net.step2_init()

            optimizer.zero_grad()
            loss = net.step2_loss() 
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('step2 poch: ' +str(i)+ ' loss: ' + str(loss))
                log.write('step2 poch: ' +str(i)+ ' loss: ' + str(loss) + '\n')

                uvec, vvec = net.getEmbedding()
                s1,s2 = link_prediction(uvec, vvec, LKTrain, LKTest)
                print("AUC-ROC = ",s1," AUC-PR = ",s2)
                log.write('step1 likehood \n')
                log.write("AUC-ROC:" + str(s1) + " AUC-PR:" + str(s2)  + '\n')
                log.flush()


        log.close()
        return p 