import numpy as np
from kit.IOKit import IOKit
from bean.UserEnum import UserEnum
from bean.ItemEnum import ItemEnum


class HandleStepLoadData:

    @staticmethod
    def read_dblp():

        print('read_dblp')

        
        # 读训练数据
        ds = []
        for line in IOKit.read_txt("Data/DBLP/train.txt"):     
            dd = line.split("	")
            d = (int(dd[0]), int(dd[1]), int(dd[2]))
            ds.append(d)
        

        user, item = set(), set()
        for w in ds:
            u, v , r = w
            user.add(u)
            item.add(v)
        user_list = list(user)
        item_list = list(item)

        ue = UserEnum()
        ie = ItemEnum()

        for u in user_list:
            ue.addEnum(u)

        for i in item_list:
            ie.addEnum(i)

        # 读测试数据
        TrainU = []
        TrainV = []
        TrainR = {}

        matrix = np.zeros((len(user_list), len(item_list)))
        for w in ds:
            u, v , r = w
            u = ue.enum[u]
            v = ie.enum[v]

            matrix[u][v] = r

            TrainU.append(u)
            TrainV.append(v)
            if TrainR.get(u) is None:
                TrainR[u] = {}
            TrainR[u][v] = r


        # 读测试数据
        TestU = set()
        TestV = set()
        TestR = {}
        for line in IOKit.read_txt("Data/DBLP/test.txt"):    
            # print(line) 

            line = line.replace(" ", "	")
            line = line.replace("\n", "")
            dd = line.split("	")

            u = int(dd[0])
            v = int(dd[1])
            r = int(dd[2])

            if u not in ue.enum or v not in ie.enum:
                continue

            u = ue.enum[u]
            v = ie.enum[v]

            TestU.add(u)
            TestV.add(v)
            if TestR.get(u) is None:
                TestR[u] = {}
            TestR[u][v] = r

        # print('-----------------')
        # print(len(TU))

        return matrix, TrainU, TrainV, TrainR, TestU, TestV, TestR

  
    def read_wiki():

        print('read_wiki')
        
        # 读训练数据
        ds = []
        for line in IOKit.read_txt("Data/WIKI/rating_train.dat"):     
            dd = line.split("\t")
            # d = (int(dd[0]), int(dd[1]), int(dd[2]))
            d = (dd[0].strip(), dd[1].strip(), int(dd[2]))
            ds.append(d)
        

        user, item = set(), set()
        for w in ds:
            u, v , r = w
            user.add(u)
            item.add(v)
        user_list = list(user)
        item_list = list(item)

        ue = UserEnum()
        ie = ItemEnum()

        for u in user_list:
            ue.addEnum(u)

        for i in item_list:
            ie.addEnum(i)

        # 读测试数据
        TrainU = []
        TrainV = []
        TrainR = {}

        matrix = np.zeros((len(user_list), len(item_list)))
        for w in ds:
            u, v , r = w
            u = ue.enum[u]
            v = ie.enum[v]

            matrix[u][v] = r

            TrainU.append(u)
            TrainV.append(v)
            if TrainR.get(u) is None:
                TrainR[u] = {}
            TrainR[u][v] = r


        # 读测试数据
        TestU = set()
        TestV = set()
        TestR = {}
        for line in IOKit.read_txt("Data/WIKI/rating_test.dat"):    
            # print(line) 

            line = line.replace(" ", "	")
            line = line.replace("\n", "")
            dd = line.split("\t")

            u = dd[0].strip()
            v = dd[1].strip()
            r = int(dd[2])

            if u not in ue.enum or v not in ie.enum:
                continue

            u = ue.enum[u]
            v = ie.enum[v]

            TestU.add(u)
            TestV.add(v)
            if TestR.get(u) is None:
                TestR[u] = {}
            TestR[u][v] = r

        return matrix, TrainU, TrainV, TrainR, TestU, TestV, TestR, ue, ie

    @staticmethod
    def read_wiki_lk(ue, ie):

        print('read_wiki_lk')

        train = []
        test = []

        
        # 读训练数据
        for line in IOKit.read_txt("Data/WIKI/case_train.dat"):     
            dd = line.split("\t")
            # d = (int(dd[0]), int(dd[1]), int(dd[2]))

            if dd[0].strip() in ue.enum.keys():
                uid = ue.enum[dd[0].strip()]
            else:
                uid = -1

            if dd[1].strip() in ie.enum.keys():
                vid = ie.enum[dd[1].strip()]
            else:
                vid = -1

            d = (uid, vid, int(dd[2]))
            train.append(d)

            # if dd[0].strip() in ue.enum.keys() and  dd[1].strip() in ie.enum.keys():
            #     d = (ue.enum[dd[0].strip()], ie.enum[dd[1].strip()], int(dd[2]))
            #     train.append(d)




        # 读训练数据
        for line in IOKit.read_txt("Data/WIKI/case_test.dat"):     
            dd = line.split("\t")
            # d = (int(dd[0]), int(dd[1]), int(dd[2]))

            # if dd[0].strip() in ue.enum.keys() and dd[1].strip() in ie.enum.keys():
            #     d = (ue.enum[dd[0].strip()], ie.enum[dd[1].strip()], int(dd[2]))
            #     test.append(d)



            if dd[0].strip() in ue.enum.keys():
                uid = ue.enum[dd[0].strip()]
            else:
                uid = -1

            if dd[1].strip() in ie.enum.keys():
                vid = ie.enum[dd[1].strip()]
            else:
                vid = -1

            d = (uid, vid, int(dd[2]))
            test.append(d)

        return train, test