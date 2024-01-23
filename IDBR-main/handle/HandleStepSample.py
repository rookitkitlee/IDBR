from bean.Param import Param
import random

class HandleStepSample:


    @staticmethod
    def sample(T):
        
        ui = {}
        vi = {}
        data1 = []


        len1 = T.shape[0]
        len2 = T.shape[1]
        for i in range(len1):
            for j in range(len2):
                r = T[i][j]
                if r == 0:
                    continue
                if i not in ui.keys():
                    ui[i] = []
                if j not in vi.keys():
                    vi[j] = []
                data1.append((i, j, r))  
                ui[i].append((i, j, r))  
                vi[j].append((i, j, r))  

        result = []
        for d in data1:
            u = d[0]
            v = d[1]
            r = d[2]

            uds = ui[u]
            if len(uds) > 1:
                for _ in range(2):
                    index = random.randint(0, len(uds)-1)
                    t = uds[index]
                    if t[2] == r:
                        continue
                    tu = t[0]
                    tv = t[1]
                    tr = t[2]
                    if r > tr:
                        result.append((u, v, tu, tv))
                    else:
                        result.append((tu, tv, u, v))
            
            vds = vi[v]
            if len(vds) > 1:
                for _ in range(2):
                    index = random.randint(0, len(vds)-1)
                    t = vds[index]
                    if t[2] == r:
                        continue
                    tu = t[0]
                    tv = t[1]
                    tr = t[2]
                    if r > tr:
                        result.append((u, v, tu, tv))
                    else:
                        result.append((tu, tv, u, v))

        return result


    @staticmethod
    def execute(p:Param):

        print('HandleStepSample')

        p.CR = HandleStepSample.sample(p.R)
        # p.CM = HandleStepSample.sample(p.M)
        # p.CN = HandleStepSample.sample(p.N)
        return p