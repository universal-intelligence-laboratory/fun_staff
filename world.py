import torch
from torch import nn
import random
import torch.nn.functional as F
import numpy as np


class World(nn.Module):
    def __init__(self,x1,x2,x3,device):
        super(World, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.spices = [Spices(x1),Spices(x2),Spices(x3)]
        self.hell = []
        self.device = device

    def forward(self):
        for _ in range(3):
            self.main_procedure()
        x = self.get_limit()
        return x
    

    def representing(self):
        for x in self.spices:
            # #print(len(self.hell),len(x.data),len(x.dnas))
            for i,d in enumerate(x.data):
                x.dnas[i] = self.to_dna(d)


    def init(self):
        for x in self.spices:
            for d in x.data:
                x.dnas.append(self.to_dna(d))
                x.score.append(1e2)
            x.leader = x.data[0]

        

    def main_procedure(self):
        self.representing()
        self.score()
        self.die()
        self.mate()
        self.voting()


        
    def mate(self):
        for x in self.spices:
            #print(len(self.hell),"mate in",x.data.shape,len(x.dnas),len(x.score))
            leader = x.leader
            # mate with same spices
            new_borns = [x.data]
            for i,d in enumerate(x.data):
                new_bee = leader * 0.8 + d * 0.2
                if i % 20 ==0:
                    new_borns.append(new_bee.reshape([1,]+list(new_bee.shape)))
                    x.dnas.append(self.to_dna(new_bee))
                    x.score.append(1e2)
            x.data = torch.cat(tuple(new_borns), 0)

            new_borns = [x.data]
            # mate with death
            # if len(self.hell) > 0:
            for h in self.hell:
                new_bee = leader * 0.8 + h * 0.2
                new_borns.append(new_bee.reshape([1,]+list(new_bee.shape)))
                x.dnas.append(self.to_dna(new_bee))
                x.score.append(1e2)
            x.data = torch.cat(tuple(new_borns), 0)

            #print(len(self.hell),"mate out",x.data.shape,len(x.dnas),len(x.score))


        # if len(self.hell) > 0:
        self.hell = []

    def get_limit(self):
        limit = 0.0
        for x in self.spices:
            limit += min(x.score)
        return limit
        
    # def distance(self,x,his_leader):
    #     return F.mse_loss(x,his_leader)

    def voting(self):
        for x in self.spices:
            #print(len(self.hell),"voting in",x.data.shape,len(x.dnas),len(x.score))
            min_score = 1e20
            tmp_leader = None
            for i,score in enumerate(x.score):
                if score < min_score:
                    tmp_leader = x.data[i]

            x.leader = tmp_leader
            #print(len(self.hell),"voting out",x.data.shape,len(x.dnas),len(x.score))

    def score(self):
        for x in self.spices:
            #print(len(self.hell),"score in",x.data.shape,len(x.dnas),len(x.score))
            for i,dna in enumerate(x.dnas):
                x.score[i] = 0.0
                referrance = random.choices(x.dnas,k=10)
                for ref_dna in referrance:
                    x.score[i] += F.cosine_similarity(ref_dna,dna,dim=0)
            #print(len(self.hell),"score out",x.data.shape,len(x.dnas),len(x.score))

    def die(self):
        for x in self.spices:
            #print(len(self.hell),"die in",x.data.shape,len(x.dnas),len(x.score))
            die_idx = []

            for i,idx_score in enumerate(x.score):
                if idx_score > torch.mean(torch.Tensor(x.score)).to(self.device):
                    die_idx.append(i)
            
            die_idx = random.sample(die_idx,k=len(die_idx)//10 + 1)
            
            for i in die_idx:
                self.hell.append(x.data[i])

            if len(die_idx) > 0:
                # x.score.pop(die_idx)
                for index in sorted(die_idx, reverse=True):
                    del x.score[index]
                    del x.dnas[index]
                    # x.dna.pop(die_idx)
                    # x.data.pop(die_idx)
                    x.data = torch.cat([x.data[:index], x.data[index+1:]])
                    # 删除一个元素
                # x.dna.pop(die_idx)
            #print(len(self.hell),"die out",x.data.shape,len(x.dnas),len(x.score))
            # debugging 复杂逻辑网络的技巧：in & out log


    def to_dna(self,x):
        x = x.view(-1,)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x = F.dropout(F.relu(self.fc3(x)), training=self.training)
        return x


class Spices(nn.Module):
    def __init__(self,x):
        super(Spices, self).__init__()
        # n_population = x.size(0)
        # #print(len(self.hell),n_population)
        # self.data = list(torch.split(x, n_population, dim=0))
        self.data = x
        self.score = []
        self.dnas = []
        self.leader = None

