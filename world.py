import torch
from torch import nn
import random
import torch.nn.functional as F
import numpy as np


class World(nn.Module):
    def __init__(self,x1,x2,x3,device,hell_reborn=False):
        super(World, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.spices = [Spices(x1),Spices(x2),Spices(x3)]
        self.hell = []
        self.device = device
        self.hell_reborn = hell_reborn

    def forward(self):
        for _ in range(3):
            self.main_procedure()
        x = self.get_limit()
        return x
    

    def representing(self):
        for x in self.spices:
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
        self.ageing() # 所有样本一起老化（加score）
        self.die()
        self.mate()
        self.voting()


    def ageing(self):
        for x in self.spices:
            for i,_ in enumerate(x.score):
                x.score[i] = x.score[i] * 1.05
        
    def mate(self):
        for x in self.spices:
            #print(len(self.hell),"mate in",x.data.shape,len(x.dnas),len(x.score))
            leader = x.leader
            # mate with same spices
            new_borns = [x.data]
            for i,d in enumerate(x.data):
                new_bee = leader * 0.8 + d * 0.2
                if i % 2 ==0:
                    new_borns.append(new_bee.reshape([1,]+list(new_bee.shape)))
                    new_dna = self.to_dna(new_bee)
                    x.dnas.append(new_dna)
                    x.score.append(self.score_x(new_dna, x.dnas)) # 给新的样本配分
            x.data = torch.cat(tuple(new_borns), 0)

            if self.hell_reborn:
                new_borns = [x.data]
                # mate with death
                for h in self.hell:
                    new_bee = leader * 0.8 + h * 0.2
                    new_borns.append(new_bee.reshape([1,]+list(new_bee.shape)))
                    new_dna = self.to_dna(new_bee)
                    x.dnas.append(new_dna)
                    x.score.append(self.score_x(new_dna, x.dnas)) # 给新的样本配分
                x.data = torch.cat(tuple(new_borns), 0)

            #print(len(self.hell),"mate out",x.data.shape,len(x.dnas),len(x.score))


        self.hell = []

    def get_limit(self):
        limit = 0.0
        for x in self.spices:
            limit += min(x.score)
        return limit
        

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


    def score_x(self,new_dna, dnas):
        xscore = 0.0
        referrance = random.choices(dnas,k=10)
        for ref_dna in referrance:
            xscore += F.cosine_similarity(ref_dna,new_dna,dim=0)
            #print(len(self.hell),"score out",x.data.shape,len(x.dnas),len(x.score))
        return xscore

    def die(self):
        for x in self.spices:
            #print(len(self.hell),"die in",x.data.shape,len(x.dnas),len(x.score))
            die_idx = []

            for i,idx_score in enumerate(x.score):
                if idx_score > torch.mean(torch.Tensor(x.score)).to(self.device):
                    die_idx.append(i)
            
            
            if self.hell_reborn:
                hell_idx = random.sample(die_idx,k=len(die_idx)//4 + 1)
                
                for i in hell_idx:
                    self.hell.append(x.data[i])

            if len(die_idx) > 0:
                for index in sorted(die_idx, reverse=True):
                    del x.score[index]
                    del x.dnas[index]
                    x.data = torch.cat([x.data[:index], x.data[index+1:]])
                    # 删除一个元素
            #print(len(self.hell),"die out",x.data.shape,len(x.dnas),len(x.score))
            # debugging 复杂逻辑网络的技巧：in & out log


    def to_dna(self,x):
        x = x.view(-1,)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x = self.fc3(x)
        return torch.sigmoid(x)


class Spices(nn.Module):
    def __init__(self,x):
        super(Spices, self).__init__()
        self.data = x
        self.score = []
        self.dnas = []
        self.leader = None

