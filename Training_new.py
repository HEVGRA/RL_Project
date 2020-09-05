import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import DQN
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from sklearn import preprocessing
from collections import namedtuple

file = open('Data\data_20.txt', 'r', encoding='UTF-8') 
line = file.readlines()

num_node = int(line[0])
num_edge = int(line[1])
num_agent = int(line[num_node + num_edge + 2])
constraint = int(line[num_node + num_edge + num_agent + 3])
maxspeed = 0 
cost = 0
lists = "Model\saved_"

class Node:
    def __init__(self, pos, number):
        self.pos = pos
        self.number = number
        self.connected_node = []        
        self.in_commu_range = []        #溝通範圍(constraint)內的node
        self.all_ag_here = []           #在這個node上的agent
        
class Edge:
    def __init__(self, distance, number):
        self.ox = 'x'
        self.distance = distance        
        self.number = number
        self.count = 0
        
class Agent:
    def __init__(self, cur, speed, number):
        self.currnode_ori = cur
        self.currnode = cur         
        self.togonode = cur         
        self.lastedge = 0
        self.togoedge = 0
        self.curedge_length = 0     
        self.step = 0
        self.speed = speed
        self.num = number
        self.historyaction = []
        self.reward = 0
        self.start = cur
        self.edgeLengthInfo = []
        self.alreadyVisitInfo = []
        self.edgeTotalConnectMap = [[0]*num_edge for i in range(num_edge)]
        self.edgeTotalConnectInfo = []
        self.totalAgentMap = [[0]*2 for i in range(num_edge)]
        self.totalAgentInfo = []
        self.edgeCountInfo = []
        for i in range(num_edge):
            self.edgeLengthInfo.append(0)
            self.alreadyVisitInfo.append(0)
            self.edgeTotalConnectInfo.append(0)
            self.totalAgentInfo.append(0)
            self.edgeCountInfo.append(0)
        self.featureUpdate = []
        for i in range(num_agent): 
            self.featureUpdate.append([])

node_ALL = []
edge_ALL = {}
agent_ALL = []

for i in range(num_node):
    k = i + 2
    line[k] = line[k].split()
    for j in range(len(line[k])): 
        line[k][j] = int(line[k][j])
    l = Node((line[k][1], line[k][2]), line[k][0])
    node_ALL.append(l)

for i in range(num_edge):
    k = num_node + i + 2
    line[k] = line[k].split()
    for j in range(len(line[k])): 
        line[k][j] = int(line[k][j])
    l = Edge(line[k][2], i)
    line[k].pop()
    edge_ALL[tuple(line[k])] = l 
    start = line[k][0]
    end = line[k][1]
    node_ALL[start].connected_node.append(end)   
    node_ALL[end].connected_node.append(start)

for i in range(num_agent):
    k = num_node + num_edge + i + 3
    line[k] = line[k].split()
    for j in range(len(line[k])): 
        line[k][j] = int(line[k][j])
    l = Agent(int(line[k][1]), int(line[k][2]), int(line[k][0]))
    agent_ALL.append(l)
    if(maxspeed < int(line[k][2])): maxspeed = int(line[k][2])
    node_ALL[l.currnode].all_ag_here.append(i)

#算哪些node在溝通範圍(constraint)內
def cal_dis(a,b):    return np.sqrt(np.square(abs(a.pos[0]-b.pos[0]))+np.square(abs(a.pos[1]-b.pos[1])))
for i in range(num_node):
    for j in range(num_node):
        if(cal_dis(node_ALL[i],node_ALL[j]) <= constraint): node_ALL[i].in_commu_range.append(j)

def find_edge(a,b):
    if tuple([a,b]) in edge_ALL:   return tuple([a,b])
    else: return tuple([b,a])

# 特徵矩陣 (todo)

num_feature = 3
def feature_matrix(ag):
    X = np.zeros((num_node, num_feature))
    for k in node_ALL[ag.currnode].connected_node:
        ed = edge_ALL[find_edge(ag.currnode,k)].number
        # 距離
        if ag.edgeLengthInfo[ed] != 0:            
            X[k][0] = ag.edgeLengthInfo[ed]
        # 被幾個edge走到
        X[k][1] = ag.edgeTotalConnectInfo[ed]
        # 此edge被走過幾次
        X[k][2] = ag.edgeCountInfo[ed]
        # # 兩邊的node上共有幾個agent（下一步可能會走到）
        # X[k][3] = ag.totalAgentInfo[ed]
        # # 有沒有被走過
        # if ag.alreadyVisitInfo[ed] == 1: X[k][4] = 0.2
        # elif ag.alreadyVisitInfo[ed] == 0: X[k][4] = 0.8
    X = np.around((X), decimals=3)
    return X

def update_info():
    for u in range(num_agent):
        for give in agent_ALL:
            for receive in agent_ALL:
                if receive.currnode in node_ALL[give.currnode].in_commu_range and give != receive:
                    for infomation in give.featureUpdate[receive.num]:
                        feat = infomation[0]
                        edge = infomation[1]
                        content = infomation[2]
                        if feat == 0:  receive.edgeLengthInfo[edge] = content
                        if feat == 1:  
                            if receive.edgeTotalConnectInfo[edge] < content: receive.edgeTotalConnectInfo[edge] = content
                        if feat == 2:  
                            if receive.edgeCountInfo[edge] < content: receive.edgeCountInfo[edge] = content
                    for i in range(num_agent): 
                        if i != give.num and i != receive.num: receive.featureUpdate[i] += give.featureUpdate[receive.num]
                    give.featureUpdate[receive.num].clear()
                elif give == receive: give.featureUpdate[receive.num].clear()

model = DQN(nfeat=num_feature)
model.load_state_dict(torch.load(lists))  #retrain
model_target = DQN(nfeat=num_feature)
model_target.load_state_dict(model.state_dict())
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.00002)    #
replay = namedtuple('replay',('nextnode','state','action','reward','next_state'))
class Replay_buffer():
    def __init__(self , buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.zeros(  [buffer_size] , dtype = replay)
        self.index = 0
        self.cur_size = 0
    def push(self,experience):
        self.buffer[self.index] = experience
        self.index = (self.index+1)%self.buffer_size
        if self.cur_size < self.buffer_size:
            self.cur_size += 1
    def sample(self,batch_size):
        sample_index = np.random.choice(np.arange(self.cur_size),size=batch_size,replace=False)
        return self.buffer[sample_index]
    def reset(self):
        self.buffer = np.zeros(  [self.buffer_size] , dtype = replay)
        self.index = 0
        self.cur_size = 0

buffer = Replay_buffer(1000)
batch_size = 32
epsilon = 0.1  #
epsilon_decay = 0.0002  #
epsilon_f = 0.1
updateTargetModeltick = 0
updateTargetModelthres = 10
BatchTrainTick = 0
BatchTrainThres = 300
testtime = 50

def pick_edge(ag):
    global BatchTrainTick
    X = feature_matrix(ag)
    #Q (state)
    output = model(torch.from_numpy(X))
    outputnum = -1
    outputmax = -math.inf
    global epsilon, epsilon_f
    Eps = random.uniform(0, 1)
    if Eps > epsilon:
        for i in range(num_node):
            if output[i] >= outputmax and i in node_ALL[ag.togonode].connected_node:
                outputmax = output[i]
                outputnum = i
    else: 
        outputnum = random.choices(node_ALL[ag.togonode].connected_node)[0]

    #Q target (next state)
    X_target = X
    X_target[outputnum][2] += 1 

    #Computing reward
    r0 = len(ag.historyaction) #時間越久reward越少
    r1 = -1   
    if(edge_ALL[find_edge(ag.togonode,outputnum)].ox == 'o'): r1 = 1 #有無走過
    r3 = edge_ALL[find_edge(ag.togonode,outputnum)].count           #被走幾次
    r4 = int(edge_ALL[find_edge(ag.togonode,outputnum)].distance/50)
    # R = -r0 - r1*3 - r3*5 - r4
    R = 0 - r1*4 - r3*6 - r4

    #store to buffer
    experience = replay(outputnum, X, outputnum, R, X_target)
    buffer.push(experience)
    #select batch and training
    if buffer.cur_size >= batch_size:
        batch = buffer.sample(batch_size=batch_size)
        BatchTrainTick += 1
        loss = 0
        for i in range(batch_size):
            nextnode = batch[i].nextnode
            state = batch[i].state
            action = batch[i].action
            reward = batch[i].reward
            next_state = batch[i].next_state
            o = model(torch.from_numpy(state))
            o_max = o[action]
            o_target = model_target(torch.from_numpy(next_state))
            o_target_max = -math.inf
            for i in range(num_node):
                if o_target[i] > o_target_max and i in node_ALL[nextnode].connected_node:
                    o_target_max = o_target[i]
            o_target_max += reward
            loss += loss_fn(o_max,o_target_max)
        loss = loss/batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update target model and epsilon
        global updateTargetModeltick
        updateTargetModeltick += 1
        if(updateTargetModeltick >= updateTargetModelthres):
            updateTargetModeltick = 0
            model_target.load_state_dict(model.state_dict())
        if epsilon-0.001 > epsilon_f : epsilon -= epsilon_decay
        else: epsilon = epsilon_f
    
        if ag.num == 0: print(BatchTrainTick, "Reward: ", R, ", Loss: ", loss, epsilon)

    return outputnum
  
def walking(ag):
    if ag.currnode_ori != ag.togonode : 
        edge_ALL[find_edge(ag.currnode_ori, ag.togonode)].ox = 'o'
        ag.edgeLengthInfo[edge_ALL[ag.togoedge].number] = ag.curedge_length
        ag.alreadyVisitInfo[edge_ALL[ag.togoedge].number] = 1
        for i in range(num_agent): ag.featureUpdate[i].append([0, edge_ALL[ag.togoedge].number, ag.curedge_length])
    ag.currnode = ag.togonode
    ag.currnode_ori = ag.togonode
    ag.lastedge = ag.togoedge
    ag.historyaction.append(ag.togonode)
    ag.step = ag.step - ag.curedge_length  
    ag.togonode = pick_edge(ag)
    togo_edge = find_edge(ag.currnode, ag.togonode)
    ag.curedge_length = edge_ALL[togo_edge].distance
    ag.togoedge = togo_edge
    if ag.lastedge != ag.togoedge and ag.lastedge != 0:
        head = edge_ALL[ag.lastedge].number
        tail = edge_ALL[ag.togoedge].number
        ag.edgeTotalConnectMap[head][tail] = 1
        ag.edgeTotalConnectMap[tail][head] = 1
        ag.edgeTotalConnectInfo[head] = sum(ag.edgeTotalConnectMap[head])
        ag.edgeTotalConnectInfo[tail] = sum(ag.edgeTotalConnectMap[tail])
        for i in range(num_agent): 
            ag.featureUpdate[i].append([1, head, ag.edgeTotalConnectInfo[head]])
            ag.featureUpdate[i].append([1, tail, ag.edgeTotalConnectInfo[tail]])
    edge_ALL[find_edge(ag.currnode_ori, ag.togonode)].count += 1
    ag.edgeCountInfo[edge_ALL[ag.togoedge].number] = edge_ALL[find_edge(ag.currnode_ori, ag.togonode)].count
    for i in range(num_agent): ag.featureUpdate[i].append([2, edge_ALL[ag.togoedge].number, edge_ALL[find_edge(ag.currnode_ori, ag.togonode)].count])

def initialize():
    global BatchTrainTick, buffer
    BatchTrainTick = 0
    for e in edge_ALL: 
        edge_ALL[e].ox = 'x'
        edge_ALL[e].count = 0
    for a in agent_ALL: 
        node_ALL[a.currnode].all_ag_here.remove(a.num)
        node_ALL[a.start].all_ag_here.append(a.num)    
        a.currnode_ori = a.start   
        a.currnode = a.start 
        a.togonode = a.start   
        a.lastedge = 0
        a.togoedge = 0
        a.step = 0
        a.curedge_length = 0
        a.curedge_walked = 0 
        a.reward = 0
        a.historyaction = []
        a.edgeLengthInfo = []
        a.alreadyVisitInfo = []
        a.edgeTotalConnectMap = [[0]*num_edge for i in range(num_edge)]
        a.edgeTotalConnectInfo = []
        a.totalAgentInfo = []
        a.totalAgentMap = [[0]*2 for i in range(num_edge)]
        a.edgeCountInfo = []
        for i in range(num_edge):
            a.edgeLengthInfo.append(0)
            a.alreadyVisitInfo.append(0)
            a.totalAgentInfo.append(0)
            a.edgeTotalConnectInfo.append(0)
            a.edgeCountInfo.append(0)
        a.featureUpdate = []
        for i in range(num_agent): 
            a.featureUpdate.append([])

while epsilon > epsilon_f:
    initialize()
    while not all(edge_ALL[r].ox == 'o' for r in edge_ALL):
        if BatchTrainTick >= BatchTrainThres: break
        for ag in agent_ALL:
            ag.step += ag.speed
            while ag.curedge_length <= ag.step:  
                update_info()
                node_ALL[ag.currnode].all_ag_here.remove(ag.num)
                walking(ag)         
                node_ALL[ag.currnode].all_ag_here.append(ag.num)
            if ag.step > ag.curedge_length/2:
                node_ALL[ag.currnode].all_ag_here.remove(ag.num)       
                ag.currnode = ag.togonode
                update_info()
                node_ALL[ag.currnode].all_ag_here.append(ag.num)

for te in range(testtime): 
    initialize()
    cost = 0
    while not all(edge_ALL[r].ox == 'o' for r in edge_ALL):
        if BatchTrainTick >= BatchTrainThres: break
        for ag in agent_ALL:
            ag.step += ag.speed
            while ag.curedge_length <= ag.step:  
                update_info()
                node_ALL[ag.currnode].all_ag_here.remove(ag.num)
                walking(ag)         
                node_ALL[ag.currnode].all_ag_here.append(ag.num)
            if ag.step > ag.curedge_length/2:
                node_ALL[ag.currnode].all_ag_here.remove(ag.num)       
                ag.currnode = ag.togonode
                update_info()
                node_ALL[ag.currnode].all_ag_here.append(ag.num)
        cost += maxspeed
    print("Testtime: ", te, "Cost: ", cost)
    torch.save(model.state_dict(), lists)

torch.save(model.state_dict(), lists)