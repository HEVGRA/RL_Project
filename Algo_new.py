import numpy as np
import random
import math
import time

# tStart = time.time()

file = open('Data\data_10.txt', 'r', encoding='UTF-8') 
line = file.readlines()

num_node = int(line[0])
num_edge = int(line[1])
num_agent = int(line[num_node + num_edge + 2])
constraint = int(line[num_node + num_edge + num_agent + 3])
maxspeed = 0 
Cost = 0              

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
        self.currnode = cur         #當下的node
        self.togonode = cur         #當下所在edge的尾端node
        self.curedge_length = 0     #當下所在edge的長度
        self.step = 0
        self.speed = speed
        self.num = number
        self.historyaction = []
        self.start = cur
        self.cost = 0

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


def find_edge(a,b):
    if tuple([a,b]) in edge_ALL:   return tuple([a,b])
    else: return tuple([b,a])

def choosing_edge(ag):
    # todonode = -5
    cnt = math.inf
    for i in node_ALL[ag.currnode].connected_node:
        if edge_ALL[find_edge(ag.currnode,i)].count <= cnt:
            cnt = edge_ALL[find_edge(ag.currnode,i)].count
            togonode = i
    return togonode

def walking(ag):
    if ag.currnode != ag.togonode : 
        edge_ALL[find_edge(ag.currnode, ag.togonode)].ox = 'o'
    ag.currnode = ag.togonode
    ag.historyaction.append(ag.togonode)
    ag.step = ag.step - ag.curedge_length

    ag.togonode = choosing_edge(ag)
    togo_edge = find_edge(ag.currnode, ag.togonode)
    edge_ALL[togo_edge].count += 1
    ag.curedge_length = edge_ALL[togo_edge].distance

def initialize():
    for e in edge_ALL: 
        edge_ALL[e].ox = 'x'
        edge_ALL[e].count = 0
    for a in agent_ALL: 
        a.currnode = a.start 
        a.togonode = a.start         
        a.curedge_length = 0 
        a.step = 0
        a.cost = 0
        a.historyaction = []


initialize()                                 
while not all(edge_ALL[r].ox == 'o' for r in edge_ALL) :
    for ag in agent_ALL:
        ag.step += ag.speed
        ag.cost += ag.speed
        while ag.curedge_length <= ag.step:  walking(ag)
    Cost += maxspeed
# Write all action to file
fileforHistoryaction = "Animation/Algo_"+ str(num_node) +".txt"
f = open(fileforHistoryaction, "w")
print(num_node, file = f)
for i in agent_ALL: print(i.historyaction, file = f)


allEdgeCost = 0
for i in edge_ALL:   allEdgeCost += edge_ALL[i].distance
allAgentCost = 0
for i in agent_ALL:   allAgentCost += i.cost
all_historyaction = -num_agent
for i in agent_ALL:  all_historyaction += len(i.historyaction)

# for i in agent_ALL:   print(i.historyaction)  
print("Map cost = ",allEdgeCost)
print("All agents' cost = ",allAgentCost)
print("Repeated rate = ","%.2f"%((all_historyaction-num_edge)/all_historyaction*100),"%")                      
print("Largest cost = ",Cost)

# tEnd = time.time()
# print("It cost %f sec" % (tEnd - tStart))


