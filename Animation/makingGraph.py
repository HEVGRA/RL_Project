import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math

num_node = 20

# file_RL = open("Animation\Algo_" + str(num_node) + ".txt", 'r', encoding='UTF-8') 
file_Algo = open("Animation\Algo_WC_" + str(num_node) + ".txt", 'r', encoding='UTF-8') 

file_RL = open("Animation\RL_dist" + str(num_node) + ".txt", 'r', encoding='UTF-8')
# file_Algo = open("Animation\RL_newedgecon_" + str(num_node) + ".txt", 'r', encoding='UTF-8')

line_Algo = file_Algo.readlines()
line_RL = file_RL.readlines()

lists = "Data\data_" + str(num_node) + ".txt"
file = open(lists, 'r', encoding='UTF-8') 
line = file.readlines()

num_node = int(line[0])
num_edge = int(line[1])
num_agent = int(line[num_node + num_edge + 2])
constraint = int(line[num_node + num_edge + num_agent + 3])
maxspeed = 0 

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
        self.speed = speed
        self.num = number
        self.historyaction = []
        self.stepCount = 0
        self.roadCount = 0
        self.x = []
        self.y = []
        self.distance = []

node_ALL = []
edge_ALL = {}
edge_ALL_RL = {}
agent_ALL = []
agent_ALL_RL = []

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
    ll = Edge(line[k][2], i)
    line[k].pop()
    edge_ALL[tuple(line[k])] = l
    edge_ALL_RL[tuple(line[k])] = ll
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
    ll = Agent(int(line[k][1]), int(line[k][2]), int(line[k][0]))
    agent_ALL.append(l)
    agent_ALL_RL.append(ll)
    if(maxspeed < int(line[k][2])): maxspeed = int(line[k][2])
    node_ALL[l.currnode].all_ag_here.append(i)

def find_edge(a,b):
    if tuple([a,b]) in edge_ALL:   return tuple([a,b])
    else: return tuple([b,a])

# Set up x,y,distance 
maxtime = 0
# Algo
for i in agent_ALL:
    h = line_Algo[i.num+1]
    h = h.strip('[').strip('\n').strip(']').split(', ')
    for j in range(len(h)): h[j] = int(h[j])
    i.historyaction = h
    for j in range(len(i.historyaction)):
        if j < len(i.historyaction)-1 : 
            edge_ALL[find_edge(i.historyaction[j],i.historyaction[j+1])].count += 1
            tmp = edge_ALL[find_edge(i.historyaction[j],i.historyaction[j+1])].count
            if tmp > maxtime: maxtime = tmp
# RL
for i in agent_ALL_RL:
    h = line_RL[i.num+1]
    h = h.strip('[').strip('\n').strip(']').split(', ')
    for j in range(len(h)): h[j] = int(h[j])
    i.historyaction = h
    for j in range(len(i.historyaction)):
        if j < len(i.historyaction)-1 : 
            edge_ALL_RL[find_edge(i.historyaction[j],i.historyaction[j+1])].count += 1
            tmp = edge_ALL_RL[find_edge(i.historyaction[j],i.historyaction[j+1])].count
            if tmp > maxtime: maxtime = tmp

print(maxtime)
# Drawing
r_up, g_up, b_up = 255, 248, 220
r_down, g_down, b_down = 70, 60, 47

fig, (ax, ax2) = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (10, 7))
ax.set_title('Algorithm(Greedy)', fontsize=14)
ax2.set_title('RL(DQN)', fontsize=14)
for a in node_ALL:
    for b in node_ALL:
        if find_edge(a.number,b.number) in edge_ALL: 
            tmp = edge_ALL[find_edge(a.number,b.number)].count
            rr = ((maxtime-tmp)*r_up + tmp*r_down)/maxtime
            gg = ((maxtime-tmp)*g_up + tmp*g_down)/maxtime
            bb = ((maxtime-tmp)*b_up + tmp*b_down)/maxtime
            ax.plot([a.pos[0],b.pos[0]], [a.pos[1],b.pos[1]], color=(rr/255,gg/255,bb/255))
        if find_edge(a.number,b.number) in edge_ALL_RL:
            tmp = edge_ALL_RL[find_edge(a.number,b.number)].count
            rr = ((maxtime-tmp)*r_up + tmp*r_down)/maxtime
            gg = ((maxtime-tmp)*g_up + tmp*g_down)/maxtime
            bb = ((maxtime-tmp)*b_up + tmp*b_down)/maxtime
            ax2.plot([a.pos[0],b.pos[0]], [a.pos[1],b.pos[1]], color=(rr/255,gg/255,bb/255))
           
for node in node_ALL:
    ax.plot([node.pos[0]], [node.pos[1]], 'o-', color=(94/255,38/255,18/255), markersize=8)
    ax2.plot([node.pos[0]], [node.pos[1]], 'o-', color=(94/255,38/255,18/255), markersize=8)

fig.savefig("Animation/graph.png", dpi=300)
fig.show()