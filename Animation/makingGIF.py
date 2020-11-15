import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import math

num_node = 10

# file_Algo = open("Animation\Algo_" + str(num_node) + ".txt", 'r', encoding='UTF-8') 
file_Algo = open("Animation\Algo_WC_" + str(num_node) + ".txt", 'r', encoding='UTF-8') 
line_Algo = file_Algo.readlines()

file_RL = open("Animation\RL_" + str(num_node) + ".txt", 'r', encoding='UTF-8')
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
    ll = Agent(int(line[k][1]), int(line[k][2]), int(line[k][0]))
    agent_ALL.append(l)
    agent_ALL_RL.append(ll)
    if(maxspeed < int(line[k][2])): maxspeed = int(line[k][2])
    node_ALL[l.currnode].all_ag_here.append(i)

def find_edge(a,b):
    if tuple([a,b]) in edge_ALL:   return tuple([a,b])
    else: return tuple([b,a])

# Set up x,y,distance 
# Algo
N = 0
for i in agent_ALL:
    h = line_Algo[i.num+1]
    h = h.strip('[').strip('\n').strip(']').split(', ')
    for j in range(len(h)): h[j] = int(h[j])
    i.historyaction = h
    for j in range(len(i.historyaction)):
        k = i.historyaction[j]
        i.x.append(node_ALL[k].pos[0])
        i.y.append(node_ALL[k].pos[1])
        if j < (len(i.historyaction)-1):
            i.distance.append(edge_ALL[find_edge(i.historyaction[j],i.historyaction[j+1])].distance)
for i in agent_ALL: 
    allstep = len(i.historyaction)
    for j in i.distance: allstep += math.ceil(j/i.speed)
    N = max(N,allstep)

# RL
for i in agent_ALL_RL:
    h = line_RL[i.num+1]
    h = h.strip('[').strip('\n').strip(']').split(', ')
    for j in range(len(h)): h[j] = int(h[j])
    i.historyaction = h
    for j in range(len(i.historyaction)):
        k = i.historyaction[j]
        i.x.append(node_ALL[k].pos[0])
        i.y.append(node_ALL[k].pos[1])
        if j != (len(i.historyaction)-1): 
            i.distance.append(edge_ALL[find_edge(i.historyaction[j],i.historyaction[j+1])].distance)
for i in agent_ALL_RL: 
    allstep = len(i.historyaction)
    for j in i.distance: allstep += math.ceil(j/i.speed)
    N = max(N,allstep)

# Drawing
# fig = plt.figure(figsize=(10, 7), dpi=100)
# ax = fig.gca()
fig, (ax, ax2) = plt.subplots(1, 2, sharex = False, sharey = False, figsize = (10, 7))
ax.set_title('Algorithm(Greedy)', fontsize=14)
ax2.set_title('RL(DQN)', fontsize=14)
for a in node_ALL:
    for b in node_ALL:
        if find_edge(a.number,b.number) in edge_ALL: 
            ax.plot([a.pos[0],b.pos[0]], [a.pos[1],b.pos[1]], color=(210/255,180/255,140/255))
            ax2.plot([a.pos[0],b.pos[0]], [a.pos[1],b.pos[1]], color=(210/255,180/255,140/255))
for node in node_ALL:
    ax.plot([node.pos[0]], [node.pos[1]], 'o-', color=(94/255,38/255,18/255), markersize=8)
    ax2.plot([node.pos[0]], [node.pos[1]], 'o-', color=(94/255,38/255,18/255), markersize=8)

# Algo
dot = []
trace = []
for i in agent_ALL:
    j, = ax.plot([], [], color=(25/255,25/255,122/255), marker='o', markersize=8, markeredgecolor='black', linestyle='')
    dot.append(j)
for i in agent_ALL:
    j = []
    for ii in range(len(i.historyaction)):
        k, = ax.plot([], [], color=(61/255,89/255,171/255), markersize=2, linestyle='-')
        j.append(k)
    trace.append(j)
# RL
dot_RL = []
trace_RL = []
for i in agent_ALL_RL:
    j, = ax2.plot([], [], color=(25/255,25/255,122/255), marker='o', markersize=8, markeredgecolor='black', linestyle='')
    dot_RL.append(j)
for i in agent_ALL_RL:
    j = []
    for ii in range(len(i.historyaction)):
        k, = ax2.plot([], [], color=(61/255,89/255,171/255), markersize=2, linestyle='-')
        j.append(k)
    trace_RL.append(j)

def PositionCalculate(ag):
    num = ag.roadCount
    step = ag.stepCount
    speed = ag.speed
    x, y, z = ag.x, ag.y, ag.distance
    xx, yy = 0, 0
    if num >= len(ag.historyaction)-1: 
        xx = x[-1]
        yy = y[-1]
    else:
        xx = (step*x[num+1] + ((z[num]/speed)-step)*x[num])/(z[num]/speed)
        yy = (step*y[num+1] + ((z[num]/speed)-step)*y[num])/(z[num]/speed)
        ag.stepCount += 1
        if abs(xx-x[num]) >= abs(x[num+1]-x[num]): 
            xx = x[num+1]
            yy = y[num+1]
            ag.stepCount = 0
            ag.roadCount += 1
    return xx, yy, num

def update(i):
    alltrace = []
    alltrace_RL = []
    for i in range(num_agent):
        xx, yy, num = PositionCalculate(agent_ALL[i])
        dot[i].set_data(xx, yy)
        ag = agent_ALL[i]
        trace[i][num].set_data([ag.x[num], xx], [ag.y[num], yy])
        alltrace += trace[i]

        xx2, yy2, num2 = PositionCalculate(agent_ALL_RL[i])
        dot_RL[i].set_data(xx2, yy2)
        ag2 = agent_ALL_RL[i]
        trace_RL[i][num2].set_data([ag2.x[num2], xx2], [ag2.y[num2], yy2])
        alltrace_RL += trace_RL[i]
    return alltrace + dot + alltrace_RL + dot_RL

def init():
    for i in range(num_agent):
        dot[i].set_data(agent_ALL[i].x[0], agent_ALL[i].y[0])
        dot_RL[i].set_data(agent_ALL_RL[i].x[0], agent_ALL_RL[i].y[0])
    return dot + dot_RL

ani = animation.FuncAnimation(fig=fig, func=update, frames=N, init_func=init, interval=100/N, blit=True, repeat=False, save_count=N)
plt.show()
# ani.save(r'MovingPoint.gif', writer='imagemagick')