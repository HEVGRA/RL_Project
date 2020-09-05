import random
import numpy as np

node = 10
edge = int(node*(node-1)/2 - min(3,int(node*(node-1)/12)))
# edge = int(node*(node-1)/2/4)
agent = 3
constraint = 500

lists = "Data/data_"+ str(node) +".txt"
f = open(lists, "w")

nodeXY_min = 200
nodeXY_max = 1000
edgeLength_mix = 50
edgeLength_max = 1000
speed_min = 1
speed_max = 10

node_list = {}
edge_list = {}
print(node, file = f)
print(edge, file = f)
i=0
while i < node:
    x = random.randint(nodeXY_min, nodeXY_max)
    y = random.randint(nodeXY_min, nodeXY_max)
    pos = tuple([x,y])
    if pos not in node_list: 
        node_list[pos] = 1
        print(i,x,y, file = f)
        i+=1
i=0
while i < edge:
    x = random.randint(0, node-1)
    y = random.randint(0, node-1)
    l = random.randint(edgeLength_mix, edgeLength_max)
    pos1 = tuple([x,y])
    pos2 = tuple([y,x])
    if pos1 not in edge_list and pos2 not in edge_list and x != y: 
        edge_list[pos1] = 1
        print(x,y,l, file = f)
        i+=1
print(agent, file = f)
for j in range(agent):
    pos = random.randint(0, node-1)
    # speed = random.randint(speed_min, speed_max)
    speed = 9
    print(j,pos,speed, file = f)
print(constraint, file = f)

f.close()