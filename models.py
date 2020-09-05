import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module): #Shallow QN
    def __init__(self, nfeat):
        super(DQN, self).__init__()
        self.L1 = nn.Linear(nfeat, 1024)
        self.L2 = nn.Linear(1024, 512)
        self.L3 = nn.Linear(512, 256)
        self.L4 = nn.Linear(256, 128)
        self.L5 = nn.Linear(128, 64)
        self.L6 = nn.Linear(64, 32)
        self.L7 = nn.Linear(32, 16)
        self.L8 = nn.Linear(16, 8)
        self.L9 = nn.Linear(8, 4)
        self.L10 = nn.Linear(4, 2)
        self.out = nn.Linear(2, 1)
    
    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))
        x = F.relu(self.L5(x))
        x = F.relu(self.L6(x))
        x = F.relu(self.L7(x))
        x = F.relu(self.L8(x))
        x = F.relu(self.L9(x))
        x = F.relu(self.L10(x))
        x = self.out(x)
        return x