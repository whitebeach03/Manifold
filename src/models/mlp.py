import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Manifold_MLP(nn.Module):
    def __init__(self):
        super(Manifold_MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward_to_fc2(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x  

    def forward_to_fc4(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x  
    
    def forward(self, x): 
        x = self.forward_to_fc2(x)  
        x = self.forward_to_fc4(x)     
        x = self.fc7(x)          
        return x


# Number of parameters
# mlp = MLP()
# mmlp = Manifold_MLP()

# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     return total_params

# params_mlp  = count_parameters(mlp)
# params_mmlp = count_parameters(mmlp)
# print(f"Number of parameters: {params_mlp}")
# print(f"Number of parameters: {params_mmlp}")