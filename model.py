#import tensorflow as tf
#from keras.models import Model
#from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
import os

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

ACTION_SPACE = 4
INPUT_SHAPE = (16, 4, 4)
DEPTH_1 = 128
DEPTH_2 = 256
HIDDEN_UNITS = 512


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(3.11)
        
class Model2048(nn.Module):

    def __init__(self, total_tuple_len = 4 * 17, num_tuples = 17, max_values = 16, ACTION_SPACE=4, pretrained=False):
        super(Model2048, self).__init__()
        
        
        print(f"total_tuple_len: {total_tuple_len},  num_tuples: {num_tuples}")
        
        
        #self.Input_dim = int(num_tuples * (max_values * tuple_len))
        self.Input_dim = int(max_values * total_tuple_len)
        self.H_dim1 = 256
        # self.H_dim2 = 512
        # self.H_dim3 = 128
        # self.H_dim4 = 64
        
        # self.H_dim1 = 256
        # self.H_dim2 = 128
        # self.H_dim3 = 64
        # self.H_dim4 = 32    
        
        #print(f"input:  {self.Input_dim}")
        
        self.FC =  nn.Sequential(
            nn.Linear(self.Input_dim, self.H_dim1),
            nn.ReLU(),
            # nn.Linear(self.H_dim1, self.H_dim2),
            # nn.ReLU(),
            # nn.Linear(self.H_dim2, self.H_dim3),
            # nn.ReLU(),
            # nn.Linear(self.H_dim3, self.H_dim4),
            # nn.ReLU(),
        )
        
        self.baseline = nn.Sequential(
            nn.Linear(self.H_dim1, 1),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self.H_dim1, ACTION_SPACE),
        )
        
        self.FC.apply(init_weights)
        self.baseline.apply(init_weights)
        self.advantage.apply(init_weights)
        
    def forward(self, x):
        
        VALUE_MAX = 10.0
        x = self.FC(x.reshape((-1, self.Input_dim)).float())
        
        y_baseline = self.baseline(x)
        y_advantage = self.advantage(x)
        #y = self.FC(x.reshape((-1, self.Input_dim)).float()) * VALUE_MAX
        #y = 1 / (1 + torch.exp(0.05 * -1 * y)) * VALUE_MAX
        
        y_baseline = 1 / (1 + torch.exp(0.05 * -1 * y_baseline)) * VALUE_MAX
        y_advantage = 1 / (1 + torch.exp(0.05 * -1 * y_advantage)) * VALUE_MAX
        
        y = y_baseline + (y_advantage - torch.mean(y_advantage))
        return y

