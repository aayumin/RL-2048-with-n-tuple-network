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
INITIAL_LR = 1e-4


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(3.11)
        
class Model2048(nn.Module):

    def __init__(self, _tuple_len = 4, _num_tuples = 17, max_values = 16, ACTION_SPACE=4, pretrained=False):
        super(Model2048, self).__init__()
        
        
        tuple_len = _tuple_len
        num_tuples = _num_tuples
        
        self.Input_dim = int(num_tuples * (max_values * tuple_len))
        self.H_dim1 = 64
        #self.H_dim2 = 32
        self.H_dim2 = 128
        #self.H_dim3 = 32
        
        self.FC =  nn.Sequential(
            nn.Linear(self.Input_dim, self.H_dim1),
            nn.ReLU(),
            nn.Linear(self.H_dim1, self.H_dim2),
            nn.ReLU(),
            #nn.Linear(self.H_dim2, self.H_dim3),
            #nn.ReLU(),
            #nn.Linear(self.H_dim3, self.H_dim2),
            #nn.ReLU(),
            nn.Linear(self.H_dim2, ACTION_SPACE),
            #nn.Linear(self.H_dim2, 1),
            nn.Sigmoid(),
            #nn.ReLU(),
        )
        self.FC.apply(init_weights)
        
    def forward(self, x):
        y = self.FC(x.reshape((-1, self.Input_dim)).float())
        return y

