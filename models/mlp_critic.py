import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *

class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim, gpu=True, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.gpu = gpu
        self.num_layers = num_layers 

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)
            
    def forward(self, x, h=None, test=False):  # seq * batch * 22
        value_ori, hidden = self.gru(x, h)
        value = self.dense2(F.relu(self.dense1(value_ori)))    # seq * batch * 1
        
        if not test:
            return value, hidden
        else:
            return value, hidden, value_ori

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))