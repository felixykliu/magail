import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *

# Don't use it now. Unfinished!
class DiscretePolicy(nn.Module):
    '''
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)
    '''
    def __init__(self, activation='relu'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        
        self.pool3 = nn.MaxPool2d(3, stride=1)
        self.pool5 = nn.MaxPool2d(5, stride=1)
        self.pool9 = nn.MaxPool2d(9, stride=1)
        
        self.conv1 = nn.Conv2d(16, 12, 25, stride=5) # 76 * 68
        self.conv2 = nn.Conv2d(12, 8, 21, stride=5) # 12 * 10
        
        self.fc = nn.Linear(12*10*8, 289)
        self.fc.weight.data.mul_(0.1)
        self.fc.bias.data.mul_(0.0)
        
    '''
    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = F.softmax(self.action_head(x))
        return action_prob
    '''
    def forward(self, x):
    
        x = torch.cat((x, self.pool3(x), self.pool5(x), self.pool9(x)), 1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x.view(-1, 12 * 10 * 8)

        action_prob = F.softmax(self.fc(x))
        return action_prob
    

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial()
        return action.data

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = Variable(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return action_prob.gather(1, actions.unsqueeze(1))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).data
        return M, action_prob, {}

