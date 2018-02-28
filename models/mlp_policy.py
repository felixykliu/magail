import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_size, gpu=True, log_std=0.0, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_size = action_size
        self.gpu = gpu
        self.num_layers = num_layers 
        self.is_disc_action = False

        self.gru = nn.GRU(state_dim, hidden_dim, num_layers)
        self.dense = nn.Linear(hidden_dim, action_size)
        
        self.action_log_std = nn.Parameter(torch.ones(1, action_size) * log_std)
            
    def forward(self, x, h=None):
        action_mean, hidden = self.gru(x, h)   ## action: seq * batch * hidden_size, hidden: layers * batch * hidden_size
        action_mean = (F.sigmoid(self.dense(action_mean)) - 0.5) / 5.0
        # action_mean = self.dense(action_mean)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std, hidden

    def select_action(self, x, h, test=False):
        action_mean, _, action_std, hidden = self.forward(x, h)
        action = torch.normal(action_mean, action_std)
        if not test:
            return action, hidden
        else:
            return action, hidden, action_mean, action_std
    
    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))

    def get_kl(self, x):
        mean1, log_std1, std1, _ = self.forward(x)

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std, _ = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}
    
    def get_params(self):
        return self.parameters()

