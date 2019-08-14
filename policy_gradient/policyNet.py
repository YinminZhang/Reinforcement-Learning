import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class PolicyNet(nn.Module):
    def __init__(self, n_actions, n_features):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Sequential(
            OrderedDict([
                ('l1', nn.Linear(n_features, 10)),
                ('tanh', nn.Tanh())
            ])
        )
        self.l2 = nn.Sequential(
            OrderedDict([
                ('l2', nn.Linear(10, n_actions)),
            ])
        )

    def forward(self, x):
        x = self.l1(x)
        all_act = self.l2(x)
        # all_act_prob = F.softmax(all_act)
        return all_act
