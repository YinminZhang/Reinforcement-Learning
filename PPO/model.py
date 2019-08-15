"""
The Network of Actor and Critic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.distributions.normal import Normal

import numpy as np

class Critic(nn.Module):
    def __init__(self, n_features):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(n_features, 100)),
            ('relu', nn.ReLU6(inplace=True)),
            ('l2', nn.Linear(100, 1))
        ]))

    def forward(self, x):
        self.v = self.layers(x)
        return self.v

    def get_v(self, s):
        self.eval()
        if len(s.shape) < 2: s = s.unsqueeze(0)
        v = self.forward(s)
        return v

    def loss_func(self, s, r):
        self.train()
        advantage = r - self.forward(s)
        c_loss = advantage.pow(2).mean()
        return c_loss

    def advantage(self, s, r):
        self.eval()
        advantage = r - self.forward(s)
        # advantage = (advantage -advantage.mean())/(advantage.std()+1e-6)
        return advantage

class Actor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(n_features, 100)
        self.sigma = nn.Sequential(OrderedDict([
            ('l_sigma', nn.Linear(100, n_actions)),
            ('softplus', nn.Softplus())
        ]))
        self.mu = nn.Sequential(OrderedDict([
            ('l_mu', nn.Linear(100, n_actions)),
            ('tanh', nn.Tanh())
        ]))

    def forward(self, x):
        mu = 2 * self.mu(self.l1(x))
        sigma = self.sigma(self.l1(x))
        norm_dist = Normal(loc=mu, scale=sigma)
        return norm_dist

    def choose_action(self, s):
        self.eval()
        m = self.forward(s)
        a = m.sample().numpy()[0]
        return np.clip(a, -2, 2)
