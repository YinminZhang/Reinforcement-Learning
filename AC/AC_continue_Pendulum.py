"""
Actor-Critic using TD-error as the advantage
"""

import numpy as np
import torch
import torch.nn as nn
# import torch.distributions.normal
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

import gym

np.random.seed(2)
torch.manual_seed(2)  # reproducible

class Actor(object):
    def __init__(self, n_features, action_bound, lr=0.001):
        self.n_features = n_features
        # self.n_actiosn = n_actions
        self.action_bound = action_bound
        self.lr = lr
        self.l1 = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(self.n_features, 30)),
            ('relu', nn.ReLU()),
        ]))
        self.mu = nn.Sequential(OrderedDict([
            ('l2', nn.Linear(30, 1)),
            ('tanh', nn.Tanh()),
        ]))
        self.sigma = nn.Sequential(OrderedDict([
            ('l', nn.Linear(30, 1)),
            ('act_prob', nn.Softplus()),
        ]))

    def learn(self, s, a, td):
        s = torch.from_numpy(s[np.newaxis, :]).float()
        td_no_grad = td.detach()
        mu, sigma = torch.squeeze(self.mu(self.l1(s))), torch.squeeze(self.sigma(self.l1(s)))
        normal_dist =  Normal(mu*2, sigma+0.1)
        # action = torch.clamp(normal_dist.sample(1), self.action_bound[0], self.action_bound[1])
        log_prob = normal_dist.log_prob(torch.from_numpy(a))
        self.exp_v = log_prob * td_no_grad
        self.exp_v += 0.01*normal_dist.entropy()
        self.exp_v = -self.exp_v
        optimizer = optim.Adam([{'params':self.l1.parameters()},{'params':self.sigma.parameters()},{'params':self.mu.parameters()}], lr=self.lr)

        # optimize the model
        optimizer.zero_grad()
        self.exp_v.backward()
        optimizer.step()
        return -self.exp_v

    def choose_action(self, s):
        s = torch.from_numpy(s[np.newaxis, :]).float()
        mu, sigma = torch.squeeze(self.mu(self.l1(s))), torch.squeeze(self.sigma(self.l1(s)))
        normal_dist = Normal(mu * 2, sigma + 0.1)
        action = torch.clamp(normal_dist.sample(), self.action_bound[0][0], self.action_bound[1][0])
        return action

class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.lr = lr
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(n_features, 30)),
            ('relu', nn.ReLU()),
            ('V', nn.Linear(30, 1))
        ]))

    def learn(self, s, r, s_):
        s, s_ = torch.from_numpy(s[np.newaxis, :]).float(), torch.from_numpy(s_[np.newaxis, :]).float()
        v, v_ = self.model(s), self.model(s_)
        td_error = r + GAMMA * v_ -v
        loss = torch.pow(td_error, 2)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return td_error

OUTPUT_GRAPH = False
MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
GAMMA = 0.9
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high

actor = Actor(n_features=N_S, lr=LR_A, action_bound=[-A_BOUND, A_BOUND])
critic = Critic(n_features=N_S, lr=LR_C)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []
    while True:
        # if RENDER:
        env.render()
        a = np.array([actor.choose_action(s).numpy()])

        s_, r, done, info = env.step(a)
        r /= 10

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1
        ep_rs.append(r)
        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
