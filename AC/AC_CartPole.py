"""
Actor-Critic using TD-error as the advantage
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

import gym

np.random.seed(2)
torch.manual_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.01     # learning rate for actor
LR_C = 0.05     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1) # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        self.n_features = n_features
        self.n_actiosn = n_actions
        self.lr = lr
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(self.n_features, 20)),
            ('relu', nn.ReLU()),
            ('l2', nn.Linear(20, self.n_actiosn)),
            ('acts_prob', nn.Softmax()),
        ]))

    def learn(self, s, a, td):
        s = torch.from_numpy(s[np.newaxis, :]).float()
        td_no_grad = td.detach()
        self.acts_prob = self.model(s)
        log_prob = torch.log(self.acts_prob[0, a])
        loss = -torch.sum(log_prob * td_no_grad)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return -loss

    def choose_action(self, s):
        s = torch.from_numpy(s[np.newaxis, :]).float()
        probs = self.model(s).detach().numpy()
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.lr = lr
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(n_features, 20)),
            ('relu', nn.ReLU()),
            ('V', nn.Linear(20, 1))
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

actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F, lr=LR_C)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break