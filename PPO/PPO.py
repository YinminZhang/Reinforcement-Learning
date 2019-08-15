"""
Proximal Policy Optimization(PPO) using pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.kl import kl_divergence

import numpy as np
import matplotlib.pyplot as plt
import gym
from model import *

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]                                                # choose the method for optimization

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class PPO(object):
    def __init__(self, n_features, n_actions):
        self.critic = Critic(n_features=n_features)
        self.actor_new = Actor(n_features=n_features, n_actions=n_actions)
        self.actor_old = Actor(n_features=n_features, n_actions=n_actions)
        self.actor_old.load_state_dict(self.actor_new.state_dict())

    def choose_action(self, s):
        return self.actor_old.choose_action(s)

    def get_v(self, s):
        return self.critic.get_v(s)

    def update(self, s, a, r, method, A_update_steps, C_update_steps, A_lr, C_lr):
        self.actor_old.load_state_dict(self.actor_new.state_dict())
        advantage = self.critic.advantage(s, r)
        # update actor
        if method['name'] == 'kl_pen':
            for _ in range(A_update_steps):
                advantage = self.critic.advantage(s, r)
                ratio = self.actor_new(s).prob(a)/self.actor_old(s).prob(a)
                surr = ratio * advantage

                kl = kl_divergence(self.actor_new, self.actor_old)
                kl_mean = kl.mean()
                a_loss = -(surr - method['lam'] * kl).mean()

                # optimize the new policy model
                A_optim = optim.Adam(self.actor_new.parameters(), A_lr)
                A_optim.zero_grad()
                a_loss.back()
                A_optim.step()

                if kl_mean > 4 * method['kl_target']:   # this in in google's paper
                    break
                if kl_mean < method['kl_target']/1.5:       # adaptive lambda, this is in OpenAI's paper
                    method['lam'] /= 2
                elif kl_mean > method['kl_target']*1.5:
                    method['lam'] *= 2
                method['lam'] = np.clip(method['lam'], 1e-4, 10)

        else:   # clipping method, find this is better (OpenAI's paper)
            for _ in range(A_update_steps):
                # tensorflow ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                advantage = self.critic.advantage(s, r)
                ratio = torch.exp(self.actor_new(s).log_prob(a) - self.actor_old(s).log_prob(a))
                surr = ratio * advantage
                a_loss = torch.min(surr, torch.clamp(torch.tensor(1.-method['epsilon']), torch.tensor(1.0+method['epsilon']))*advantage).mean()

                # optimize the new policy model
                A_optim = optim.Adam(self.actor_new.parameters(), A_lr)
                A_optim.zero_grad()
                a_loss.backward()
                A_optim.step()

        # update critic
        for _ in range(C_update_steps):
            c_loss = self.critic.loss_func(s, r)
            c_optim = optim.Adam(self.critic.parameters(), C_lr)

            # optimize the critic model
            c_optim.zero_grad()
            c_loss.backward()
            c_optim.step()

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO(S_DIM, A_DIM)
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        # env.render()
        a = ppo.choose_action(v_wrap(s[None, :]))
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(v_wrap(s[None, :]))
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = v_wrap(np.vstack(buffer_s)), v_wrap(np.vstack(buffer_a)), v_wrap(np.array(discounted_r)[:, np.newaxis])
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br, METHOD, A_UPDATE_STEPS, C_UPDATE_STEPS, A_LR, C_LR)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

