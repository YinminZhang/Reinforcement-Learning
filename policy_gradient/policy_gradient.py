import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from policyNet import PolicyNet
# reproducible
np.random.seed(1)

class PolicyGradient():
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.99,
                 outgraph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.model = PolicyNet(self.n_actions, self.n_features)

    def choose_action(self, observation):
        self.all_act_prob = F.softmax(self.model.forward(torch.from_numpy(observation[np.newaxis, :]).float()))
        prob_weights = self.all_act_prob.detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = torch.from_numpy(self._discount_and_norm_rewards()).float()

        # train on episode
        self.all_act_prob = F.softmax(self.model.forward(torch.from_numpy(np.vstack(self.ep_obs)).float()))
        # x = F.one_hot(torch.from_numpy(np.array(self.ep_as)).to(torch.int64), self.n_actions).float()
        neg_log_prob = torch.sum(-torch.log(self.all_act_prob) *
                                 F.one_hot(torch.from_numpy(np.array(self.ep_as)).to(torch.int64), self.n_actions).float(), dim=1)
        loss = torch.mean(neg_log_prob* discounted_ep_rs_norm).float()
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add*self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs