import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

class DQN(nn.Module):
    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(n_features, 10, bias=True)
        self.linear2 = nn.Linear(10, n_actions, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Agent():
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None,
        out_graph=False,):
        
        # initial
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learning_step_counter = 0
        
        # intialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # consist of [target_net, evaluate_net]
        self.policy_net = DQN(n_features, n_actions,)
        self.target_net = DQN(n_features, n_actions,)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.cost_hist = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new momery
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.from_numpy(observation[np.newaxis, :]).float()

        if np.random.uniform()<self.epsilon:
            action_value = self.policy_net.forward(observation)
            action = torch.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_hist)), self.cost_hist)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def learn(self):
        if self.learning_step_counter%self.replace_target_iter==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('\ntarget params replaced')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = torch.from_numpy(self.memory[sample_index, :]).float()
        q_eval = self.policy_net.forward(batch_memory[:, :self.n_features])
        q_next = self.target_net.forward(batch_memory[:, -self.n_features:])

        q_target = q_eval.clone()
        batch_index = torch.arange(self.batch_size, dtype=torch.long)
        eval_act_index = batch_memory[:, self.n_features].long()
        reward = batch_memory[:, self.n_features+1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, dim=1)[0]

        cost = F.smooth_l1_loss(q_eval, q_target)
        optimizer = optim.RMSprop(self.policy_net.parameters())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        self.cost_hist.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon
        self.learning_step_counter += 1
