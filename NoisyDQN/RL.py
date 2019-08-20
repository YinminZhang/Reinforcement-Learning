import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class DQN(nn.Module):
    def __init__(self, n_features, n_actions):
        super(DQN, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.mu_weight_1 = nn.Parameter(torch.empty(10, n_features))
        self.mu_bias_1 = nn.Parameter(torch.empty(10))
        self.sigma_weight_1 = nn.Parameter(torch.empty(10, n_features))
        self.sigma_bias_1 = nn.Parameter(torch.empty(10))
        self.weight_epsilon1 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(10, n_features)))
        self.bias_epsilon1 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=10))

        self.mu_weight_2 = nn.Parameter(torch.empty(n_actions, 10))
        self.mu_bias_2 = nn.Parameter(torch.empty(n_actions))
        self.sigma_weight_2 = nn.Parameter(torch.empty(n_actions, 10))
        self.sigma_bias_2 = nn.Parameter(torch.empty(n_actions))
        self.weight_epsilon2 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(n_actions, 10)))
        self.bias_epsilon2 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=n_actions))

    def forward(self, x):
        x = F.linear(x, self.mu_weight_1 + self.sigma_weight_1 * self.weight_epsilon1, self.mu_bias_1 + self.sigma_bias_1 * self.bias_epsilon1)
        x = F.relu(x)
        x = F.linear(x, self.mu_weight_2 + self.sigma_weight_2 * self.weight_epsilon2, self.mu_bias_2 + self.sigma_bias_2 * self.bias_epsilon2)
        return x

    def add_noisy(self, method="FGN"):
        # use Factorised Gaussian noise
        if method == "FGN":
            in_features = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=self.n_features))
            out_features = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=10))
            self.weight_epsilon1 = out_features.ger(in_features)
            self.bias_epsilon1 = out_features

            in_features = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=10))
            out_features = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=self.n_actions))
            self.weight_epsilon2 = out_features.ger(in_features)
            self.bias_epsilon2 = out_features
        else:
            self.weight_epsilon1 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(10, self.n_features)))
            self.bias_epsilon1 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=10))

            self.weight_epsilon2 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=(self.n_actions, 10)))
            self.bias_epsilon2 = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=self.n_actions))



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

        # if np.random.uniform()<self.epsilon:
        #     action_value = self.policy_net.forward(observation)
        #     action = torch.argmax(action_value).numpy()
        # else:
        #     action = np.random.randint(0, self.n_actions)
        action_value = self.policy_net.forward(observation)
        action = torch.argmax(action_value).numpy()
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
        q_eval = self.policy_net(batch_memory[:, :self.n_features])
        self.target_net.eval()
        q_next = self.target_net(batch_memory[:, -self.n_features:])

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
