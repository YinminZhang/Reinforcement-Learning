import numpy as np
import pandas as pd

class SarsaTable():
    def __init__(self, actions, learning_rate = 0.01, reward_deacy = 0.9, e_greedy= 0.9):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_deacy
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def choose_action(self, obersvation):
        self._check_state_exist(obersvation)
        if np.random.uniform()<self.epsilon:
            state_action = self.q_table.loc[obersvation,:]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action
    
    def _check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state
                )
            )

    def learn(self, s, a, r, s_, a_):
        self._check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha*(q_target - q_predict)