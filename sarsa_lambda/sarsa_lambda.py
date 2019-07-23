import numpy as np
import pandas as pd

class SarsaLambdaTable():
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_lambda=0.9):
        self.actions = action_space
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.lambda_ = trace_lambda
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=action_space, dtype=np.float64)
    
    def choose_action(self, obersvation):
        self._check_state_exist(obersvation):
        if np.random.uniform()<self.epsilon:
            actions = self.q_table.loc[obersvation,:]
            action  = np.random.choice(actions[actions==np.max(actions)].index)
        else:
            action = np.random.choice(self.q_table.loc[obersvation,:].index)
        return action

    def _check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(actions)
                )
            )