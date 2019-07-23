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
        self.eligibility_trace = self.q_table.copy()

    def choose_action(self, obersvation):
        self._check_state_exist(obersvation)
        if np.random.uniform()<self.epsilon:
            actions = self.q_table.loc[obersvation,:]
            action  = np.random.choice(actions[actions==np.max(actions)].index)
        else:
            action = np.random.choice(self.q_table.loc[obersvation,:].index)
        return action

    def _check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append =pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_, method=2):
        self._check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        if method == 1:
            self.eligibility_trace.loc[s, a] += 1
        elif method == 2:
            self.eligibility_trace.loc[s, :] = 0
            self.eligibility_trace.loc[s, a] = 1
        self.q_table += self.alpha * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_