import numpy as np
import pandas as pd

class RL():
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self._check_state_exist(observation)
        if np.random.uniform()<self.epsilon:
            actions = self.q_table.loc[observation, :]
            action = np.random.choice(actions[actions == np.max(actions)].index)
        else:
            action = np.random.choice(self.q_table.loc[observation, :].index)
        return action
  
    def _check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )
    
    def learn(self, *args):
        pass

class Q_learning(RL):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(Q_learning, self).__init__(action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9)

    def learn(self, s, a, r, s_):
        self._check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

class Sarsa(RL):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(Sarsa, self).__init__(action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9)

    def learn(self, s, a, r, s_, a_):
        self._check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)

class SarsaLambda(RL):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_lambda=0.9):
        super(SarsaLambda, self).__init__(action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9)
        self.lambda_ = trace_lambda
        self.eligibility_trace = self.q_table.copy()

    def _check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_, method=2):
        self._check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma*self.q_table.loc[s_, a_]
        else:
            q_target = r
        error = q_target - q_predict

        if method == 1:
            self.eligibility_trace.loc[s, a] += 1
        elif method == 2:
            self.eligibility_trace.loc[s, :] *= 0
            self.eligibility_trace.loc[s, a] = 1

        self.q_table += self.alpha*error*self.eligibility_trace
        self.eligibility_trace *= self.lambda_*self.gamma
