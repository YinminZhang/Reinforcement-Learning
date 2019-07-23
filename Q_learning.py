import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_Sataes = 6
Actions = ['left', 'right']
Epsilon = 0.9
Alpha = 0.1
Lambda = 0.9
Max_Episodes = 13
Fresh_time = 0.3

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    print(table)
    return table

def choose_actions(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform()>Epsilon) or (state_actions.all() == 0):
        action_name = np.random.choice(Actions)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_Sataes - 2:
            S = "terminal"
            R = 1
        else:
            S = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S = S
        else:
            S = S -1
    return S, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_Sataes-1)+['T']
    if S == "terminal":
        interaction = "Episode %s: total_steps = %s" %(episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                       ', end='')
    else:
        env_list[S]='o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(Fresh_time)

if __name__ == "__main__":
    q_table = build_q_table(N_Sataes, Actions)

    for episode in range(Max_Episodes):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_actions(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + Lambda * q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True
            q_table.ix[S, A] += Alpha * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
    print(q_table)