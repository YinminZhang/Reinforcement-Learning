from maze import Maze
from RL import *

Max_Episodes = 100
Algorithm = 'Q_learning' # Q_learning, Sarsa, SarsaLambda

def update_Sarsa():
    for episode in range(Max_Episodes):
        # Initial oberservation
        observation = env.reset()
        action = RL.choose_action(str(observation))
        while True:
            # Fresh env
            env.render()
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL choose the next actions
            action_ = RL.choose_action(str(observation))
            # Save the transition from S to S'
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # Swap observation
            observation = observation_
            action = action_
            if done:
                break
        
    print('game over')
    env.destroy()

def update():
    for episode in range(Max_Episodes):
        # Initial oberservation
        observation = env.reset()
        while True:
            # Fresh env
            env.render()
            # RL choose actions
            action = RL.choose_action(str(observation))
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # Save the transition from S to S'
            RL.learn(str(observation), action, reward, str(observation_))
            # Swap observation
            observation = observation_
            if done:
                break
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    if Algorithm == 'SarsaLambda':
        RL = SarsaLambda(action_space = list(range(env.n_actions)))
        env.after(100, update_Sarsa)
    elif Algorithm == 'Q_learning':
        RL = Q_learning(action_space = list(range(env.n_actions)))
        env.after(100, update)
    elif Algorithm == 'Sarsa':
        RL = Sarsa(action_space = list(range(env.n_actions)))
        env.after(100, update_Sarsa)
    env.mainloop()
