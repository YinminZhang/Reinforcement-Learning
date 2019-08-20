from maze import Maze
from Q_Learning import QLearningTable

Max_Episodes = 100

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
    RL = QLearningTable(actions = list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
