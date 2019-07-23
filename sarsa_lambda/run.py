from maze import Maze
from sarsa_lambda import SarsaLambdaTable

Max_Episodes = 100

def update():
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
            action_ = RL.choose_action(str(observation_))
            # Save the transition from S to S'
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # Swap observation
            observation = observation_
            action = action_
            if done:
                break
        
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(action_space = list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
