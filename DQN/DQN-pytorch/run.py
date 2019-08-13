from maze_env import Maze
from RL import Agent

def run_maze():
    step = 0
    
    for epidode in range(300):
        # Intial observation
        observation = env.reset()

        while True:
            # Fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            # Experience replay
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # Swap observation
            observation = observation_

            # Break while lopp when end of this episode
            if done:
                # print('\n {:d}'.format(epidode))
                break
            step += 1
    # End of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = Agent(env.n_actions, env.n_features,
                    learning_rate=0.01,
                    reward_decay=0.9,
                    e_greedy=0.9,
                    replace_target_iter=200,
                    memory_size=2000,
                    )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()