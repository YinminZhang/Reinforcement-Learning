from maze import Maze
from sarsa import SarsaTable

Max_Episodes = 100

def update():
    for episode in range(Max_Episodes):
        # Initial oberservation
        obersvation = env.reset()
        action = RL.choose_action(str(obersvation))
        while True:
            # Fresh env
            env.render()
            # RL take action and get next obersvation and reward
            obersvation_, reward, done = env.step(action)
            # RL choose the next actions
            action_ = RL.choose_action(str(obersvation_))
            # Save the transition from S to S'
            RL.learn(str(obersvation), action, reward, str(obersvation_), action_)
            # Swap obersvation
            obersvation = obersvation_
            action = action_
            if done:
                break
        
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions = list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
