import numpy as np
from tqdm import tqdm

def calculate_performance(episodes, env, agent):

    episodic_returns = []
    
    for epi in tqdm(range(episodes)):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            s = ns
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

def calculate_sample_efficiency(episodes, env, agent):

    episodic_returns = []
    
    for epi in tqdm(range(episodes)):
        
        s = env.reset()

        done = False
        cum_reward = 0.0
        timestep = 0

        while not done:  
            timestep += 1
            action = agent.action(s)
            s, reward, done, _ = env.step(action)
            cum_reward += reward
            #s = ns
            # saving reward and is_terminals
            print(reward)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            # e.g., agent.update()
            if (timestep) % 5 ==0:
                agent.train(epi)

            #####################
        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()


        #####################
        
        episodic_returns.append(cum_reward)
    agent.save()
                    
    return np.sum(episodic_returns)

