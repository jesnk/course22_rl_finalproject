import numpy as np

def calculate_performance(episodes, env, agent):

    episodic_returns = []
    
    for epi in range(episodes):
        
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
    
    for epi in range(episodes):
        
        s = env.reset()

        done = False
        cum_reward = 0.0

        while not done:    
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            #s = ns
            
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            # e.g., agent.update()
            experience = (s, action, reward, ns, done)
            s = ns
            agent.buffer.append(experience)
            if epi > 10 :
                agent.train(epi)

            #####################
        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()


        #####################
        
        episodic_returns.append(cum_reward)
                    
    return np.sum(episodic_returns)

