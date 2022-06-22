
import numpy as np
from tqdm import tqdm
import random
import numpy as np
import torch
import argparse

import os
import sys
from chain_mdp import ChainMDP


env_num = 1

if env_num == 0 :
    from team9.agent_chainMDP import agent as Agent
    agent = Agent()
    n = 10
    env = ChainMDP(n)
elif env_num == 1 :
    from team9.agent_lava import agent as Agent
    agent = Agent()
    Env = __import__(f'lava_grid', fromlist=['ZigZag6x10']).ZigZag6x10
    env_str = 'lava'
    env_kwargs = {"max_steps":100, "act_fail_prob":0, "goal":(5, 9), "numpy_state":False}
    env = Env(**env_kwargs)
    n = 60


if __name__ == "__main__" :

    if env_num == 0 :                
        episodes = 1000
        train_step = 20
    elif env_num == 1 :
        episodes = 2000
        train_step = 100


    episodic_returns = []
    timestep = 0
    for epi in tqdm(range(episodes)):
        
        s = env.reset()
        done = False
        cum_reward = 0.0 

        while not done:  
            timestep += 1
            action = agent.action(s)
            ns, reward, done, _ = env.step(action)
            cum_reward += reward
            # saving reward and is_terminals
            if s.any():
                experience = [reward, done, s, ns]
                agent.buffer.update(experience)
            s = ns
            #####################
            # If your agent needs to update the weights at every time step, complete your update process in this area.
            # e.g., agent.update()
            if (timestep) % train_step ==0:
                # print("Train!!!!!!!")
                agent.train()
        agent.buffer.trace_init()
        #####################
        # elif your agent needs to update the weights at the end of every episode, complete your update process in this area.
        # e.g., agent.update()


        #####################
        episodic_returns.append(cum_reward)
        if cum_reward > 0 :
            print(cum_reward)
    agent.save()
