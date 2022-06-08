import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib.pyplot as plt


episodes = 1000  # number of episodes to run
initialize = 300  # initial time steps before start updating

from chain_mdp import ChainMDP
from agent_lava import agent
from lava_grid import ZigZag6x10
# default setting
max_steps = 100
stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
# recieve 1 at rightmost stae and recieve small reward at leftmost state
env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)
agent = agent(load_model=None)
s = env.reset()
rrecord = []
totalstep = 0
for ite in range(episodes):
    obs = env.reset()
    done = False
    rsum = 0
    while not done:
        totalstep +=1
        action = agent.action(obs)
        next_obs,reward,done,info = env.step(action)
        rsum += reward
        try :
            if obs == 0 :
                if totalstep>initialize:
                    agent.train(totalstep)
                obs = next_obs    

        except ValueError :
            pass
        experience = (obs,action,reward,next_obs,done)
        agent.buffer.append(experience)

        if totalstep>initialize:
            agent.train(totalstep)
        obs = next_obs
                        
################################################################################
    ## DO NOT CHANGE THIS PART!
    rrecord.append(rsum)
    if ite % 200 == 0:
        print('iteration {} ave reward {}'.format(ite, np.mean(rrecord[-10:])))
    
    ave100 = np.mean(rrecord[-100:])   
    if  ave100 > 17.5:
        print("Solved after %d episodes."%ite)
        break
agent.save("./saved_models/lava/")