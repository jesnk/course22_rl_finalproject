import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


episodes = 1000  # number of episodes to run
initialize = 300  # initial time steps before start updating

from chain_mdp import ChainMDP
from agent_chainMDP import agent
env = ChainMDP(10)
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
agent.save("./saved_models/chain/")