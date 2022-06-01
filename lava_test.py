from models import dqn
import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
from lava_grid import ZigZag6x10
from agent_lava import agent
import random

max_steps = 100
stochasticity = 0 # probability of the selected action failing and instead executing any of the remaining 3
no_render = True

env = ZigZag6x10(max_steps=max_steps, act_fail_prob=stochasticity, goal=(5, 9), numpy_state=False)


episodes = 3000  # number of episodes to run
initialize = 500  # initial time steps before start updating

# recieve 1 at rightmost stae and recieve small reward at leftmost state
agent = agent(env)
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
        if type(obs) == np.int64 :
            obs = np.zeros((60,)).tolist()
            #print(obs, "test")
        rsum += reward
        experience = (obs,action,reward,next_obs,done)
        agent.buffer.append(experience)

        if totalstep>initialize:
            agent.train(totalstep)
        obs = next_obs

################################################################################
    ## DO NOT CHANGE THIS PART!
    rrecord.append(rsum)
    if ite % 1000 == 0:
        print('iteration {} ave reward {}'.format(ite, np.mean(rrecord[-10:])))
    
    ave100 = np.mean(rrecord[-100:])   
    if  ave100 >= 0:
        print("Solved after %d episodes."%ite)
        break