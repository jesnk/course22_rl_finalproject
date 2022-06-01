from models import dqn
import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class agent():
    
    def __init__(self, env=None):
        self.env = env
        obssize = 10
        actsize = env.action_space.n
        lr = 5e-4
        maxlength = 10000
        self.tau = 100
        initialize = 500
        self.epsilon = 1
        self.epsilon_decay = .995
        self.batchsize = 64
        self.gamma = .999
        hidden_dims=[10,5]
        # optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        # initialize networks
        self.Qprincipal = dqn.DQN(obssize, actsize, hidden_dims, self.optimizer)
        self.Qtarget = dqn.DQN(obssize, actsize, hidden_dims, self.optimizer)

        # initialization of buffer
        self.buffer = dqn.ReplayBuffer(maxlength)        
        self.Qtarget.update_weights(self.Qprincipal)

    
    def action(self, obs=None, eps_on=True):
        if not obs.any() :
            return self.env.action_space.sample()

        if eps_on :
            if np.random.rand() < self.epsilon :
                action = self.env.action_space.sample()
            else :
                action = np.argmax(self.Qprincipal.compute_Qvalues(obs))
        
        return action

    def train(self,totalstep) :
        samples = self.buffer.sample(self.batchsize)
        curr_obs = np.stack([sample[0] for sample in samples])
        curr_act = np.expand_dims(np.stack([sample[1] for sample in samples]), axis = 1)
        curr_rew = np.stack([sample[2] for sample in samples])
        curr_nobs = np.stack([sample[3] for sample in samples])
        curr_done = np.stack([sample[4] for sample in samples])
        tmp = self.Qtarget.compute_Qvalues(curr_nobs)
        tmp = np.max(tmp,axis=1)*(1-curr_done)
        target_d = np.expand_dims(curr_rew+tmp*self.gamma,axis=1)
        loss = self.Qprincipal.train(curr_obs,curr_act,target_d)
        
        if totalstep % self.tau == 0:
            self.Qtarget.update_weights(self.Qprincipal)
        
        self.epsilon*= self.epsilon_decay
        