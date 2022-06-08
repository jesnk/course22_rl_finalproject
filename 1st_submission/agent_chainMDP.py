import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle


import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import gym

class Qfunction(keras.Model):   
    def __init__(self, obssize, actsize, hidden_dims):
        super(Qfunction, self).__init__()
        initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)
        self.input_layer = keras.layers.InputLayer(input_shape=(obssize,))
        self.hidden_layers = [] #10.5
        for hidden_dim in hidden_dims:
            layer = keras.layers.Dense(hidden_dim, activation='relu',
                                      kernel_initializer=initializer)
            self.hidden_layers.append(layer)
        self.output_layer = keras.layers.Dense(actsize) 
    
    @tf.function
    def call(self, states):
        x = self.input_layer(states)
        for i, layer in enumerate(self.hidden_layers):
            x = self.hidden_layers[i](x)
        q_value = self.output_layer(x)

        return q_value


class DQN(object):
    
    def __init__(self, obssize, actsize, hidden_dims, optimizer, load_model = None):
        if load_model is not None :
            print("trying load "+load_model)
            self.qfunction = tf.keras.models.load_model(load_model,compile=False)
        else :
            self.qfunction = Qfunction(obssize, actsize, hidden_dims)
        self.optimizer = optimizer
        self.obssize = obssize
        self.actsize = actsize

    def save(self,name) :
        self.qfunction.save(name)

    def _predict_q(self, states, actions):
        q_values = self.compute_Qvalues(states)
        q_values_about_actions = tf.gather(q_values, actions.reshape(-1, 1), axis=1, batch_dims=1)
        return q_values_about_actions

    def _loss(self, Qpreds, targets):
        return tf.math.reduce_mean(tf.square(Qpreds - targets))

    
    def compute_Qvalues(self, states):
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)


    def train(self, states, actions, targets):
        with tf.GradientTape() as tape:
            Qpreds = self._predict_q(states, actions)
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):
        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables
        
        for v1, v2 in zip(from_var, to_var):
            v2.assign(v1)

class ReplayBuffer(object):
    
    def __init__(self, maxlength):
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength
    
    def append(self, experience):
        self.buffer.append(experience)
        self.number += 1
        if(self.number > self.maxlength):
            self.pop()
        
    def pop(self):
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):
        inds = np.random.choice(len(self.buffer), batchsize, replace=False)
        return [self.buffer[idx] for idx in inds]
        

class agent():
    
    def __init__(self, load_model="./saved_models/chain/"): # For evaluation, load model
        
        obssize = 10
        actsize = 2
        lr = 5e-4
        maxlength = 10000
        self.tau = 100
        initialize = 500
        self.epsilon = 1
        self.epsilon_decay = .995
        self.batchsize = 64
        self.gamma = .999
        hidden_dims=[10,5]
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
    
        if load_model is not None :
            print("load model : ", load_model)
            self.Qprincipal = DQN(obssize, actsize, hidden_dims, self.optimizer, load_model=load_model+"Qprincipal")
            self.Qtarget = DQN(obssize, actsize, hidden_dims, self.optimizer, load_model=load_model+"Qtarget")
            self.Qtarget.update_weights(self.Qprincipal)
            with open(load_model+'Buffer', 'rb') as f :
                self.buffer = pickle.load(f)
            with open(load_model+'Infos', 'rb') as f :
                self.epsilon = pickle.load(f)['epsilon']
        else :
            print("initialize model")
            self.Qprincipal = DQN(obssize, actsize, hidden_dims, self.optimizer)
            self.Qtarget = DQN(obssize, actsize, hidden_dims, self.optimizer)
            self.Qtarget.update_weights(self.Qprincipal)
            self.buffer = ReplayBuffer(maxlength)        

    def action(self, obs=None, eps_on=True):
        if not obs.any() :
            return random.randrange(0,2)
        if eps_on :
            if np.random.rand() < self.epsilon :
                action = random.randrange(0,2)
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
        
    def save(self, name="test") :
        self.Qprincipal.save(name+"Qprincipal")
        self.Qtarget.save(name+"Qtarget")
        
        infos = {}
        infos['epsilon'] = self.epsilon
        
        with open(name+"Buffer", 'wb') as f :
            pickle.dump(self.buffer,f)

        with open(name+"Infos",'wb') as f :
            pickle.dump(infos,f)