import tensorflow as tf
from tensorflow import keras

from collections import deque
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import gym
# define neural net Q_\theta(s,a) as a class

class Qfunction(keras.Model):
    
    def __init__(self, obssize, actsize, hidden_dims):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        hidden_dims: list containing output dimension of hidden layers 
        """
        super(Qfunction, self).__init__()

        # Layer weight initializer
        initializer = keras.initializers.RandomUniform(minval=-1., maxval=1.)

        # Input Layer
        self.input_layer = keras.layers.InputLayer(input_shape=(obssize,))
        
        # Hidden Layer
        self.hidden_layers = [] #10.5
        for hidden_dim in hidden_dims:
            # TODO: define each hidden layers
            layer = keras.layers.Dense(hidden_dim, activation='relu',
                                      kernel_initializer=initializer)
            
            self.hidden_layers.append(layer)
            
            
            
        
        # Output Layer : 
        # TODO: Define the output layer.
        self.output_layer = keras.layers.Dense(actsize) 
    


    @tf.function
    def call(self, states):
        ########################################################################
        # TODO: You SHOULD implement the model's forward pass
        x = self.input_layer(states)
        for i, layer in enumerate(self.hidden_layers):
            x = self.hidden_layers[i](x)
        q_value = self.output_layer(x)

        return q_value
        ########################################################################


# Wrapper class for training Qfunction and updating weights (target network) 

class DQN(object):
    
    def __init__(self, obssize, actsize, hidden_dims, optimizer, load_model = None):
        """
        obssize: dimension of state space
        actsize: dimension of action space
        optimizer: 
        """
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
        """
        states represent s_t
        actions represent a_t
        """
        ########################################################################
        # TODO: Define the logic for calculate  Q_\theta(s,a
        q_values = self.compute_Qvalues(states)
        q_values_about_actions = tf.gather(q_values, actions.reshape(-1, 1), axis=1, batch_dims=1)
        return q_values_about_actions
        ########################################################################
        

    def _loss(self, Qpreds, targets):
        """
        Qpreds represent Q_\theta(s,a)
        targets represent the terms E[r+gamma Q] in Bellman equations

        This function is OBJECTIVE function
        """
        return tf.math.reduce_mean(tf.square(Qpreds - targets))

    
    def compute_Qvalues(self, states):
        """
        states: numpy array as input to the neural net, states should have
        size [numsamples, obssize], where numsamples is the number of samples
        output: Q values for these states. The output should have size 
        [numsamples, actsize] as numpy array
        """
        inputs = np.atleast_2d(states.astype('float32'))
        return self.qfunction(inputs)


    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        with tf.GradientTape() as tape:
            Qpreds = self._predict_q(states, actions)
            loss = self._loss(Qpreds, targets)
        variables = self.qfunction.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def update_weights(self, from_network):
        """
        We need a subroutine to update target network 
        i.e. to copy from principal network to target network. 
        This function is for copying  𝜃←𝜃target 
        """
        
        from_var = from_network.qfunction.trainable_variables
        to_var = self.qfunction.trainable_variables
        
        for v1, v2 in zip(from_var, to_var):
            v2.assign(v1)


            # Implement replay buffer
class ReplayBuffer(object):
    
    def __init__(self, maxlength):
        """
        maxlength: max number of tuples to store in the buffer
        if there are more tuples than maxlength, pop out the oldest tuples
        """
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength
    
    def append(self, experience):
        """
        this function implements appending new experience tuple
        experience: a tuple of the form (s,a,r,s^\prime)
        """
        self.buffer.append(experience)
        self.number += 1
        if(self.number > self.maxlength):
            self.pop()
        
    def pop(self):
        """
        pop out the oldest tuples if self.number > self.maxlength
        """
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1
    
    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,s^\prime)
        """
        inds = np.random.choice(len(self.buffer), batchsize, replace=False)
        return [self.buffer[idx] for idx in inds]
        