import sys
sys.path.append("./team9")
## reference
## https://github.com/nikhilbarhate99/PPO-PyTorch

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import random

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self, obs_size=10):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.obs_size = obs_size

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def update(self, experience) :
        reward, done, s, ns = experience        
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def trace_init(self) :
        pass

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(), #Tanh(),
                        #nn.Linear(64, 64),
                        #nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.ReLU(),
                        #nn.Linear(64, 64),
                        #nn.ReLU(),
                        nn.Linear(64, 1)
                    )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        # print(state.shape)
        action_probs = self.actor(state)
        # print(action_probs)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class agent():
    
    def __init__(self, load_model=None): # For evaluation, load model
        
        self.obssize = 10
        self.actsize = 2
        self.gamma = .999
        self.eps_clip = 0.2
        self.K_epochs = 20
        self.lr_actor = 0.0003 *0.1
        self.lr_critic = 0.001 *0.1
        
        self.MseLoss = nn.MSELoss()
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.obssize, self.actsize).to(device)
        self.policy_old = ActorCritic(self.obssize, self.actsize).to(device)
    
        if load_model is not None :
            self.load_weights()
        else :
            print("initialize model")            
            self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        

    def load_weights(self) :
        load_model = "./saved_models/chain/"+"model.ckpt"
        print("load model : ", load_model)
        self.policy_old.load_state_dict(torch.load(load_model, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(load_model, map_location=lambda storage, loc: storage))       

    def action(self, obs=None, eps_on=True):
        if not obs.any() :
            return random.randrange(0,2)
        
        with torch.no_grad():
            state = torch.FloatTensor(obs).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def train(self) :
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.scheduler.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
    def save(self, name="test") :
        load_model = "./team9/saved_models/chain/"
        torch.save(self.policy_old.state_dict(), load_model+"model.ckpt")
        print("model saved")
