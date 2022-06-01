from chain_mdp import ChainMDP
from agent_chainMDP import agent


# recieve 1 at rightmost stae and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()


""" Your agent"""
agent = agent()

done = False
cum_reward = 0.0
# always move right left: 0, right: 1
action = 1
while not done:    
    action = agent.action()
    ns, reward, done, _ = env.step(action)
    cum_reward += reward
print(f"total reward: {cum_reward}")
