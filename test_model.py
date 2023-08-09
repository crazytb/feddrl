# https://www.youtube.com/@cartoonsondemand

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count, chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from drl_framework.custom_env import *
from drl_framework.dqn import *
from drl_framework.params import *

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)
    
# Make test model
def test_model(model, env, iterations=100):
    rewards = []
    for _ in range(iterations):
        next_state, _ = env.reset()
        for i in range(MAX_EPOCH_SIZE):
            reward = 0
            next_state = env.flatten_dict_values(next_state)
            selected_action = model.forward(torch.tensor(next_state, dtype=torch.float32, device=device)).max(0)[1].view(1, 1)
            # print(f"selected_action: {selected_action}")
            next_state, reward_inst, terminated, truncated, _ = env.step(selected_action.item())
            # print(f"next_state: {next_state}")
            reward += reward_inst
        rewards.append(reward)
            
    return rewards
    
# Define test env and test model    
test_env = CustomEnv(max_comp_units=MAX_COMP_UNITS,
                    max_terminals=MAX_TERMINALS,
                    max_epoch_size=MAX_EPOCH_SIZE,
                    max_queue_size=MAX_QUEUE_SIZE,
                    reward_weights=REWARD_WEIGHTS)
policy_net = torch.load("policy_model.pt")
policy_net.eval()
print("policy_net: ", policy_net)

rewards = test_model(policy_net, test_env)

# Plot rewards
plt.figure(2)
plt.clf()
rewards_t = torch.tensor(rewards, dtype=torch.float)
plt.title('Test Rewards')
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.plot(rewards_t.numpy())
plt.show()
