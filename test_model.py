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
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from drl_framework.custom_env import *
from drl_framework.dqn import *
from drl_framework.params import *

import train_model_dqn as model_dqn
import train_model_drqn as model_drqn

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
else:
    device = torch.device("cpu")
    
target_key = ['available_computation_units',
        #   'channel_quality',
            'remain_epochs',
            'mec_comp_units',
            'mec_proc_times',
            'queue_comp_units',
            'queue_proc_times']
    
# Make test model
def test_model(env, model=None, iterations=MAX_EPOCH_SIZE, simmode="dqn"):
    df = pd.DataFrame()
    rewards = []
    for iter in tqdm(range(iterations)):
        next_state, _ = env.reset()
        reward = 0
        for i in range(MAX_EPOCH_SIZE):
            next_state = {key: next_state[key] for key in target_key}
            next_state = env.flatten_dict_values(next_state)
            # next_state = np.delete(next_state, 2)
            if simmode == "dqn":
                q_values = model.forward(torch.tensor(next_state, dtype=torch.float32, device=device))
                selected_action = q_values.max(0)[1].view(1, 1)
            elif simmode == "drqn":
                h = torch.zeros(1, 64, device=device)
                c = torch.zeros(1, 64, device=device)
                q_values, h, c = model.forward(torch.tensor(next_state, dtype=torch.float32, device=device).view(1, -1), h, c)
                selected_action = q_values.squeeze().max(0)[1].view(1, 1)
            # selected_action = 1 only if the channel quality is good
            elif simmode == "offload_only":
                selected_action = torch.tensor([1], dtype=torch.int64, device=device)
            elif simmode == "local_only":
                selected_action = torch.tensor([0], dtype=torch.int64, device=device)
            else:
                raise ValueError("Invalid simmode")
            # print(f"selected_action: {selected_action}")
            next_state, reward_inst, _, _, _ = env.step(selected_action.item())
            # print(f"next_state: {next_state}")
            reward += reward_inst
            df_data = pd.DataFrame(data=[next_state.values()], columns=next_state.keys(), index=[iter])
            df_misc = pd.DataFrame(data=[[i, selected_action.item(), reward_inst, reward]], 
                               columns=['epoch', 'action', 'reward_inst', 'reward'],
                               index=[iter])
            df1 = pd.concat([df_misc, df_data], axis=1)
            df = pd.concat([df, df1], axis=0)
        rewards.append(reward)
            
    return df, rewards
    
# Define test env and test model    
test_env = CustomEnv(max_comp_units=MAX_COMP_UNITS,
                    max_epoch_size=MAX_EPOCH_SIZE,
                    max_queue_size=MAX_QUEUE_SIZE,
                    reward_weights=REWARD_WEIGHTS)
state, _ = test_env.reset()
state = {key: state[key] for key in target_key}
n_observation = len(test_env.flatten_dict_values(state))
n_actions = test_env.action_space.n
dqn = model_dqn.Q_net(state_space=n_observation, action_space=n_actions).to(device)
drqn = model_drqn.Q_net(state_space=n_observation, action_space=n_actions).to(device)
# Load trained models
dqn.load_state_dict(torch.load("DQN_POMDP_SEED_1.pth", map_location=device))
drqn.load_state_dict(torch.load("DRQN_POMDP_Random_SEED_1.pth", map_location=device))
dqn.eval()
drqn.eval()

for i, simmode in enumerate(["dqn", "drqn", "offload_only", "local_only"]):
    if simmode == "dqn":
        model = dqn
    elif simmode == "drqn":
        model = drqn
    else:
        model = None
    df, rewards = test_model(test_env, model=model, iterations=200, simmode=simmode)
    filename = "test_log_" + simmode + ".csv"
    df.to_csv(filename)

    # Plot rewards
    plt.figure()
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.xlabel('Episode #')
    plt.ylabel('Return')
    plt.plot(rewards_t.numpy())

    means = rewards_t.unfold(0, 20, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(19), means))
    plt.plot(means.numpy())
    # Save plot into files
    filename = "test_rewards_" + simmode + ".png"
    plt.savefig(filename)

# plt.show()