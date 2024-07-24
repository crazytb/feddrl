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
# from drl_framework.dqn import *
from drl_framework.drqn import *
from drl_framework.params import *

env = CustomEnv(max_comp_units=MAX_COMP_UNITS,
                max_terminals=MAX_TERMINALS, 
                max_epoch_size=MAX_EPOCH_SIZE,
                max_queue_size=MAX_QUEUE_SIZE,
                reward_weights=REWARD_WEIGHTS)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
print("device: ", device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Parameters
HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 10

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
state = env.flatten_dict_values(state)
n_observations = len(state)

policy_net = DRQN(n_observations, HIDDEN_SIZE, n_actions).to(device)
target_net = DRQN(n_observations, HIDDEN_SIZE, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0

episode_durations = []

def select_action(state, hidden):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            state = state.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            q_values, new_h, new_c = policy_net(state, hidden)
            return q_values.max(1)[1].view(1, 1), (new_h, new_c)
    else:
        h = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
        c = torch.zeros(1, 1, HIDDEN_SIZE, device=device)
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), (h, c)


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).unsqueeze(1)  # Add sequence dimension
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values, _ = policy_net(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_batch = non_final_next_states.unsqueeze(1)  # Add sequence dimension
        next_state_values[non_final_mask], _ = target_net(next_state_batch)
        next_state_values[non_final_mask] = next_state_values[non_final_mask].max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if device == torch.device("cuda") or device == torch.device("mps"):
    num_episodes = 2000
else:
    num_episodes = 50

# Main training loop
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(env.flatten_dict_values(state), dtype=torch.float32, device=device)
    hidden = None
    episode_reward = 0
    
    for t in range(MAX_EPOCH_SIZE):
        action, hidden = select_action(state, hidden)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(env.flatten_dict_values(observation), dtype=torch.float32, device=device)

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(episode_reward)
            # plot_durations()
            break
    
print('Complete')

# Save the model
torch.save(policy_net.state_dict(), 'policy_net_drqn.pth')