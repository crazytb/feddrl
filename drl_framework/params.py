import os
from datetime import datetime
import torch
import numpy as np

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Env parameters
# MAX_COMP_UNITS = 100
# MAX_TERMINALS = 10
MAX_EPOCH_SIZE = 10 #
MAX_QUEUE_SIZE = 20
REWARD_WEIGHTS = 0.1

# Env params
ENV_PARAMS = {
    # 'max_comp_units': np.random.randint(1, 101),  # Max computation units
    'max_comp_units': 100,  # Max computation units
    'max_epoch_size': MAX_EPOCH_SIZE,  # Max epoch size
    'max_queue_size': MAX_QUEUE_SIZE,  # Max queue size
    'reward_weights': REWARD_WEIGHTS,  # Reward weights
    # 'agent_velocities': np.random.randint(10, 101)  # Agent velocities
    'agent_velocities': 50  # Agent velocities
    }

# DQN parameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Set parameters
batch_size = 1
learning_rate = 1e-4 #
buffer_len = int(100000)
min_buffer_len = 20
min_epi_num = 20 # Start moment to train the Q network
episodes = 200 #
print_per_iter = 20
target_update_period = 10 #
eps_start = 0.1
eps_end = 0.01 #
eps_decay = 0.998 #
tau = 1e-2
max_step = 20

# DRQN param
random_update = False # If you want to do random update instead of sequential update
lookup_step = 20 # If you want to do random update instead of sequential update
max_epi_len = 100 
max_epi_step = max_step

