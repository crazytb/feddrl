import os
from datetime import datetime
import torch

# Env parameters
MAX_COMP_UNITS = 100
MAX_TERMINALS = 10
MAX_EPOCH_SIZE = 200 #
MAX_QUEUE_SIZE = 20
REWARD_WEIGHTS = 0.1

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

def get_fixed_timestamp():
    timestamp_file = '#timestamp.txt'
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            return f.read().strip()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(timestamp_file, 'w') as f:
            f.write(timestamp)
        return timestamp
    
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    save_path = os.path.join(models_dir, path)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
TIMESTAMP = get_fixed_timestamp()