import time
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def flatten_dict_values(dict):
        flattened = np.array([])
        for v in list(dict.values()):
            if isinstance(v, np.ndarray):
                flattened = np.concatenate([flattened, v])
            else:
                flattened = np.concatenate([flattened, np.array([v])])
        return flattened

def measure_time(func):
    """Decorator to measure execution time of a function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def plot_rewards(single_agent_rewards, federated_rewards, sync_interval=10):
    """Plot training rewards for single agent and federated agents
    
    Args:
        single_agent_rewards: List of rewards from single agent training
        federated_rewards: List of rewards from federated training
        sync_interval: Number of episodes between synchronizations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(single_agent_rewards, label='Individual Agents', alpha=0.8)
    plt.plot(federated_rewards, label='Federated Agents', alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for federation sync points
    for i in range(0, len(federated_rewards), sync_interval):
        plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
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