import time
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import numpy as np


class TrainingLogger:
    """Logger for federated training data"""
    def __init__(self, num_agents, scheme_name):
        self.num_agents = num_agents
        self.scheme_name = scheme_name
        self.data = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
    def log_step(self, episode, epoch, states, actions, rewards, cloud_remaining):
        """Log training data for one step"""
        log_entry = {
            'episode': episode,
            'epoch': epoch,
            'cloud_remaining_units': cloud_remaining
        }
        
        # Log data for each agent
        for i in range(self.num_agents):
            # Log states
            log_entry.update({
                f'agent{i}_comp_units': states[i]['available_computation_units'],
                f'agent{i}_channel_quality': states[i]['channel_quality'],
                f'agent{i}_remain_epochs': states[i]['remain_epochs'],
                f'agent{i}_power': states[i]['power'],
                f'agent{i}_action': actions[i],
                f'agent{i}_reward': rewards[i]
            })
            
            # Log MEC computation units
            for j, comp_unit in enumerate(states[i]['mec_comp_units']):
                log_entry[f'agent{i}_mec_comp_unit_{j}'] = comp_unit
        
        self.data.append(log_entry)
    
    def save_to_csv(self):
        """Save logged data to CSV file"""
        df = pd.DataFrame(self.data)
        
        # Create filename with timestamp, scheme name, and number of nodes
        filename = f'logs/training_log_{self.scheme_name}_n{self.num_agents}_{self.timestamp}.csv'
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Training log saved to {filename}")
        
        # Clear data after saving
        self.data = []
        
        return filename
    
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
    # Save the plot with timestamp
    plot_filename = f'training_reward_plot_{TIMESTAMP}.png'
    plt.savefig(plot_filename)
    
    
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