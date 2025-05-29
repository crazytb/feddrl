import gymnasium as gym
import torch
import json
from drl_framework.networks import LocalNetwork
from drl_framework.trainer import train_individual_agent, train_federated_agents
from drl_framework.utils import *
from drl_framework.custom_env import *
from drl_framework.params import device
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

def save_models(agents, model_type, num_agents, timestamp):
    """Save trained models to models directory"""
    models_dir = f"models/{model_type}_{num_agents}agents_{timestamp}"
    os.makedirs(models_dir, exist_ok=True)
    
    for i, agent in enumerate(agents):
        model_path = os.path.join(models_dir, f"agent_{i+1}.pth")
        torch.save({
            'model_state_dict': agent.state_dict(),
            'model_architecture': {
                'state_dim': agent.mlp.fc1.in_features,
                'action_dim': agent.q_head.out_features,
                'hidden_dim': agent.mlp.fc1.out_features
            }
        }, model_path)
        print(f"Saved {model_type} Agent {i+1} to {model_path}")
    
    return models_dir

def main():
    
    print(f"Device: {device}")
    
    # Set the seed
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Environment and hyperparameters
    num_agents = NUM_AGENTS
    
    # Create diverse environments for each agent
    envs = [
        CustomEnv(
            max_comp_units=np.random.randint(1, 101),  # 1 to 100
            max_epoch_size=10,
            max_queue_size=10,
            reward_weights=1,
            agent_velocities=np.random.randint(10, 101)  # 10 to 100
        ) for _ in range(num_agents)
    ]
    
    # Print environment configurations for each agent
    for i, env in enumerate(envs):
        print(f"Agent {i+1} Environment:")
        print(f"  - Max Computation Units: {env.max_comp_units}")
        print(f"  - Agent Velocities: {env.agent_velocities}")
        print()
    
    # Use the first environment to determine state dimensions (assuming same structure)
    state_dim = len(flatten_dict_values(envs[0].observation_space.sample()))
    action_dim = envs[0].action_space.n
    hidden_dim = 8
    episodes = 500
    learning_rate = 0.001
    sync_interval = 100

    # Independent Agent Training (each agent trains independently without federated learning)
    independent_agents = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
    independent_optimizers = [torch.optim.Adam(agent.parameters(), lr=learning_rate) for agent in independent_agents]
    independent_agent_rewards = train_individual_agent(
        envs=envs,  # Use same diverse environments
        agents=independent_agents,
        optimizers=independent_optimizers,
        device=device,
        episodes=episodes
    )

    # Federated Agents Training with diverse environments
    federated_agents = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) for _ in range(num_agents)]
    federated_optimizers = [torch.optim.Adam(agent.parameters(), lr=learning_rate) for agent in federated_agents]
    federated_rewards = train_federated_agents(
        envs=envs,  # Pass list of environments
        agents=federated_agents,
        optimizers=federated_optimizers,
        device=device,
        episodes=episodes,
        sync_interval=sync_interval
    )

    # Save trained models
    timestamp = TIMESTAMP
    independent_models_dir = save_models(independent_agents, "independent", num_agents, timestamp)
    federated_models_dir = save_models(federated_agents, "federated", num_agents, timestamp)
    
    # Save environment configurations
    print(f"\nSaving environment configurations...")
    env_config = {
        'num_agents': num_agents,
        'episodes': episodes,
        'sync_interval': sync_interval,
        'envs': []
    }
    
    for i, env in enumerate(envs):
        env_config['envs'].append({
            'agent_id': i + 1,
            'max_comp_units': int(env.max_comp_units),
            'max_epoch_size': int(env.max_epoch_size),
            'max_queue_size': int(env.max_queue_size),
            'reward_weights': float(env.reward_weight),
            'agent_velocities': int(env.agent_velocities)
        })
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    env_config_path = f"models/env_configs_{num_agents}agents_{timestamp}.json"
    try:
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)
        print(f"✅ Environment config saved to: {env_config_path}")
    except Exception as e:
        print(f"❌ Failed to save environment config: {e}")
    
    # Also save as readable text file
    env_config_txt_path = f"models/env_configs_{num_agents}agents_{timestamp}.txt"
    try:
        with open(env_config_txt_path, 'w') as f:
            f.write(f"Training Configuration - {timestamp}\n")
            f.write(f"Number of Agents: {num_agents}\n")
            f.write(f"Episodes: {episodes}\n")
            f.write(f"Sync Interval: {sync_interval}\n\n")
            for env_data in env_config['envs']:
                f.write(f"Agent {env_data['agent_id']} Environment:\n")
                f.write(f"  - Max Computation Units: {env_data['max_comp_units']}\n")
                f.write(f"  - Agent Velocities: {env_data['agent_velocities']}\n")
                f.write(f"  - Max Epoch Size: {env_data['max_epoch_size']}\n")
                f.write(f"  - Max Queue Size: {env_data['max_queue_size']}\n")
                f.write(f"  - Reward Weights: {env_data['reward_weights']}\n\n")
        print(f"✅ Environment config text saved to: {env_config_txt_path}")
    except Exception as e:
        print(f"❌ Failed to save environment config text: {e}")
    
    print(f"\nEnvironment configurations saved to: {env_config_path}")
    print(f"Independent models saved to: {independent_models_dir}")
    print(f"Federated models saved to: {federated_models_dir}")

    # Plot the results
    plot_rewards(np.mean(independent_agent_rewards, axis=0), np.mean(federated_rewards, axis=0), sync_interval)
    print("Training Complete")

if __name__ == "__main__":
    main()