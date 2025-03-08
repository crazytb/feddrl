import torch
import numpy as np
import matplotlib.pyplot as plt
from drl_framework.networks import LocalNetwork
from drl_framework.trainer import train_single_agent, train_federated_agents
from drl_framework.custom_env import CustomEnv, CloudController
from drl_framework.utils import seed_torch, flatten_dict_values
from drl_framework.params import device
import time
from typing import List, Tuple
import pandas as pd
from datetime import datetime

def run_convergence_experiment(
    n_trials: int = 5,
    n_agents: int = 5,
    n_episodes: int = 200,
    sync_interval: int = 10,
    learning_rate: float = 0.001,
    hidden_dim: int = 16
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Run multiple trials comparing individual and federated learning convergence.
    
    Returns:
        Tuple containing lists of rewards for individual and federated learning trials
    """
    individual_rewards = []
    federated_rewards = []
    
    # Environment setup parameters
    max_available_computation_units = [10]*1 + [50]*2 + [100]*2
    agent_velocities = [10]*1 + [20]*2 + [30]*2
    channel_patterns = ['urban']*1 + ['suburban']*2 + ['rural']*2

    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Create environments
        cloud_controller = CloudController(max_comp_units=100)
        envs = [CustomEnv(
            max_comp_units=10,
            max_available_computation_units=max_available_computation_units[i],
            max_epoch_size=10,
            max_power=10,
            agent_velocity=agent_velocities[i],
            channel_pattern=channel_patterns[i],
            cloud_controller=cloud_controller
        ) for i in range(n_agents)]

        # Get dimensions from first environment
        env = envs[0]
        state_dim = len(flatten_dict_values(env.observation_space.sample()))
        action_dim = env.action_space.n

        # Individual Learning
        print("Running Individual Learning...")
        agents_sin = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) 
                     for _ in range(n_agents)]
        optimizers_sin = [torch.optim.Adam(agent.parameters(), lr=learning_rate) 
                         for agent in agents_sin]
        rewards_sin = train_federated_agents(
            envs=envs,
            agents=agents_sin,
            optimizers=optimizers_sin,
            device=device,
            episodes=n_episodes,
            sync_interval=n_episodes,
            hidden_dim=hidden_dim,
            averaging_scheme='fedavg',
            cloud_controller=cloud_controller,
        )
        individual_rewards.append(np.mean(rewards_sin, axis=0))

        # Federated Learning
        print("Running Federated Learning...")
        agents_fed = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) 
                     for _ in range(n_agents)]
        optimizers_fed = [torch.optim.Adam(agent.parameters(), lr=learning_rate) 
                         for agent in agents_fed]
        rewards_fed = train_federated_agents(
            envs=envs,
            agents=agents_fed,
            optimizers=optimizers_fed,
            device=device,
            episodes=n_episodes,
            sync_interval=sync_interval,
            hidden_dim=hidden_dim,
            averaging_scheme='fedavg',
            cloud_controller=cloud_controller,
        )
        federated_rewards.append(np.mean(rewards_fed, axis=0))

    return individual_rewards, federated_rewards

def plot_convergence_comparison(
    individual_rewards: List[np.ndarray],
    federated_rewards: List[np.ndarray],
    window_size: int = 10
) -> None:
    """
    Plot the convergence comparison with confidence intervals.
    """
    # Convert to numpy arrays
    individual_data = np.array(individual_rewards)
    federated_data = np.array(federated_rewards)
    
    # Calculate means and standard deviations
    individual_mean = np.mean(individual_data, axis=0)
    federated_mean = np.mean(federated_data, axis=0)
    individual_std = np.std(individual_data, axis=0)
    federated_std = np.std(federated_data, axis=0)
    
    # Smooth the curves
    def smooth(data, window_size):
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    
    individual_mean_smooth = smooth(individual_mean, window_size)
    federated_mean_smooth = smooth(federated_mean, window_size)
    individual_std_smooth = smooth(individual_std, window_size)
    federated_std_smooth = smooth(federated_std, window_size)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    episodes = range(len(individual_mean))
    
    # Plot means and confidence intervals
    plt.plot(episodes, individual_mean_smooth, label='Individual Learning', color='blue')
    plt.fill_between(episodes, 
                    individual_mean_smooth - individual_std_smooth,
                    individual_mean_smooth + individual_std_smooth,
                    alpha=0.2, color='blue')
    
    plt.plot(episodes, federated_mean_smooth, label='Federated Learning', color='red')
    plt.fill_between(episodes, 
                    federated_mean_smooth - federated_std_smooth,
                    federated_mean_smooth + federated_std_smooth,
                    alpha=0.2, color='red')
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Convergence Comparison: Individual vs Federated Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'convergence_comparison_{timestamp}.png')
    plt.close()

def analyze_convergence_metrics(
    individual_rewards: List[np.ndarray],
    federated_rewards: List[np.ndarray],
    threshold: float = 0.9
) -> dict:
    """
    Calculate various convergence metrics for comparison.
    """
    individual_data = np.array(individual_rewards)
    federated_data = np.array(federated_rewards)
    
    # Calculate final performance (average of last 10% of episodes)
    final_window = int(individual_data.shape[1] * 0.1)
    individual_final = np.mean(individual_data[:, -final_window:])
    federated_final = np.mean(federated_data[:, -final_window:])
    
    # Calculate episodes to convergence (90% of final performance)
    individual_target = individual_final * threshold
    federated_target = federated_final * threshold
    
    individual_convergence = [np.where(trial >= individual_target)[0][0] 
                            if any(trial >= individual_target) else -1 
                            for trial in individual_data]
    federated_convergence = [np.where(trial >= federated_target)[0][0] 
                            if any(trial >= federated_target) else -1 
                            for trial in federated_data]
    
    return {
        'individual_final_performance': individual_final,
        'federated_final_performance': federated_final,
        'individual_episodes_to_convergence': np.mean([ep for ep in individual_convergence if ep != -1]),
        'federated_episodes_to_convergence': np.mean([ep for ep in federated_convergence if ep != -1]),
        'individual_convergence_std': np.std([ep for ep in individual_convergence if ep != -1]),
        'federated_convergence_std': np.std([ep for ep in federated_convergence if ep != -1])
    }

def main():
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Run experiment
    print("Starting convergence comparison experiment...")
    start_time = time.time()
    
    individual_rewards, federated_rewards = run_convergence_experiment(
        n_trials=5,
        n_agents=5,
        n_episodes=200,
        sync_interval=10
    )
    
    # Plot results
    plot_convergence_comparison(individual_rewards, federated_rewards)
    
    # Analyze and print metrics
    metrics = analyze_convergence_metrics(individual_rewards, federated_rewards)
    print("\nConvergence Analysis Results:")
    print(f"Individual Learning:")
    print(f"  Final Performance: {metrics['individual_final_performance']:.2f}")
    print(f"  Episodes to Convergence: {metrics['individual_episodes_to_convergence']:.2f} ± {metrics['individual_convergence_std']:.2f}")
    print(f"\nFederated Learning:")
    print(f"  Final Performance: {metrics['federated_final_performance']:.2f}")
    print(f"  Episodes to Convergence: {metrics['federated_episodes_to_convergence']:.2f} ± {metrics['federated_convergence_std']:.2f}")
    
    print(f"\nTotal experiment time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()