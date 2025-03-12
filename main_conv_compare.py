import torch
import numpy as np
import matplotlib.pyplot as plt
from drl_framework.networks import LocalNetwork
from drl_framework.trainer import train_federated_agents
from drl_framework.custom_env import CustomEnv, CloudController
from drl_framework.utils import seed_torch, flatten_dict_values
from drl_framework.params import device
import time
import os
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
from torchinfo import summary
import json
import seaborn as sns

# Create a directory for saving models
MODELS_DIR = "saved_models"
RESULTS_DIR = "test_results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def measure_flops(state_dim: int, action_dim: int, hidden_dim: int) -> None:
    """
    Measure the number of FLOPs for a given network.
    """
    model = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
    flops_info = summary(model, input_size=(1, state_dim), device=device, verbose=0)
    print("\nModel FLOPs Analysis:")
    print(flops_info)

def train_and_save_models(
    sync_intervals: List[int],
    n_trials: int = 3,
    n_agents: int = 5,
    n_episodes: int = 300,
    learning_rate: float = 0.001,
    hidden_dim: int = 16,
    averaging_scheme: str = 'fedavg'
) -> Dict[int, List[str]]:
    """
    Train models with different sync intervals and save them to disk.
    
    Returns:
        Dictionary mapping sync intervals to lists of model paths
    """
    model_paths = {interval: [] for interval in sync_intervals}
    rewards_dict = {}
    computation_times = {}
    
    # Environment setup parameters
    max_available_computation_units = [10]*1 + [50]*2 + [100]*2
    agent_velocities = [10]*1 + [20]*2 + [30]*2
    channel_patterns = ['urban']*1 + ['suburban']*2 + ['rural']*2
    channel_pattern_change_interval = 100
    
    # Timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for sync_interval in sync_intervals:
        print(f"\n{'='*50}")
        print(f"Training with sync_interval = {sync_interval}")
        print(f"{'='*50}")
        
        interval_rewards = []
        interval_start_time = time.time()
        
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
                channel_pattern_change_interval=channel_pattern_change_interval,
                cloud_controller=cloud_controller
            ) for i in range(n_agents)]

            # Get dimensions from first environment
            env = envs[0]
            state_dim = len(flatten_dict_values(env.observation_space.sample()))
            action_dim = env.action_space.n
            
            # Measure FLOPs for network in the first trial of the first sync interval
            if trial == 0 and sync_interval == sync_intervals[0]:
                measure_flops(state_dim, action_dim, hidden_dim)

            # Create agents and optimizers
            agents = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) 
                        for _ in range(n_agents)]
            optimizers = [torch.optim.Adam(agent.parameters(), lr=learning_rate) 
                        for agent in agents]
            
            # Train agents
            print(f"Training with sync_interval={sync_interval}...")
            rewards = train_federated_agents(
                envs=envs,
                agents=agents,
                optimizers=optimizers,
                device=device,
                episodes=n_episodes,
                sync_interval=sync_interval,
                hidden_dim=hidden_dim,
                averaging_scheme=averaging_scheme,
                cloud_controller=cloud_controller,
            )
            interval_rewards.append(np.mean(rewards, axis=0))
            
            # Save each agent's model
            for i, agent in enumerate(agents):
                model_path = os.path.join(MODELS_DIR, f"agent_{i}_sync_{sync_interval}_trial_{trial}_{timestamp}.pt")
                torch.save(agent.state_dict(), model_path)
                model_paths[sync_interval].append(model_path)
                print(f"Saved model to {model_path}")
        
        rewards_dict[sync_interval] = interval_rewards
        computation_times[sync_interval] = time.time() - interval_start_time
    
    # Save metadata about the training
    metadata = {
        "sync_intervals": sync_intervals,
        "n_trials": n_trials,
        "n_agents": n_agents,
        "n_episodes": n_episodes,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "averaging_scheme": averaging_scheme,
        "model_paths": {str(k): v for k, v in model_paths.items()},
        "timestamp": timestamp
    }
    
    with open(os.path.join(MODELS_DIR, f"training_metadata_{timestamp}.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Plot and save results
    plot_sync_interval_comparison(rewards_dict, timestamp=timestamp)
    
    # Analyze and print metrics
    metrics_df = analyze_sync_interval_metrics(rewards_dict)
    print("\nSync Interval Analysis Results:")
    print(metrics_df.to_string(index=False))
    
    # Print computation times
    print("\nComputation Time Analysis:")
    for sync_interval, comp_time in computation_times.items():
        print(f"  Sync Interval {sync_interval}: {comp_time/60:.2f} minutes")
        
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f'training_metrics_{timestamp}.csv'), index=False)
    
    return model_paths, rewards_dict, timestamp

def plot_sync_interval_comparison(
    rewards_dict: Dict[int, List[np.ndarray]],
    window_size: int = 10,
    timestamp: str = None
) -> None:
    """
    Plot the comparison of different sync intervals with confidence intervals.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    plt.figure(figsize=(14, 8))
    
    # Define colors for different sync intervals
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, (sync_interval, rewards_list) in enumerate(rewards_dict.items()):
        # Convert to numpy array
        data = np.array(rewards_list)
        
        # Calculate means and standard deviations
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # Smooth the curves
        def smooth(data, window_size):
            return pd.Series(data).rolling(window=window_size, min_periods=1).mean()
        
        mean_smooth = smooth(mean, window_size)
        std_smooth = smooth(std, window_size)
        
        # Plot means and confidence intervals
        color = colors[i % len(colors)]
        episodes = range(len(mean))
        plt.plot(episodes, mean_smooth, label=f'Sync Interval = {sync_interval}', color=color)
        plt.fill_between(episodes, 
                        mean_smooth - std_smooth,
                        mean_smooth + std_smooth,
                        alpha=0.2, color=color)
    
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Comparison of Different Synchronization Intervals in Federated Learning', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(RESULTS_DIR, f'sync_interval_comparison_{timestamp}.png'))
    plt.close()

def analyze_sync_interval_metrics(
    rewards_dict: Dict[int, List[np.ndarray]],
    threshold: float = 0.9
) -> pd.DataFrame:
    """
    Calculate various metrics for different sync intervals.
    """
    results = []
    
    for sync_interval, rewards_list in rewards_dict.items():
        data = np.array(rewards_list)
        
        # Calculate final performance (average of last 10% of episodes)
        final_window = int(data.shape[1] * 0.1)
        final_performance = np.mean(data[:, -final_window:])
        
        # Calculate episodes to convergence (90% of final performance)
        target = final_performance * threshold
        
        convergence_episodes = [np.where(trial >= target)[0][0] 
                              if any(trial >= target) else -1 
                              for trial in data]
        
        valid_convergence = [ep for ep in convergence_episodes if ep != -1]
        avg_convergence = np.mean(valid_convergence) if valid_convergence else float('nan')
        std_convergence = np.std(valid_convergence) if len(valid_convergence) > 1 else 0
        
        results.append({
            'sync_interval': sync_interval,
            'final_performance': final_performance,
            'episodes_to_convergence': avg_convergence,
            'convergence_std': std_convergence
        })
    
    return pd.DataFrame(results)

def test_models_dynamic_environment(
    model_paths: Dict[int, List[str]],
    environment_scenarios: List[Dict],
    n_test_episodes: int = 100,
    hidden_dim: int = 16,
    timestamp: str = None
) -> pd.DataFrame:
    """
    Test saved models in different dynamic environments.
    
    Args:
        model_paths: Dictionary mapping sync intervals to lists of model paths
        environment_scenarios: List of environment configuration dictionaries
        n_test_episodes: Number of episodes to test each model
        hidden_dim: Hidden dimension size of the network
        timestamp: Timestamp for saving results
        
    Returns:
        DataFrame with test results
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = []
    
    # For each sync interval
    for sync_interval, paths in model_paths.items():
        print(f"\n{'='*50}")
        print(f"Testing models with sync_interval = {sync_interval}")
        print(f"{'='*50}")
        
        # For each saved model with this sync interval
        for model_idx, model_path in enumerate(paths):
            # Extract agent index and trial number from path
            path_parts = os.path.basename(model_path).split('_')
            agent_idx = int(path_parts[1])
            trial_idx = int(path_parts[5])
            
            # For each environment scenario
            for scenario_idx, scenario in enumerate(environment_scenarios):
                print(f"\nTesting sync={sync_interval}, agent={agent_idx}, trial={trial_idx} on scenario {scenario_idx + 1}")
                
                # Create environment with this scenario
                cloud_controller = CloudController(max_comp_units=scenario.get("max_comp_units", 100))
                env = CustomEnv(
                    max_comp_units=10,
                    max_available_computation_units=scenario.get("max_available_computation_units", 50),
                    max_epoch_size=10,
                    max_power=10,
                    agent_velocity=scenario.get("agent_velocity", 20),
                    channel_pattern=scenario.get("channel_pattern", "suburban"),
                    channel_pattern_change_interval=scenario.get("channel_pattern_change_interval", 20),
                    cloud_controller=cloud_controller
                )
                
                # Load model
                state_dim = len(flatten_dict_values(env.observation_space.sample()))
                action_dim = env.action_space.n
                model = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()  # Set to evaluation mode
                
                # Test the model
                episode_rewards = []
                for episode in range(n_test_episodes):
                    state = env.reset()[0]
                    state = flatten_dict_values(state)
                    done = False
                    total_reward = 0
                    
                    while not done:
                        # Convert state to tensor
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        
                        # Get action from model
                        with torch.no_grad():
                            q_values = model(state_tensor)
                            action = q_values.max(1)[1].item()
                        
                        # Take action in environment
                        next_state, reward, done, _, _ = env.step(action)
                        next_state = flatten_dict_values(next_state)
                        
                        # Update state and accumulate reward
                        state = next_state
                        total_reward += reward
                    
                    episode_rewards.append(total_reward)
                    
                    if (episode + 1) % 10 == 0:
                        print(f"  Episode {episode + 1}/{n_test_episodes}, Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")
                
                # Record results
                results.append({
                    'sync_interval': sync_interval,
                    'agent_idx': agent_idx,
                    'trial_idx': trial_idx,
                    'scenario_idx': scenario_idx,
                    'avg_reward': np.mean(episode_rewards),
                    'std_reward': np.std(episode_rewards),
                    'max_reward': np.max(episode_rewards),
                    'min_reward': np.min(episode_rewards),
                    'scenario': json.dumps(scenario)
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, f'dynamic_environment_results_{timestamp}.csv'), index=False)
    
    # Create summary by sync interval and scenario
    summary_df = results_df.groupby(['sync_interval', 'scenario_idx']).agg({
        'avg_reward': ['mean', 'std'],
        'max_reward': 'max',
        'min_reward': 'min'
    }).reset_index()
    
    # Plot results
    plot_dynamic_environment_results(results_df, timestamp)
    
    return results_df, summary_df

def plot_dynamic_environment_results(results_df: pd.DataFrame, timestamp: str = None):
    """
    Create visualizations of the dynamic environment test results.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Bar plot for average rewards by sync interval across scenarios
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    pivot_df = results_df.pivot_table(
        index='scenario_idx', 
        columns='sync_interval', 
        values='avg_reward', 
        aggfunc='mean'
    )
    
    # Plot
    pivot_df.plot(kind='bar', yerr=results_df.pivot_table(
        index='scenario_idx', 
        columns='sync_interval', 
        values='avg_reward', 
        aggfunc='std'
    ), capsize=5, figsize=(14, 8))
    
    plt.title('Average Rewards Across Different Scenarios by Sync Interval', fontsize=14)
    plt.xlabel('Scenario Index', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.legend(title='Sync Interval', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(RESULTS_DIR, f'scenario_comparison_{timestamp}.png'))
    plt.close()
    
    # 2. Heatmap of sync interval performance across scenarios
    plt.figure(figsize=(12, 8))
    
    # Normalize rewards for better visualization (optional)
    # This makes it easier to see which sync interval performs best in each scenario
    norm_pivot = pivot_df.copy()
    for idx in norm_pivot.index:
        row_max = pivot_df.loc[idx].max()
        row_min = pivot_df.loc[idx].min()
        if row_max != row_min:  # Avoid division by zero
            norm_pivot.loc[idx] = (pivot_df.loc[idx] - row_min) / (row_max - row_min)
    
    # Plot heatmap
    plt.imshow(norm_pivot, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Average Reward')
    
    # Add text annotations
    for i in range(norm_pivot.shape[0]):
        for j in range(norm_pivot.shape[1]):
            plt.text(j, i, f'{pivot_df.iloc[i, j]:.1f}', 
                    ha='center', va='center', color='white')
    
    plt.title('Normalized Performance Heatmap: Sync Intervals vs Scenarios', fontsize=14)
    plt.xlabel('Sync Interval', fontsize=12)
    plt.ylabel('Scenario Index', fontsize=12)
    plt.xticks(range(len(norm_pivot.columns)), norm_pivot.columns)
    plt.yticks(range(len(norm_pivot.index)), norm_pivot.index)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(RESULTS_DIR, f'performance_heatmap_{timestamp}.png'))
    plt.close()
    
    # 3. Boxplot of rewards distribution by sync interval
    plt.figure(figsize=(14, 8))
    
    # Create boxplot
    ax = sns.boxplot(x='sync_interval', y='avg_reward', data=results_df)
    
    # Add scatter points for individual data points
    sns.swarmplot(x='sync_interval', y='avg_reward', data=results_df, 
                 color='black', alpha=0.5, size=5)
    
    plt.title('Distribution of Rewards by Sync Interval Across All Scenarios', fontsize=14)
    plt.xlabel('Sync Interval', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(RESULTS_DIR, f'reward_distribution_{timestamp}.png'))
    plt.close()

def create_dynamic_scenarios() -> List[Dict]:
    """
    Create a list of scenarios with different environmental conditions.
    """
    scenarios = [
        # Baseline scenario - moderate conditions
        {
            "name": "Baseline",
            "max_comp_units": 100,
            "max_available_computation_units": 50,
            "agent_velocity": 20,
            "channel_pattern": "suburban",
            "channel_pattern_change_interval": 50
        },
        
        # Urban scenario - high computational resources, high velocity, frequent channel changes
        {
            "name": "Urban Rush Hour",
            "max_comp_units": 150,
            "max_available_computation_units": 80,
            "agent_velocity": 30,
            "channel_pattern": "urban",
            "channel_pattern_change_interval": 10
        },
        
        # Rural scenario - low computational resources, low velocity, stable channel
        {
            "name": "Rural Area",
            "max_comp_units": 50,
            "max_available_computation_units": 30,
            "agent_velocity": 10,
            "channel_pattern": "rural",
            "channel_pattern_change_interval": 100
        },
        
        # High volatility scenario - resources and channels changing rapidly
        {
            "name": "High Volatility",
            "max_comp_units": 120,
            "max_available_computation_units": 60,
            "agent_velocity": 25,
            "channel_pattern": "urban",
            "channel_pattern_change_interval": 5
        },
        
        # Resource constrained scenario - low computational resources
        {
            "name": "Resource Constrained",
            "max_comp_units": 30,
            "max_available_computation_units": 20,
            "agent_velocity": 15,
            "channel_pattern": "suburban",
            "channel_pattern_change_interval": 70
        }
    ]
    
    return scenarios

def main():
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Import seaborn for advanced plotting (if not available, install with pip)
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Seaborn not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "seaborn"])
        import seaborn as sns
        sns.set_style("whitegrid")
    
    # Parameters
    sync_intervals = [1, 5, 10, 20, 50]
    n_trials = 2  # Reduced for demonstration; increase for more reliable results
    n_agents = 5
    n_episodes = 20  # Reduced for faster training
    
    # Start timing
    start_time = time.time()
    
    # Train and save models with different sync intervals
    print("Training and saving models with different sync intervals...")
    model_paths, rewards_dict, timestamp = train_and_save_models(
        sync_intervals=sync_intervals,
        n_trials=n_trials,
        n_agents=n_agents,
        n_episodes=n_episodes
    )
    
    # Create dynamic scenarios for testing
    scenarios = create_dynamic_scenarios()
    print(f"\nCreated {len(scenarios)} dynamic testing scenarios:")
    for i, scenario in enumerate(scenarios):
        print(f"  Scenario {i+1}: {scenario['name']}")
    
    # Test models in dynamic environments
    print("\nTesting models in dynamic environments...")
    results_df, summary_df = test_models_dynamic_environment(
        model_paths=model_paths,
        environment_scenarios=scenarios,
        n_test_episodes=50,  # Reduced for faster testing
        timestamp=timestamp
    )
    
    # Print summary results
    print("\nSummary of results by sync interval and scenario:")
    print(summary_df.to_string())
    
    # Find best sync interval for each scenario
    best_by_scenario = results_df.loc[results_df.groupby('scenario_idx')['avg_reward'].idxmax()]
    print("\nBest sync interval for each scenario:")
    for _, row in best_by_scenario.iterrows():
        scenario_idx = int(row['scenario_idx'])
        scenario_name = scenarios[scenario_idx]['name']
        print(f"  Scenario {scenario_idx} ({scenario_name}): Sync Interval {int(row['sync_interval'])} - Avg Reward: {row['avg_reward']:.2f}")
    
    # Find overall best sync interval
    avg_by_sync = results_df.groupby('sync_interval')['avg_reward'].mean()
    best_sync = avg_by_sync.idxmax()
    print(f"\nOverall best sync interval: {best_sync} with average reward: {avg_by_sync[best_sync]:.2f}")
    
    # Print completion time
    print(f"\nTotal experiment time: {(time.time() - start_time)/60:.2f} minutes")
    
    # Print path to results
    print(f"\nResults saved to:")
    print(f"  Models: {os.path.abspath(MODELS_DIR)}")
    print(f"  Test results: {os.path.abspath(RESULTS_DIR)}")

if __name__ == "__main__":
    main()