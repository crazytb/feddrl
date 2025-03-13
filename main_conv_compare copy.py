import torch
import numpy as np
import matplotlib.pyplot as plt
from drl_framework.networks import LocalNetwork
from drl_framework.trainer import train_single_agent, train_federated_agents
from drl_framework.custom_env import CustomEnv, CloudController
from drl_framework.utils import seed_torch, flatten_dict_values
from drl_framework.params import device
import time
import os
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
import seaborn as sns
import json

# Create directories for saving models and results
MODELS_DIR = "saved_models"
RESULTS_DIR = "test_results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_save_models(
    n_agents: int = 5,
    n_episodes: int = 300,
    sync_interval: int = 20,
    learning_rate: float = 0.001,
    hidden_dim: int = 16
) -> Tuple[Dict[str, str], str]:
    """
    Train both individual and federated models and save them to disk.
    
    Args:
        n_agents: Number of agents
        n_episodes: Number of training episodes
        sync_interval: Synchronization interval for federated learning
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension of the network
    
    Returns:
        Dictionary mapping model types to paths, and timestamp
    """
    model_paths = {}
    
    # Environment setup parameters for training
    max_available_computation_units = [10]*7 + [50]*7 + [100]*8
    agent_velocities = [10]*7 + [20]*7 + [30]*8
    channel_patterns = ['urban']*7 + ['suburban']*7 + ['rural']*8
    channel_pattern_change_interval = 100
    
    # Timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Print environment settings
    print("\nTraining Environment Settings:")
    print(f"  - Number of agents: {n_agents}")
    print(f"  - Max computation units: {[10]*n_agents}")
    print(f"  - Available computation units: {max_available_computation_units[:n_agents]}")
    print(f"  - Agent velocities: {agent_velocities[:n_agents]}")
    print(f"  - Channel patterns: {channel_patterns[:n_agents]}")
    print(f"  - Channel pattern change interval: {channel_pattern_change_interval}")
    
    # ====== Train Individual Learning Models ======
    print("\n" + "="*50)
    print("Training Individual Learning Models")
    print("="*50)
    
    individual_start_time = time.time()
    
    # Create agents and optimizers
    individual_agents = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) 
                        for _ in range(n_agents)]
    individual_optimizers = [torch.optim.Adam(agent.parameters(), lr=learning_rate) 
                        for agent in individual_agents]
    
    # Train individual agents (using sync_interval = n_episodes means no synchronization)
    individual_rewards = train_federated_agents(
        envs=envs,
        agents=individual_agents,
        optimizers=individual_optimizers,
        device=device,
        episodes=n_episodes,
        sync_interval=n_episodes,  # No synchronization for individual learning
        hidden_dim=hidden_dim,
        averaging_scheme='fedavg',
        cloud_controller=cloud_controller,
    )
    
    individual_time = time.time() - individual_start_time
    
    # Save individual models
    for i, agent in enumerate(individual_agents):
        model_path = os.path.join(MODELS_DIR, f"individual_agent_{i}_{timestamp}.pt")
        torch.save(agent.state_dict(), model_path)
        model_paths[f"individual_agent_{i}"] = model_path
        print(f"Saved individual model to {model_path}")
    
    # ====== Train Federated Learning Models ======
    print("\n" + "="*50)
    print(f"Training Federated Learning Models (sync_interval={sync_interval})")
    print("="*50)
    
    federated_start_time = time.time()
    
    # Create agents and optimizers
    federated_agents = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) 
                    for _ in range(n_agents)]
    federated_optimizers = [torch.optim.Adam(agent.parameters(), lr=learning_rate) 
                    for agent in federated_agents]
    
    # Train federated agents
    federated_rewards = train_federated_agents(
        envs=envs,
        agents=federated_agents,
        optimizers=federated_optimizers,
        device=device,
        episodes=n_episodes,
        sync_interval=sync_interval,
        hidden_dim=hidden_dim,
        averaging_scheme='fedavg',
        cloud_controller=cloud_controller,
    )
    
    federated_time = time.time() - federated_start_time
    
    # Save federated models
    for i, agent in enumerate(federated_agents):
        model_path = os.path.join(MODELS_DIR, f"federated_agent_{i}_{timestamp}.pt")
        torch.save(agent.state_dict(), model_path)
        model_paths[f"federated_agent_{i}"] = model_path
        print(f"Saved federated model to {model_path}")
    
    # Save training metadata
    training_metadata = {
        "n_agents": n_agents,
        "n_episodes": n_episodes,
        "sync_interval": sync_interval,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "model_paths": model_paths,
        "individual_training_time": individual_time,
        "federated_training_time": federated_time,
        "timestamp": timestamp,
        "training_environment": {
            "max_comp_units": 10,
            "max_available_computation_units": max_available_computation_units[:n_agents],
            "agent_velocities": agent_velocities[:n_agents],
            "channel_patterns": channel_patterns[:n_agents],
            "channel_pattern_change_interval": channel_pattern_change_interval
        }
    }
    
    with open(os.path.join(MODELS_DIR, f"training_metadata_{timestamp}.json"), 'w') as f:
        json.dump(training_metadata, f, indent=4)
    
    # Plot training performance comparison
    plot_training_performance(individual_rewards, federated_rewards, timestamp)
    
    # Print training times
    print("\nTraining Time Analysis:")
    print(f"  Individual Learning Time: {individual_time/60:.2f} minutes")
    print(f"  Federated Learning Time: {federated_time/60:.2f} minutes")
    
    return model_paths, timestamp

def plot_training_performance(
    individual_rewards: List[np.ndarray],
    federated_rewards: List[np.ndarray],
    timestamp: str,
    window_size: int = 10
):
    """
    Plot the training performance comparison between individual and federated learning.
    """
    plt.figure(figsize=(14, 8))
    
    # Calculate means and standard deviations
    individual_mean = np.mean(individual_rewards, axis=0)
    federated_mean = np.mean(federated_rewards, axis=0)
    individual_std = np.std(individual_rewards, axis=0)
    federated_std = np.std(federated_rewards, axis=0)
    
    # Smooth the curves
    def smooth(data, window_size):
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    
    individual_mean_smooth = smooth(individual_mean, window_size)
    federated_mean_smooth = smooth(federated_mean, window_size)
    individual_std_smooth = smooth(individual_std, window_size)
    federated_std_smooth = smooth(federated_std, window_size)
    
    # Plot means and confidence intervals
    episodes = range(len(individual_mean))
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
    
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Training Performance: Individual vs Federated Learning', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(RESULTS_DIR, f'training_performance_{timestamp}.png'))
    plt.close()

def create_test_scenarios(n_agents: int = 5) -> List[Dict]:
    """
    Create a list of test scenarios with different environmental conditions
    that differ from the training environment.
    
    Args:
        n_agents: Number of agents to configure in each scenario
        
    Returns:
        List of scenario dictionaries
    """
    
    # Helper function to repeat and extend patterns based on agent count
    def create_pattern(patterns, n_agents):
        """Generate a pattern that distributes values evenly across n_agents"""
        # Calculate how many agents should get each pattern
        counts = []
        base_count = n_agents // len(patterns)
        remainder = n_agents % len(patterns)
        
        for i in range(len(patterns)):
            counts.append(base_count + (1 if i < remainder else 0))
        
        # Create the extended pattern
        result = []
        for i, pattern in enumerate(patterns):
            result.extend([pattern] * counts[i])
        
        return result
    
    # Create base patterns for standard distributions
    standard_computation_units = create_pattern([10, 50, 100], n_agents)
    standard_velocities = create_pattern([10, 20, 30], n_agents)
    standard_channel_patterns = create_pattern(['urban', 'suburban', 'rural'], n_agents)
    
    # Create modified patterns for each scenario
    scenarios = [
        # Baseline scenario - similar to training but slightly different
        {
            "name": "Baseline (Similar)",
            "max_comp_units": 100,
            "max_available_computation_units": create_pattern([20, 60, 110], n_agents),
            "agent_velocities": create_pattern([15, 25, 35], n_agents),
            "channel_patterns": standard_channel_patterns,
            "channel_pattern_change_interval": 80
        },
        
        # High volatility scenario - much more frequent channel changes
        {
            "name": "High Channel Volatility",
            "max_comp_units": 100,
            "max_available_computation_units": standard_computation_units,
            "agent_velocities": standard_velocities,
            "channel_patterns": standard_channel_patterns,
            "channel_pattern_change_interval": 10  # Much more frequent changes
        },
        
        # Resource constrained scenario - lower computational resources
        {
            "name": "Resource Constrained",
            "max_comp_units": 50,  # Lower max computation units
            "max_available_computation_units": [max(5, val // 2) for val in standard_computation_units],  # Half the resources
            "agent_velocities": standard_velocities,
            "channel_patterns": standard_channel_patterns,
            "channel_pattern_change_interval": 100
        },
        
        # High mobility scenario - agents moving much faster
        {
            "name": "High Mobility",
            "max_comp_units": 100,
            "max_available_computation_units": standard_computation_units,
            "agent_velocities": [vel * 3 for vel in standard_velocities],  # 3x the velocities
            "channel_patterns": standard_channel_patterns,
            "channel_pattern_change_interval": 100
        },
        
        # Changed environment distribution - more urban, less rural
        {
            "name": "Urban Dominant",
            "max_comp_units": 100,
            "max_available_computation_units": standard_computation_units,
            "agent_velocities": standard_velocities,
            "channel_patterns": create_pattern(['urban', 'urban', 'suburban'], n_agents),  # More urban
            "channel_pattern_change_interval": 100
        },
        
        # Extreme heterogeneity scenario - very diverse agent capabilities
        {
            "name": "Extreme Heterogeneity",
            "max_comp_units": 120,
            "max_available_computation_units": [5 * (i + 1) for i in range(n_agents)],  # Linear increase
            "agent_velocities": [10 * (i + 1) for i in range(n_agents)],  # Linear increase
            "channel_patterns": create_pattern(['urban'] * (n_agents // 3 + 1) + 
                                              ['suburban'] * (n_agents // 3 + 1) + 
                                              ['rural'] * (n_agents // 3 + 1), n_agents),
            "channel_pattern_change_interval": 50
        },
        
        # High computation demand scenario - requires heavy computation
        {
            "name": "High Computation Demand",
            "max_comp_units": 200,
            "max_available_computation_units": [val * 2 for val in standard_computation_units],  # Double resources 
            "agent_velocities": standard_velocities,
            "channel_patterns": standard_channel_patterns,
            "channel_pattern_change_interval": 100
        }
    ]
    
    # Print scenario details
    print(f"\nGenerated test scenarios for {n_agents} agents:")
    for scenario in scenarios:
        print(f"  - {scenario['name']}:")
        print(f"    Max comp units: {scenario['max_comp_units']}")
        print(f"    Comp units distribution: {scenario['max_available_computation_units'][:5]}{'...' if n_agents > 5 else ''}")
        print(f"    Agent velocities: {scenario['agent_velocities'][:5]}{'...' if n_agents > 5 else ''}")
        print(f"    Channel patterns: {scenario['channel_patterns'][:5]}{'...' if n_agents > 5 else ''}")
        print(f"    Channel change interval: {scenario['channel_pattern_change_interval']}")
    
    return scenarios

def test_models_in_changed_environments(
    model_paths: Dict[str, str],
    n_agents: int = 5,
    n_test_episodes: int = 100,
    hidden_dim: int = 16,
    timestamp: str = None
) -> pd.DataFrame:
    """
    Test saved models in environments that differ from the training environment.
    
    Args:
        model_paths: Dictionary mapping model names to file paths
        n_agents: Number of agents
        n_test_episodes: Number of episodes to test each model
        hidden_dim: Hidden dimension of the network
        timestamp: Timestamp for saving results
        
    Returns:
        DataFrame with test results
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create test scenarios
    scenarios = create_test_scenarios(n_agents)
    
    print("\nTest Scenarios:")
    for i, scenario in enumerate(scenarios):
        print(f"  {i+1}. {scenario['name']}")
    
    results = []
    
    # Group models by learning type
    individual_models = {k: v for k, v in model_paths.items() if 'individual' in k}
    federated_models = {k: v for k, v in model_paths.items() if 'federated' in k}
    
    # For each scenario
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n{'='*50}")
        print(f"Testing in Scenario {scenario_idx + 1}: {scenario['name']}")
        print(f"{'='*50}")
        
        # Print scenario settings
        print("\nScenario Settings:")
        print(f"  - Max computation units: {scenario['max_comp_units']}")
        print(f"  - Available computation units: {scenario['max_available_computation_units'][:n_agents]}")
        print(f"  - Agent velocities: {scenario['agent_velocities'][:n_agents]}")
        print(f"  - Channel patterns: {scenario['channel_patterns'][:n_agents]}")
        print(f"  - Channel pattern change interval: {scenario['channel_pattern_change_interval']}")
        
        # Create cloud controller
        cloud_controller = CloudController(max_comp_units=scenario['max_comp_units'])
        
        # Create environments for each agent
        envs = []
        for i in range(n_agents):
            env = CustomEnv(
                max_comp_units=10,
                max_available_computation_units=scenario['max_available_computation_units'][i],
                max_epoch_size=10,
                max_power=10,
                agent_velocity=scenario['agent_velocities'][i],
                channel_pattern=scenario['channel_patterns'][i],
                channel_pattern_change_interval=scenario['channel_pattern_change_interval'],
                cloud_controller=cloud_controller
            )
            envs.append(env)
        
        # Get dimensions from first environment
        state_dim = len(flatten_dict_values(envs[0].observation_space.sample()))
        action_dim = envs[0].action_space.n
        
        # Test individual models
        print("\nTesting Individual Learning Models...")
        individual_episode_rewards = []
        
        for agent_idx, (model_name, model_path) in enumerate(individual_models.items()):
            print(f"  Testing {model_name} in scenario {scenario_idx + 1}...")
            
            # Load model
            model = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode
            
            # Test the model
            agent_rewards = []
            for episode in range(n_test_episodes):
                state = envs[agent_idx].reset()[0]
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
                    next_state, reward, done, _, _ = envs[agent_idx].step(action)
                    next_state = flatten_dict_values(next_state)
                    
                    # Update state and accumulate reward
                    state = next_state
                    total_reward += reward
                
                agent_rewards.append(total_reward)
                
            individual_episode_rewards.append(agent_rewards)
            
            # Record results
            results.append({
                'learning_type': 'Individual',
                'agent_idx': agent_idx,
                'scenario_idx': scenario_idx,
                'scenario_name': scenario['name'],
                'avg_reward': np.mean(agent_rewards),
                'std_reward': np.std(agent_rewards),
                'max_reward': np.max(agent_rewards),
                'min_reward': np.min(agent_rewards)
            })
            
            print(f"    Avg Reward: {np.mean(agent_rewards):.2f}")
        
        # Calculate overall individual performance
        individual_avg = np.mean([np.mean(rewards) for rewards in individual_episode_rewards])
        print(f"  Overall Individual Learning Average Reward: {individual_avg:.2f}")
        
        # Test federated models
        print("\nTesting Federated Learning Models...")
        federated_episode_rewards = []
        
        for agent_idx, (model_name, model_path) in enumerate(federated_models.items()):
            print(f"  Testing {model_name} in scenario {scenario_idx + 1}...")
            
            # Load model
            model = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode
            
            # Test the model
            agent_rewards = []
            for episode in range(n_test_episodes):
                state = envs[agent_idx].reset()[0]
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
                    next_state, reward, done, _, _ = envs[agent_idx].step(action)
                    next_state = flatten_dict_values(next_state)
                    
                    # Update state and accumulate reward
                    state = next_state
                    total_reward += reward
                
                agent_rewards.append(total_reward)
                
            federated_episode_rewards.append(agent_rewards)
            
            # Record results
            results.append({
                'learning_type': 'Federated',
                'agent_idx': agent_idx,
                'scenario_idx': scenario_idx,
                'scenario_name': scenario['name'],
                'avg_reward': np.mean(agent_rewards),
                'std_reward': np.std(agent_rewards),
                'max_reward': np.max(agent_rewards),
                'min_reward': np.min(agent_rewards)
            })
            
            print(f"    Avg Reward: {np.mean(agent_rewards):.2f}")
        
        # Calculate overall federated performance
        federated_avg = np.mean([np.mean(rewards) for rewards in federated_episode_rewards])
        print(f"  Overall Federated Learning Average Reward: {federated_avg:.2f}")
        
        # Compare performance
        performance_diff = federated_avg - individual_avg
        performance_ratio = federated_avg / individual_avg if individual_avg != 0 else float('inf')
        
        print(f"\nPerformance Comparison in Scenario {scenario_idx + 1}:")
        print(f"  Federated vs Individual Difference: {performance_diff:.2f}")
        print(f"  Federated/Individual Ratio: {performance_ratio:.2f}x")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, f'changed_environment_results_{timestamp}.csv'), index=False)
    
    # Visualize results
    plot_environment_comparison(results_df, timestamp)
    
    return results_df

def plot_environment_comparison(results_df: pd.DataFrame, timestamp: str):
    """
    Create visualizations comparing individual and federated learning performance.
    """
    # Set up seaborn style
    sns.set(style="whitegrid")
    
    # 1. Bar plot of average rewards by learning type across scenarios
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    scenario_order = results_df['scenario_name'].unique()
    
    # Group by scenario and learning type
    summary_data = results_df.groupby(['scenario_name', 'learning_type'])['avg_reward'].mean().reset_index()
    
    # Plot
    ax = sns.barplot(
        x='scenario_name', 
        y='avg_reward', 
        hue='learning_type',
        data=summary_data,
        order=scenario_order,
        palette={'Individual': 'blue', 'Federated': 'red'}
    )
    
    plt.title('Average Rewards Across Different Scenarios by Learning Type', fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Learning Type', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(RESULTS_DIR, f'learning_type_comparison_{timestamp}.png'))
    plt.close()
    
    # 2. Calculate performance ratio
    ratio_data = summary_data.pivot(index='scenario_name', columns='learning_type', values='avg_reward')
    ratio_data['Ratio'] = ratio_data['Federated'] / ratio_data['Individual']
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        x=ratio_data.index,
        y=ratio_data['Ratio'],
        palette=['green' if x > 1 else 'orange' for x in ratio_data['Ratio']]
    )
    
    plt.axhline(y=1.0, color='red', linestyle='--')
    plt.title('Federated vs Individual Learning Performance Ratio', fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Federated/Individual Ratio', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for i, ratio in enumerate(ratio_data['Ratio']):
        ax.text(i, ratio + 0.05, f'{ratio:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'performance_ratio_{timestamp}.png'))
    plt.close()
    
    # 3. Box plots showing reward distribution by learning type for each scenario
    plt.figure(figsize=(16, 10))
    
    ax = sns.boxplot(
        x='scenario_name', 
        y='avg_reward', 
        hue='learning_type',
        data=results_df,
        order=scenario_order,
        palette={'Individual': 'blue', 'Federated': 'red'}
    )
    
    plt.title('Reward Distribution by Scenario and Learning Type', fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Learning Type', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, f'reward_distribution_{timestamp}.png'))
    plt.close()

def main():
    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Import seaborn if not already imported
    try:
        import seaborn
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "seaborn"])
        import seaborn as sns
        sns.set(style="whitegrid")
    
    # Parameters
    n_agents = 20
    n_episodes = 300
    sync_interval = 20  # As requested
    hidden_dim = 16
    n_test_episodes = 50
    
    # Start timing
    start_time = time.time()
    
    print("="*70)
    print(f"INDIVIDUAL VS FEDERATED LEARNING COMPARISON (SYNC_INTERVAL={sync_interval})")
    print("="*70)
    
    # Train and save both individual and federated models
    print("\nStep 1: Training and Saving Models")
    model_paths, timestamp = train_and_save_models(
        n_agents=n_agents,
        n_episodes=n_episodes,
        sync_interval=sync_interval,
        hidden_dim=hidden_dim
    )
    
    # Test models in changed environments
    print("\nStep 2: Testing Models in Changed Environments")
    results_df = test_models_in_changed_environments(
        model_paths=model_paths,
        n_agents=n_agents,
        n_test_episodes=n_test_episodes,
        hidden_dim=hidden_dim,
        timestamp=timestamp
    )
    
    # Summary analysis
    print("\nResults Summary:")
    summary = results_df.groupby(['scenario_name', 'learning_type'])['avg_reward'].mean().unstack()
    summary['Difference'] = summary['Federated'] - summary['Individual']
    summary['Ratio'] = summary['Federated'] / summary['Individual']
    print(summary)
    
    # Save summary to CSV
    summary.to_csv(os.path.join(RESULTS_DIR, f'results_summary_{timestamp}.csv'))
    
    # Overall conclusion
    federated_better_count = sum(summary['Federated'] > summary['Individual'])
    individual_better_count = sum(summary['Federated'] < summary['Individual'])
    
    print("\nOverall Conclusion:")
    print(f"  - Federated learning performed better in {federated_better_count} of {len(summary)} scenarios")
    print(f"  - Individual learning performed better in {individual_better_count} of {len(summary)} scenarios")
    
    avg_ratio = summary['Ratio'].mean()
    print(f"  - On average, federated learning performed {avg_ratio:.2f}x better than individual learning")
    
    # Print completion time
    print(f"\nTotal experiment time: {(time.time() - start_time)/60:.2f} minutes")
    
    # Print path to results
    print(f"\nResults saved to:")
    print(f"  Models: {os.path.abspath(MODELS_DIR)}")
    print(f"  Test results: {os.path.abspath(RESULTS_DIR)}")

if __name__ == "__main__":
    main()