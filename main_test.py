import gymnasium as gym
import torch
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from drl_framework.networks import LocalNetwork
from drl_framework.custom_env import CustomEnv
from drl_framework.utils import *
from drl_framework.params import device

def load_model(model_path, device):
    """Load a trained model from file"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model architecture
    arch = checkpoint['model_architecture']
    model = LocalNetwork(
        state_dim=arch['state_dim'],
        action_dim=arch['action_dim'],
        hidden_dim=arch['hidden_dim']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model

def load_training_environments(config_path):
    """Load the exact environments used during training"""
    # Try JSON file first
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        envs = []
        for env_data in config['envs']:
            env = CustomEnv(
                max_comp_units=env_data['max_comp_units'],
                max_epoch_size=env_data['max_epoch_size'],
                max_queue_size=env_data['max_queue_size'],
                reward_weights=env_data['reward_weights'],
                agent_velocities=env_data['agent_velocities']
            )
            envs.append(env)
        
        print(f"Loaded {len(envs)} training environments from {config_path}")
        return envs, config
        
    except FileNotFoundError:
        print(f"JSON config file not found: {config_path}")
        
        # Try text file as fallback
        txt_config_path = config_path.replace('.json', '.txt')
        print(f"Trying text file fallback: {txt_config_path}")
        
        try:
            return load_training_environments_from_txt(txt_config_path)
        except Exception as txt_error:
            print(f"Text file fallback also failed: {txt_error}")
            return None, None
            
    except Exception as e:
        print(f"Error loading training environments: {e}")
        return None, None

def load_training_environments_from_txt(txt_path):
    """Load training environments from text config file (fallback)"""
    if not os.path.exists(txt_path):
        print(f"Text config file not found: {txt_path}")
        return None, None
    
    envs = []
    current_agent = {}
    
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('Agent ') and 'Environment:' in line:
                if current_agent:  # Save previous agent
                    env = CustomEnv(
                        max_comp_units=current_agent['max_comp_units'],
                        max_epoch_size=current_agent.get('max_epoch_size', 10),
                        max_queue_size=current_agent.get('max_queue_size', 5),
                        reward_weights=current_agent.get('reward_weights', 0.1),
                        agent_velocities=current_agent['agent_velocities']
                    )
                    envs.append(env)
                current_agent = {}
            elif '- Max Computation Units:' in line:
                current_agent['max_comp_units'] = int(line.split(':')[1].strip())
            elif '- Agent Velocities:' in line:
                current_agent['agent_velocities'] = int(line.split(':')[1].strip())
            elif '- Max Epoch Size:' in line:
                current_agent['max_epoch_size'] = int(line.split(':')[1].strip())
            elif '- Max Queue Size:' in line:
                current_agent['max_queue_size'] = int(line.split(':')[1].strip())
            elif '- Reward Weights:' in line:
                current_agent['reward_weights'] = float(line.split(':')[1].strip())
        
        # Don't forget the last agent
        if current_agent:
            env = CustomEnv(
                max_comp_units=current_agent['max_comp_units'],
                max_epoch_size=current_agent.get('max_epoch_size', 10),
                max_queue_size=current_agent.get('max_queue_size', 5),
                reward_weights=current_agent.get('reward_weights', 0.1),
                agent_velocities=current_agent['agent_velocities']
            )
            envs.append(env)
        
        print(f"Loaded {len(envs)} training environments from text file: {txt_path}")
        return envs, {'envs': []} # Return minimal config
        
    except Exception as e:
        print(f"Error parsing text config file: {e}")
        return None, None

def create_test_environments(num_agents, seed=None):
    """Create new random test environments (for generalization testing)"""
    if seed:
        np.random.seed(seed)
    
    envs = [
        CustomEnv(
            max_comp_units=np.random.randint(1, 101),  # 1 to 100
            max_epoch_size=10,
            max_queue_size=5,
            reward_weights=0.1,
            agent_velocities=np.random.randint(10, 101)  # 10 to 100
        ) for _ in range(num_agents)
    ]
    
    print(f"Created {len(envs)} new random test environments")
    return envs

def test_agent(env, agent, episodes=100, render=False, agent_name="Agent"):
    """Test a single agent on an environment"""
    total_rewards = []
    episode_lengths = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = flatten_dict_values(state)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Use greedy policy (no exploration) for testing
            with torch.no_grad():
                action = agent(state_tensor).argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render and episode < 5:  # Render first 5 episodes
                print(f"{agent_name} Episode {episode+1}, Step {episode_length}: "
                      f"Action={action}, Reward={reward:.2f}, Done={done}")
            
            state = flatten_dict_values(next_state)
        
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 20 == 0:
            print(f"{agent_name} Test Episode {episode+1}: Reward = {episode_reward:.1f}, Length = {episode_length}")
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'all_rewards': total_rewards,
        'all_lengths': episode_lengths
    }

def test_models(model_dir, test_envs, test_episodes=100, env_type="training"):
    """Test all models in a directory"""
    model_files = sorted(glob.glob(os.path.join(model_dir, "agent_*.pth")))
    results = []
    
    print(f"\nTesting models from: {model_dir}")
    print(f"Using {env_type} environments")
    print(f"Found {len(model_files)} model files")
    
    for i, model_file in enumerate(model_files):
        print(f"\n--- Testing Agent {i+1} ---")
        
        # Load model
        agent = load_model(model_file, device)
        
        # Test on corresponding environment
        if i < len(test_envs):
            env = test_envs[i]
            print(f"Testing on Environment {i+1}: "
                  f"max_comp_units={env.max_comp_units}, "
                  f"agent_velocities={env.agent_velocities}")
            
            # Test the agent
            result = test_agent(env, agent, episodes=test_episodes, agent_name=f"Agent {i+1}")
            result['agent_id'] = i + 1
            result['model_file'] = model_file
            result['env_type'] = env_type
            results.append(result)
            
            print(f"Results - Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}, "
                  f"Mean Length: {result['mean_length']:.2f} ± {result['std_length']:.2f}")
        else:
            print(f"No test environment available for Agent {i+1}")
    
    return results

def compare_results(independent_results, federated_results, env_type="training"):
    """Compare independent vs federated learning results"""
    print("\n" + "="*80)
    print(f"COMPARISON RESULTS ({env_type.upper()} ENVIRONMENTS)")
    print("="*80)
    
    # Individual agent comparison
    print("\nIndividual Agent Performance:")
    print("-" * 60)
    print(f"{'Agent':<8} {'Independent':<15} {'Federated':<15} {'Improvement':<12}")
    print("-" * 60)
    
    improvements = []
    for i in range(min(len(independent_results), len(federated_results))):
        ind_reward = independent_results[i]['mean_reward']
        fed_reward = federated_results[i]['mean_reward']
        improvement = ((fed_reward - ind_reward) / abs(ind_reward)) * 100 if ind_reward != 0 else 0
        improvements.append(improvement)
        
        print(f"Agent {i+1:<2} {ind_reward:<15.2f} {fed_reward:<15.2f} {improvement:<12.1f}%")
    
    # Overall statistics
    print("-" * 60)
    ind_mean = np.mean([r['mean_reward'] for r in independent_results])
    fed_mean = np.mean([r['mean_reward'] for r in federated_results])
    overall_improvement = ((fed_mean - ind_mean) / abs(ind_mean)) * 100 if ind_mean != 0 else 0
    
    print(f"{'Average':<8} {ind_mean:<15.2f} {fed_mean:<15.2f} {overall_improvement:<12.1f}%")
    
    print(f"\nSummary ({env_type} environments):")
    print(f"- Independent Learning Average: {ind_mean:.2f}")
    print(f"- Federated Learning Average: {fed_mean:.2f}")
    print(f"- Overall Improvement: {overall_improvement:.1f}%")
    print(f"- Agents with Positive Improvement: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    
    return {
        'independent_mean': ind_mean,
        'federated_mean': fed_mean,
        'overall_improvement': overall_improvement,
        'individual_improvements': improvements,
        'env_type': env_type
    }

def plot_test_results(results_training, results_generalization, save_path=None):
    """Plot comparison of test results for both training and generalization"""
    # Determine subplot layout based on available results
    if results_training and results_generalization:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        has_both = True
    elif results_training:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        has_both = False
    elif results_generalization:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        has_both = False
    else:
        print("No results to plot!")
        return
    
    # Function to plot single comparison
    def plot_comparison(ax, independent_results, federated_results, title_suffix):
        agents = [f"Agent {i+1}" for i in range(len(independent_results))]
        ind_rewards = [r['mean_reward'] for r in independent_results]
        fed_rewards = [r['mean_reward'] for r in federated_results]
        
        x = np.arange(len(agents))
        width = 0.35
        
        ax.bar(x - width/2, ind_rewards, width, label='Independent', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, fed_rewards, width, label='Federated', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Agents')
        ax.set_ylabel('Mean Reward')
        ax.set_title(f'Mean Reward Comparison - {title_suffix}')
        ax.set_xticks(x)
        ax.set_xticklabels(agents)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_distribution(ax, independent_results, federated_results, title_suffix):
        data_for_box = [ind_res['all_rewards'] for ind_res in independent_results] + \
                       [fed_res['all_rewards'] for fed_res in federated_results]
        labels_for_box = [f'Ind-A{i+1}' for i in range(len(independent_results))] + \
                         [f'Fed-A{i+1}' for i in range(len(federated_results))]
        
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        
        # Color the boxes
        colors = ['skyblue'] * len(independent_results) + ['lightcoral'] * len(federated_results)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_xlabel('Agent Type')
        ax.set_ylabel('Reward Distribution')
        ax.set_title(f'Reward Distribution - {title_suffix}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Plot results based on what's available
    if has_both:
        # Plot training environment results (top row)
        if results_training:
            print("Plotting training environment results...")
            plot_comparison(ax1, results_training['independent'], results_training['federated'], 'Training Environments')
            plot_distribution(ax2, results_training['independent'], results_training['federated'], 'Training Environments')
        
        # Plot generalization results (bottom row)
        if results_generalization:
            print("Plotting generalization results...")
            plot_comparison(ax3, results_generalization['independent'], results_generalization['federated'], 'New Environments')
            plot_distribution(ax4, results_generalization['independent'], results_generalization['federated'], 'New Environments')
    else:
        # Single row layout
        if results_training:
            print("Plotting training environment results only...")
            plot_comparison(ax1, results_training['independent'], results_training['federated'], 'Training Environments')
            plot_distribution(ax2, results_training['independent'], results_training['federated'], 'Training Environments')
        elif results_generalization:
            print("Plotting generalization results only...")
            plot_comparison(ax1, results_generalization['independent'], results_generalization['federated'], 'New Environments')
            plot_distribution(ax2, results_generalization['independent'], results_generalization['federated'], 'New Environments')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def main():
    print(f"Device: {device}")
    
    # Set seed for reproducible testing
    test_seed = 123
    np.random.seed(test_seed)
    seed_torch(test_seed)
    
    # Configuration
    test_episodes = 100
    
    # Find the most recent model directories
    model_base_dir = "models"
    if not os.path.exists(model_base_dir):
        print(f"Models directory '{model_base_dir}' not found!")
        print("Please run main_training.py first to train and save models.")
        return
    
    # Get all model directories
    independent_dirs = glob.glob(os.path.join(model_base_dir, "independent_*"))
    federated_dirs = glob.glob(os.path.join(model_base_dir, "federated_*"))
    
    if not independent_dirs or not federated_dirs:
        print("No trained models found!")
        print("Please run main_training.py first to train and save models.")
        return
    
    # Use the most recent models (sort by modification time)
    independent_dir = max(independent_dirs, key=os.path.getmtime)
    federated_dir = max(federated_dirs, key=os.path.getmtime)
    
    print(f"Using Independent models from: {independent_dir}")
    print(f"Using Federated models from: {federated_dir}")
    
    # Extract timestamp and number of agents
    try:
        # timestamp = independent_dir.split('_')[-1]
        num_agents = int(independent_dir.split('_')[1].replace('agents', ''))
    except:
        num_agents = 3  # Default fallback
        # timestamp = "unknown"
        
    timestamp = TIMESTAMP
    
    print(f"Number of agents: {num_agents}")
    print(f"Timestamp: {timestamp}")
    
    # Load training environments
    config_path = os.path.join(model_base_dir, f"env_configs_{num_agents}agents_{timestamp}.json")
    print(f"Looking for config file: {config_path}")
    
    training_envs, training_config = load_training_environments(config_path)
    
    if training_envs:
        print(f"✅ Successfully loaded {len(training_envs)} training environments")
        print(f"\nOriginal Training Environment Configurations:")
        for i, env in enumerate(training_envs):
            print(f"Agent {i+1} Training Environment:")
            print(f"  - Max Computation Units: {env.max_comp_units}")
            print(f"  - Agent Velocities: {env.agent_velocities}")
    else:
        print(f"❌ Failed to load training environments from {config_path}")
        print(f"File exists: {os.path.exists(config_path)}")
        if os.path.exists(config_path):
            print("Config file exists but failed to parse. Check file contents.")
        else:
            print("Config file does not exist. Training environments will be skipped.")
    
    # Create new test environments for generalization testing
    generalization_envs = create_test_environments(num_agents, seed=test_seed + 100)  # Different seed
    print(f"\nNew Test Environment Configurations:")
    for i, env in enumerate(generalization_envs):
        print(f"Agent {i+1} Test Environment:")
        print(f"  - Max Computation Units: {env.max_comp_units}")
        print(f"  - Agent Velocities: {env.agent_velocities}")
    
    results_training = None
    results_generalization = None
    
    # Test on training environments (if available)
    if training_envs:
        print("\n" + "="*80)
        print("TESTING ON TRAINING ENVIRONMENTS")
        print("="*80)
        
        independent_results_train = test_models(independent_dir, training_envs, test_episodes, "training")
        federated_results_train = test_models(federated_dir, training_envs, test_episodes, "training")
        
        comparison_train = compare_results(independent_results_train, federated_results_train, "training")
        results_training = {
            'independent': independent_results_train,
            'federated': federated_results_train,
            'comparison': comparison_train
        }
        print(f"✅ Training environment testing completed")
    else:
        print("\n" + "="*80)
        print("⚠️  SKIPPING TRAINING ENVIRONMENT TESTING")
        print("Training environment configuration not found")
        print("="*80)
    
    # Test on new environments (generalization)
    print("\n" + "="*80)
    print("TESTING ON NEW ENVIRONMENTS (GENERALIZATION)")
    print("="*80)
    
    independent_results_gen = test_models(independent_dir, generalization_envs, test_episodes, "generalization")
    federated_results_gen = test_models(federated_dir, generalization_envs, test_episodes, "generalization")
    
    comparison_gen = compare_results(independent_results_gen, federated_results_gen, "generalization")
    results_generalization = {
        'independent': independent_results_gen,
        'federated': federated_results_gen,
        'comparison': comparison_gen
    }
    
    # Plot results
    print(f"\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    print(f"Results available:")
    print(f"- Training environments: {'✅ Yes' if results_training else '❌ No'}")
    print(f"- Generalization environments: {'✅ Yes' if results_generalization else '❌ No'}")
    
    plot_path = f"models/test_comparison_{timestamp}.png"
    plot_test_results(results_training, results_generalization, save_path=plot_path)
    
    print(f"\nTesting completed!")
    print(f"Results plot saved to: {plot_path}")

if __name__ == "__main__":
    main()