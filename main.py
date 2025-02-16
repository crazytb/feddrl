import gymnasium as gym
import torch
from drl_framework.networks import LocalNetwork
from drl_framework.trainer import train_single_agent, train_federated_agents
from drl_framework.utils import *
from drl_framework.custom_env import *
from drl_framework.params import device
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def main():
    print(f"Device: {device}")
    
    # Set the seed
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Environment and hyperparameters
    # env = gym.make("CartPole-v1")
    n_agents = 5
    max_available_computation_units = [10]*1 + [50]*2 + [100]*2
    agent_velocities = [10]*1 + [20]*2 + [30]*2
    channel_patterns = ['urban']*1 + ['suburban']*2 + ['rural']*2
    
    cloud_controller = CloudController(max_comp_units=100)
    envs = [CustomEnv(max_comp_units=10,
                      max_available_computation_units=max_available_computation_units[i],
                      max_epoch_size=10,
                      max_power=10,
                      agent_velocity=agent_velocities[i],
                      channel_pattern=channel_patterns[i],
                      cloud_controller=cloud_controller
                      ) for i in range(n_agents)]
    env = envs[0]
    state_dim = len(flatten_dict_values(env.observation_space.sample()))
    action_dim = env.action_space.n
    hidden_dim = 16
    episodes = 1000
    learning_rate = 0.001
    sync_interval = 100

    # Federated Agents Training
    averaging_schemes = ['fedavg', 'fedprox', 'fedadam', 'fedcustom']
    # averaging_schemes = ['fedcustom']
    for scheme in averaging_schemes:
        agents_fed = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) for _ in range(n_agents)]
        optimizers_fed = [torch.optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents_fed]
        rewards_fed = train_federated_agents(
            envs=envs,
            agents=agents_fed,
            optimizers=optimizers_fed,
            device=device,
            episodes=episodes,
            sync_interval=sync_interval,
            hidden_dim=hidden_dim,
            averaging_scheme=scheme,
            cloud_controller=cloud_controller
        )
        
    print("Training Complete")

if __name__ == "__main__":
    main()