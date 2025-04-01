import gymnasium as gym
import torch
from drl_framework.networks import SharedMLP, LocalHead
from drl_framework.trainer import train_personalized_federated_agents, train_individual_agents
from drl_framework.utils import *
from drl_framework.custom_env import *
from drl_framework.params import device
import numpy as np
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def make_envs(comp, vel, chan, cloud):
    return [CustomEnv(
                max_comp_units=10,
                max_available_computation_units=comp[i],
                max_epoch_size=10,
                max_power=10,
                agent_velocity=vel[i],
                channel_pattern=chan[i],
                cloud_controller=cloud
            ) for i in range(len(comp))]

def main():
    print(f"Device: {device}")
    
    # Set the seed
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Environment and hyperparameters
    n_agents = 5
    cloud_controller = CloudController(max_comp_units=100)

    def envs_provider(episode):
        if episode < 500:
            comp = [10]*1 + [50]*2 + [100]*2
            vel = [10]*1 + [20]*2 + [30]*2
            chan = ['urban']*1 + ['suburban']*2 + ['rural']*2
        else:
            comp = [20]*1 + [60]*2 + [120]*2
            vel = [15]*1 + [25]*2 + [35]*2
            chan = ['urban']*2 + ['rural']*3
        return make_envs(comp, vel, chan, cloud_controller)

    # Use sample env to determine dimensions
    env_sample = envs_provider(0)[0]
    state_dim = len(flatten_dict_values(env_sample.observation_space.sample()))
    action_dim = env_sample.action_space.n
    hidden_dim = 16
    episodes = 1000
    learning_rate = 0.001
    sync_interval = 10

    # Initialize shared layers and personalized heads
    shared_layers = [SharedMLP(state_dim, hidden_dim).to(device) for _ in range(n_agents)]
    local_heads = [LocalHead(hidden_dim, action_dim).to(device) for _ in range(n_agents)]
    optimizers = [torch.optim.Adam(
        list(shared_layers[i].parameters()) + list(local_heads[i].parameters()),
        lr=learning_rate) for i in range(n_agents)]

    # Shared writer for logging
    writer = SummaryWriter(log_dir="outputs/personalized_federated")

    # Training (Personalized Federated Learning)
    print("\n[Personalized Federated Learning]")
    avg_reward_fed = train_personalized_federated_agents(
        envs_provider=envs_provider,
        shared_layers=shared_layers,
        local_heads=local_heads,
        optimizers=optimizers,
        device=device,
        episodes=episodes,
        sync_interval=sync_interval,
        writer=writer
    )

    # Individual Learning (re-initialize)
    print("\n[Individual Learning - No Communication]")
    shared_layers_ind = [SharedMLP(state_dim, hidden_dim).to(device) for _ in range(n_agents)]
    local_heads_ind = [LocalHead(hidden_dim, action_dim).to(device) for _ in range(n_agents)]
    optimizers_ind = [torch.optim.Adam(
        list(shared_layers_ind[i].parameters()) + list(local_heads_ind[i].parameters()),
        lr=learning_rate) for i in range(n_agents)]
    writer_ind = SummaryWriter(log_dir="outputs/individual_learning")

    avg_reward_ind = train_individual_agents(
        envs_provider=envs_provider,
        shared_layers=shared_layers_ind,
        local_heads=local_heads_ind,
        optimizers=optimizers_ind,
        device=device,
        episodes=episodes,
        writer=writer_ind
    )

    print(f"\nAverage reward (Federated): {avg_reward_fed:.2f}")
    print(f"Average reward (Individual): {avg_reward_ind:.2f}")

if __name__ == "__main__":
    main()
