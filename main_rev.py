import gymnasium as gym
import torch
import os
from drl_framework.networks import SharedMLP, PolicyHead, ValueHead
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

def save_models(model_dir, shared_layer, policy_heads, value_heads, model_type):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(shared_layer.state_dict(), os.path.join(model_dir, f"{model_type}_sharedmlp.pth"))
    for i in range(len(policy_heads)):
        torch.save(policy_heads[i].state_dict(), os.path.join(model_dir, f"{model_type}_policyhead_agent_{i}.pth"))
        torch.save(value_heads[i].state_dict(), os.path.join(model_dir, f"{model_type}_valuehead_agent_{i}.pth"))

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

    env_sample = envs_provider(0)[0]
    state_dim = len(flatten_dict_values(env_sample.observation_space.sample()))
    action_dim = env_sample.action_space.n
    hidden_dim = 16
    episodes = 500
    learning_rate = 0.001
    sync_interval = 10

    # Federated Learning
    shared_layers = [SharedMLP(state_dim, hidden_dim).to(device) for _ in range(n_agents)]
    policy_heads = [PolicyHead(hidden_dim, action_dim).to(device) for _ in range(n_agents)]
    value_heads = [ValueHead(hidden_dim).to(device) for _ in range(n_agents)]
    optimizers = [torch.optim.Adam(
        list(shared_layers[i].parameters()) +
        list(policy_heads[i].parameters()) +
        list(value_heads[i].parameters()),
        lr=learning_rate) for i in range(n_agents)]

    writer_fed = SummaryWriter(log_dir="outputs/personalized_federated")

    print("\n[Personalized Federated Learning]")
    avg_reward_fed = train_personalized_federated_agents(
        envs_provider=envs_provider,
        shared_layers=shared_layers,
        policy_heads=policy_heads,
        value_heads=value_heads,
        optimizers=optimizers,
        device=device,
        episodes=episodes,
        sync_interval=sync_interval,
        writer=writer_fed
    )

    save_models("./models", shared_layers[0], policy_heads, value_heads, model_type="fedavg")

    # Individual Learning
    shared_layers_ind = [SharedMLP(state_dim, hidden_dim).to(device) for _ in range(n_agents)]
    policy_heads_ind = [PolicyHead(hidden_dim, action_dim).to(device) for _ in range(n_agents)]
    value_heads_ind = [ValueHead(hidden_dim).to(device) for _ in range(n_agents)]
    optimizers_ind = [torch.optim.Adam(
        list(shared_layers_ind[i].parameters()) +
        list(policy_heads_ind[i].parameters()) +
        list(value_heads_ind[i].parameters()),
        lr=learning_rate) for i in range(n_agents)]

    writer_ind = SummaryWriter(log_dir="outputs/individual_learning")

    print("\n[Individual Learning - No Communication]")
    avg_reward_ind = train_individual_agents(
        envs_provider=envs_provider,
        shared_layers=shared_layers_ind,
        policy_heads=policy_heads_ind,
        value_heads=value_heads_ind,
        optimizers=optimizers_ind,
        device=device,
        episodes=episodes,
        writer=writer_ind
    )

    save_models("./models", shared_layers_ind[0], policy_heads_ind, value_heads_ind, model_type="individual")

    print(f"\nAverage reward (Federated): {avg_reward_fed:.2f}")
    print(f"Average reward (Individual): {avg_reward_ind:.2f}")

if __name__ == "__main__":
    main()
