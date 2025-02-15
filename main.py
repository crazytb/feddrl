import gymnasium as gym
import torch
from drl_framework.networks import LocalNetwork
from drl_framework.trainer import train_single_agent, train_federated_agents
from drl_framework.utils import *
from drl_framework.custom_env import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set the seed
    seed = 42
    np.random.seed(seed)
    seed_torch(seed)
    
    # Environment and hyperparameters
    # env = gym.make("CartPole-v1")
    env = CustomEnv(max_comp_units=10, max_epoch_size=10, max_queue_size=5, reward_weights=0.1)
    # state_dim = env.observation_space.shape[0]
    # state_dim = gym.spaces.utils.flatten_space(env.observation_space).shape[0]
    state_dim = len(flatten_dict_values(env.observation_space.sample()))
    
    action_dim = env.action_space.n
    hidden_dim = 8
    episodes = 5000
    learning_rate = 0.001
    sync_interval = 100

    # Single Agent Training
    single_agent = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
    single_optimizer = torch.optim.Adam(single_agent.parameters(), lr=learning_rate)
    single_agent_rewards = train_single_agent(
        env=env,
        agent=single_agent,
        optimizer=single_optimizer,
        device=device,
        episodes=episodes
    )

    # Federated Agents Training
    agents = [LocalNetwork(state_dim, action_dim, hidden_dim).to(device) for _ in range(3)]
    optimizers = [torch.optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents]
    federated_rewards = train_federated_agents(
        env=env,
        agents=agents,
        optimizers=optimizers,
        device=device,
        episodes=episodes,
        sync_interval=sync_interval
    )

    # Plot the results
    plot_rewards(single_agent_rewards, np.mean(federated_rewards, axis=0), sync_interval)
    print("Training Complete")

if __name__ == "__main__":
    main()