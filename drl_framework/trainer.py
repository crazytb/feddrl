import torch
from typing import List, Callable
import gymnasium as gym
from .utils import *
from .networks import SharedMLP, PolicyHead, ValueHead
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

# SummaryWriter for TensorBoard
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create models directory
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
seed = 42
        
# class ReplayBuffer:
#     def __init__(self, state_dim: int, buffer_size: int = 100000, batch_size: int = 32):
#         self.state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
#         self.next_state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
#         self.action_buf = np.zeros(buffer_size, dtype=np.int64)
#         self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
#         self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        
#         self.max_size = buffer_size
#         self.batch_size = batch_size
#         self.ptr = 0
#         self.size = 0
        
#     def store(self, state: np.ndarray, action: int, reward: float, 
#               next_state: np.ndarray, done: float):
#         self.state_buf[self.ptr] = state
#         self.next_state_buf[self.ptr] = next_state
#         self.action_buf[self.ptr] = action
#         self.reward_buf[self.ptr] = reward
#         self.done_buf[self.ptr] = done
        
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
        
#     def sample(self, device: torch.device):
#         idxs = np.random.randint(0, self.size, size=self.batch_size)
#         return dict(
#             states=torch.FloatTensor(self.state_buf[idxs]).to(device),
#             next_states=torch.FloatTensor(self.next_state_buf[idxs]).to(device),
#             actions=torch.LongTensor(self.action_buf[idxs]).to(device),
#             rewards=torch.FloatTensor(self.reward_buf[idxs]).to(device),
#             dones=torch.FloatTensor(self.done_buf[idxs]).to(device)
#         )
    
#     def __len__(self):
#         return self.size

# def train_single_agent(
#     env: gym.Env,
#     agent: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: torch.device,
#     episodes: int = 500,
#     gamma: float = 0.99,
#     epsilon_start: float = 1.0,
#     epsilon_end: float = 0.01,
#     epsilon_decay: float = 0.995,
#     target_update: int = 10,
#     batch_size: int = 32,
#     buffer_size: int = 100000,
#     min_samples: int = 1000,
#     hidden_dim: int = 8
# ) -> np.ndarray:
#     """Train a single agent using DQN with replay buffer
    
#     Args:
#         env: Gymnasium environment
#         agent: Agent network (Q-network)
#         optimizer: Optimizer for Q-network
#         device: Device to use for computation
#         episodes: Number of episodes to train
#         gamma: Discount factor
#         epsilon_start: Starting value of epsilon for ε-greedy policy
#         epsilon_end: Minimum value of epsilon
#         epsilon_decay: Decay rate of epsilon
#         target_update: Number of episodes between target network updates
#         batch_size: Size of mini-batch for training
#         buffer_size: Size of replay buffer
#         min_samples: Minimum number of samples before training starts
        
#     Returns:
#         np.ndarray: Episode rewards during training
#     """
#     # SummaryWriter for TensorBoard
#     writer = SummaryWriter(output_path + "/" + "single" + "_" + TIMESTAMP)
    
#     # Initialize target network
#     # state_dim = gym.spaces.utils.flatten_space(env.observation_space).shape[0]
#     state_dim = len(flatten_dict_values(env.observation_space.sample()))
#     # target_net = type(agent)(env.observation_space.shape[0], env.action_space.n, hidden_dim).to(device)
#     target_net = type(agent)(state_dim, env.action_space.n, hidden_dim).to(device)
#     target_net.load_state_dict(agent.state_dict())
    
#     # Initialize replay buffer
#     memory = ReplayBuffer(state_dim, buffer_size, batch_size)
#     episode_rewards = np.zeros(episodes)
#     epsilon = epsilon_start

#     for episode in range(episodes):
#         state, _ = env.reset()
#         state = flatten_dict_values(state)
#         episode_reward = 0
#         done = False
        
#         while not done:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
#             # Select action (ε-greedy)
#             if np.random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 with torch.no_grad():
#                     action = agent(state_tensor).argmax().item()
            
#             # Take action in environment
#             next_state, reward, done, _, _ = env.step(action)
#             done_float = 0.0 if done else 1.0
#             episode_reward += reward
            
#             # Store transition in replay buffer
#             next_state = flatten_dict_values(next_state)
#             memory.store(state, action, reward, next_state, done_float)
            
#             # Train if we have enough samples
#             if len(memory) > min_samples:
#                 batch = memory.sample(device)
                
#                 # Compute Q(s_t, a) - current Q-values
#                 current_q = agent(batch['states']).gather(1, batch['actions'].unsqueeze(1))
                
#                 # Compute Q(s_{t+1}, a) - next Q-values
#                 with torch.no_grad():
#                     next_q = target_net(batch['next_states']).max(1)[0].unsqueeze(1)
#                     target_q = batch['rewards'].unsqueeze(1) + gamma * next_q * batch['dones'].unsqueeze(1)
                
#                 # Compute loss and update Q-network
#                 loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
            
#             state = next_state
            
#         # Store episode reward
#         episode_rewards[episode] = episode_reward
        
#         # Update target network
#         if episode % target_update == 0:
#             target_net.load_state_dict(agent.state_dict())
            
#         # Decay epsilon
#         epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
#         # Print progress
#         print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Epsilon = {epsilon:.3f}")
        
#         # Log to TensorBoard
#         writer.add_scalar('Reward', episode_reward, episode)
    
#     # Save the trained agent
#     model_path = os.path.join(models_dir, "single_agent.pth")
#     torch.save(agent.state_dict(), model_path)
#     print(f"Saved single agent model to {model_path}")
        
#     return episode_rewards

# def train_federated_agents(
#     envs: List[gym.Env],
#     agents: List[torch.nn.Module],
#     optimizers: List[torch.optim.Optimizer],
#     device: torch.device,
#     episodes: int = 500,
#     sync_interval: int = 10,
#     gamma: float = 0.99,
#     epsilon_start: float = 1.0,
#     epsilon_end: float = 0.01,
#     epsilon_decay: float = 0.995,
#     target_update: int = 10,
#     batch_size: int = 32,
#     buffer_size: int = 100000,
#     min_samples: int = 1000,
#     hidden_dim: int = 8,
#     averaging_scheme: str = 'fedavg',
#     cloud_controller=None
# ) -> np.ndarray:
#     # Initialize logger
#     logger = TrainingLogger(len(agents), averaging_scheme)
    
#     # Initialize other components (same as before)
#     writer = SummaryWriter(output_path + "/" + averaging_scheme + "_" + TIMESTAMP)
#     averaging_scheme = get_averaging_scheme(averaging_scheme)
#     state_dim = len(flatten_dict_values(envs[0].observation_space.sample()))
#     target_nets = [type(agent)(state_dim, envs[0].action_space.n, hidden_dim).to(device)
#                   for agent in agents]
#     for target_net, agent in zip(target_nets, agents):
#         target_net.load_state_dict(agent.state_dict())
    
#     memories = [ReplayBuffer(state_dim, buffer_size, batch_size) 
#                for _ in range(len(agents))]
    
#     episode_rewards = np.zeros((len(agents), episodes))
#     epsilon = epsilon_start

#     for episode in range(episodes):
#         cloud_controller.reset()
#         states = []
#         actions = []
#         rewards = []
        
#         # Get initial states for all agents
#         for env in envs:
#             state, _ = env.reset()
#             states.append(state)
        
#         for epoch in range(envs[0].max_epoch_size):
#             actions = []
#             rewards = []
            
#             # Process each agent
#             for agent_idx, (agent, optimizer, target_net, memory) in enumerate(
#                 zip(agents, optimizers, target_nets, memories)):
                
#                 state = states[agent_idx]
#                 state_flat = flatten_dict_values(state)
#                 state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
                
#                 # Select action
#                 if np.random.random() < epsilon:
#                     action = envs[agent_idx].action_space.sample()
#                 else:
#                     with torch.no_grad():
#                         action = agent(state_tensor).argmax().item()
                
#                 # Take action
#                 next_state, reward, done, _, _ = envs[agent_idx].step(action)
                
#                 # Store for logging
#                 actions.append(action)
#                 rewards.append(reward)
                
#                 # Update agent state
#                 agent.last_state = next_state
#                 agent.last_velocity = envs[agent_idx].agent_velocity
                
#                 # Store transition and train
#                 next_state_flat = flatten_dict_values(next_state)
#                 memory.store(state_flat, action, reward, next_state_flat, 
#                            0.0 if done else 1.0)
                
#                 if len(memory) > min_samples:
#                     batch = memory.sample(device)
#                     current_q = agent(batch['states']).gather(1, batch['actions'].unsqueeze(1))
#                     with torch.no_grad():
#                         next_q = target_net(batch['next_states']).max(1)[0].unsqueeze(1)
#                         target_q = batch['rewards'].unsqueeze(1) + gamma * next_q * batch['dones'].unsqueeze(1)
                    
#                     loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
                
#                 states[agent_idx] = next_state
#                 episode_rewards[agent_idx, episode] += reward
            
#             # Log data for this epoch
#             logger.log_step(
#                 episode=episode,
#                 epoch=epoch,
#                 states=states,
#                 actions=actions,
#                 rewards=rewards,
#                 cloud_remaining=cloud_controller.remaining_comp_units
#             )
        
#         # Synchronize agents
#         if episode % sync_interval == 0:
#             for agent in agents:
#                 agent.update_performance_metric(episode_rewards[agent_idx, episode])
#             averaging_scheme(agents)
        
#         # Update target networks
#         if episode % target_update == 0:
#             for target_net, agent in zip(target_nets, agents):
#                 target_net.load_state_dict(agent.state_dict())
        
#         # Decay epsilon
#         epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
#         # Print progress
#         avg_reward = episode_rewards[:, episode].mean()
#         print(f"Episode {episode + 1}: Federated Sync {episode % sync_interval == 0}, "
#               f"Average Reward = {avg_reward:.1f}, Epsilon = {epsilon:.3f}")
        
#         # Log to TensorBoard
#         writer.add_scalar('Average_Reward', avg_reward, episode)
#         for i in range(len(agents)):
#             writer.add_scalar(f'Agent_{i}_Reward', episode_rewards[i, episode], episode)
    
#     # Save all trained agents after completing all episodes
#     for i, agent in enumerate(agents):
#         model_path = os.path.join(models_dir, f'{averaging_scheme.__name__}_agent_{i}.pth')
#         torch.save(agent.state_dict(), model_path)
#         print(f"Saved model to {model_path}")
    
#     # Save logged data
#     logger.save_to_csv()
    
#     return episode_rewards

seed = 42

class ReplayBuffer:
    def __init__(self, state_dim: int, buffer_size: int = 100000, batch_size: int = 32):
        self.state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros(buffer_size, dtype=np.int64)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)

        self.max_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: float):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, device: torch.device):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        batch = dict(
            state=torch.tensor(self.state_buf[idxs]).to(device),
            next_state=torch.tensor(self.next_state_buf[idxs]).to(device),
            action=torch.tensor(self.action_buf[idxs]).to(device),
            reward=torch.tensor(self.reward_buf[idxs]).to(device),
            done=torch.tensor(self.done_buf[idxs]).to(device)
        )
        return batch

def average_shared_parameters(shared_layers):
    with torch.no_grad():
        avg_params = []
        for params in zip(*[list(s.parameters()) for s in shared_layers]):
            stacked = torch.stack(params, dim=0)
            avg = stacked.mean(dim=0)
            avg_params.append(avg)
        return avg_params

def plot_rewards(episode_rewards, filename="outputs/reward_plot.png"):
    avg_rewards = episode_rewards.mean(axis=0)
    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(episode_rewards):
        plt.plot(rewards, label=f"Agent {i}", alpha=0.4)
    plt.plot(avg_rewards, label="Average", color="black", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Personalized Federated RL - Agent Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def train_personalized_federated_agents(
    envs_provider: Callable[[int], List[gym.Env]],
    shared_layers: List[nn.Module],
    policy_heads: List[nn.Module],
    value_heads: List[nn.Module],
    optimizers: List[torch.optim.Optimizer],
    device: torch.device,
    episodes: int = 500,
    sync_interval: int = 10,
    gamma: float = 0.99,
    writer=None
) -> float:
    n_agents = len(shared_layers)
    episode_rewards = np.zeros((n_agents, episodes))

    def compute_discounted_rewards(rewards, gamma):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.insert(0, R)
        return torch.tensor(discounted, dtype=torch.float32)

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        envs = envs_provider(episode)
        states = [flatten_dict_values(env.reset(seed=episode)[0]) for env in envs]
        done_flags = [False] * n_agents
        total_rewards = [0.0] * n_agents

        agent_log_probs = [[] for _ in range(n_agents)]
        agent_values = [[] for _ in range(n_agents)]
        agent_rewards = [[] for _ in range(n_agents)]

        while not all(done_flags):
            actions = []
            for i in range(n_agents):
                state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                shared_out = shared_layers[i](state_tensor)
                probs = policy_heads[i](shared_out)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                actions.append(action.item())
                agent_log_probs[i].append(dist.log_prob(action))
                agent_values[i].append(value_heads[i](shared_out))

            next_states, rewards, dones, _, _ = zip(*[envs[i].step(actions[i]) for i in range(n_agents)])
            next_states = [flatten_dict_values(ns) for ns in next_states]

            for i in range(n_agents):
                agent_rewards[i].append(rewards[i])
                total_rewards[i] += rewards[i]
                states[i] = next_states[i]
                done_flags[i] = dones[i]

        # Update agents using Actor-Critic
        for i in range(n_agents):
            returns = compute_discounted_rewards(agent_rewards[i], gamma).to(device)
            values = torch.cat(agent_values[i]).to(device)
            log_probs = torch.stack(agent_log_probs[i]).to(device)

            advantage = returns - values.detach()

            policy_loss = -(log_probs * advantage).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + value_loss

            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

            if writer:
                writer.add_scalar(f"agent_{i}/reward_personalized", total_rewards[i], episode)

        # Federated averaging for shared layers
        if (episode + 1) % sync_interval == 0:
            avg_params = average_shared_parameters(shared_layers)
            for layer in shared_layers:
                for p, avg in zip(layer.parameters(), avg_params):
                    p.data.copy_(avg)

        for i in range(n_agents):
            episode_rewards[i, episode] = total_rewards[i]

    plot_rewards(episode_rewards, filename="outputs/reward_plot_personalized.png")
    return np.mean(episode_rewards)

# def train_personalized_federated_agents(
#     envs_provider: Callable[[int], List[gym.Env]],
#     shared_layers: List[SharedMLP],
#     local_heads: List[LocalHead],
#     optimizers: List[torch.optim.Optimizer],
#     device: torch.device,
#     episodes: int = 500,
#     sync_interval: int = 10,
#     gamma: float = 0.99,
#     epsilon_start: float = 1.0,
#     epsilon_end: float = 0.01,
#     epsilon_decay: float = 0.995,
#     target_update: int = 10,
#     batch_size: int = 32,
#     buffer_size: int = 100000,
#     min_samples: int = 1000,
#     writer=None
# ) -> float:
#     n_agents = len(shared_layers)
#     state_dim = shared_layers[0].fc1.in_features
#     epsilon = epsilon_start
#     episode_rewards = np.zeros((n_agents, episodes))
#     buffers = [ReplayBuffer(state_dim, buffer_size, batch_size) for _ in range(n_agents)]

#     for episode in range(episodes):
#         envs = envs_provider(episode)
#         states = [flatten_dict_values(env.reset(seed=seed)[0]) for env in envs]
#         done_flags = [False] * n_agents
#         total_rewards = [0.0] * n_agents

#         while not all(done_flags):
#             actions = []
#             for i in range(n_agents):
#                 state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
#                 shared_out = shared_layers[i](state_tensor)
#                 q_values = local_heads[i](shared_out)
#                 if np.random.rand() < epsilon:
#                     action = np.random.randint(q_values.shape[-1])
#                 else:
#                     action = torch.argmax(q_values).item()
#                 actions.append(action)

#             next_states, rewards, dones, _, _ = zip(*[envs[i].step(actions[i]) for i in range(n_agents)])
#             next_states = [flatten_dict_values(ns) for ns in next_states]

#             for i in range(n_agents):
#                 buffers[i].store(states[i], actions[i], rewards[i], next_states[i], float(dones[i]))
#                 total_rewards[i] += rewards[i]
#                 states[i] = next_states[i]
#                 done_flags[i] = dones[i]

#         # Update agents
#         for i in range(n_agents):
#             if buffers[i].size < min_samples:
#                 continue
#             batch = buffers[i].sample(device)
#             shared_out = shared_layers[i](batch['state'])
#             q_values = local_heads[i](shared_out)
#             current_q = q_values.gather(1, batch['action'].unsqueeze(1)).squeeze(1)

#             with torch.no_grad():
#                 next_shared_out = shared_layers[i](batch['next_state'])
#                 next_q = local_heads[i](next_shared_out)
#                 target_q = batch['reward'] + gamma * (1 - batch['done']) * next_q.max(1)[0]

#             loss = torch.nn.functional.mse_loss(current_q, target_q)
#             optimizers[i].zero_grad()
#             loss.backward()
#             optimizers[i].step()

#         if writer:
#             for i in range(n_agents):
#                 writer.add_scalar(f"agent_{i}/reward", total_rewards[i], episode)

#         # Federated averaging for shared layers
#         if (episode + 1) % sync_interval == 0:
#             avg_params = average_shared_parameters(shared_layers)
#             for layer in shared_layers:
#                 for p, avg in zip(layer.parameters(), avg_params):
#                     p.data.copy_(avg)

#         epsilon = max(epsilon_end, epsilon * epsilon_decay)
#         for i in range(n_agents):
#             episode_rewards[i, episode] = total_rewards[i]

#     plot_rewards(episode_rewards)
#     return np.mean(episode_rewards)

def train_individual_agents(
    envs_provider: Callable[[int], List[gym.Env]],
    shared_layers: List[nn.Module],
    policy_heads: List[nn.Module],
    value_heads: List[nn.Module],
    optimizers: List[torch.optim.Optimizer],
    device: torch.device,
    episodes: int = 500,
    gamma: float = 0.99,
    writer=None
) -> float:
    n_agents = len(shared_layers)
    episode_rewards = np.zeros((n_agents, episodes))

    def compute_discounted_rewards(rewards, gamma):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.insert(0, R)
        return torch.tensor(discounted, dtype=torch.float32)

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        envs = envs_provider(episode)
        states = [flatten_dict_values(env.reset(seed=episode)[0]) for env in envs]
        done_flags = [False] * n_agents
        total_rewards = [0.0] * n_agents

        agent_log_probs = [[] for _ in range(n_agents)]
        agent_values = [[] for _ in range(n_agents)]
        agent_rewards = [[] for _ in range(n_agents)]

        while not all(done_flags):
            actions = []
            for i in range(n_agents):
                state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
                shared_out = shared_layers[i](state_tensor)
                probs = policy_heads[i](shared_out)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                actions.append(action.item())
                agent_log_probs[i].append(dist.log_prob(action))
                agent_values[i].append(value_heads[i](shared_out))

            next_states, rewards, dones, _, _ = zip(*[envs[i].step(actions[i]) for i in range(n_agents)])
            next_states = [flatten_dict_values(ns) for ns in next_states]

            for i in range(n_agents):
                agent_rewards[i].append(rewards[i])
                total_rewards[i] += rewards[i]
                states[i] = next_states[i]
                done_flags[i] = dones[i]

        # Update agents using Actor-Critic
        for i in range(n_agents):
            returns = compute_discounted_rewards(agent_rewards[i], gamma).to(device)
            values = torch.cat(agent_values[i]).to(device)
            log_probs = torch.stack(agent_log_probs[i]).to(device)

            advantage = returns - values.detach()

            policy_loss = -(log_probs * advantage).mean()
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + value_loss

            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

            if writer:
                writer.add_scalar(f"agent_{i}/reward_individual", total_rewards[i], episode)

        for i in range(n_agents):
            episode_rewards[i, episode] = total_rewards[i]

    plot_rewards(episode_rewards, filename="outputs/reward_plot_individual.png")
    return np.mean(episode_rewards)


# def train_individual_agents(
#     envs_provider: Callable[[int], List[gym.Env]],
#     shared_layers: List[SharedMLP],
#     local_heads: List[LocalHead],
#     optimizers: List[torch.optim.Optimizer],
#     device: torch.device,
#     episodes: int = 500,
#     gamma: float = 0.99,
#     epsilon_start: float = 1.0,
#     epsilon_end: float = 0.01,
#     epsilon_decay: float = 0.995,
#     batch_size: int = 32,
#     buffer_size: int = 100000,
#     min_samples: int = 1000,
#     writer=None
# ) -> float:
#     n_agents = len(shared_layers)
#     state_dim = shared_layers[0].fc1.in_features
#     epsilon = epsilon_start
#     episode_rewards = np.zeros((n_agents, episodes))
#     buffers = [ReplayBuffer(state_dim, buffer_size, batch_size) for _ in range(n_agents)]

#     for episode in range(episodes):
#         envs = envs_provider(episode)
#         states = [flatten_dict_values(env.reset(seed=seed)[0]) for env in envs]
#         done_flags = [False] * n_agents
#         total_rewards = [0.0] * n_agents

#         while not all(done_flags):
#             actions = []
#             for i in range(n_agents):
#                 state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
#                 shared_out = shared_layers[i](state_tensor)
#                 q_values = local_heads[i](shared_out)
#                 if np.random.rand() < epsilon:
#                     action = np.random.randint(q_values.shape[-1])
#                 else:
#                     action = torch.argmax(q_values).item()
#                 actions.append(action)

#             next_states, rewards, dones, _, _ = zip(*[envs[i].step(actions[i]) for i in range(n_agents)])
#             next_states = [flatten_dict_values(ns) for ns in next_states]

#             for i in range(n_agents):
#                 buffers[i].store(states[i], actions[i], rewards[i], next_states[i], float(dones[i]))
#                 total_rewards[i] += rewards[i]
#                 states[i] = next_states[i]
#                 done_flags[i] = dones[i]

#         # Update agents independently
#         for i in range(n_agents):
#             if buffers[i].size < min_samples:
#                 continue
#             batch = buffers[i].sample(device)
#             shared_out = shared_layers[i](batch['state'])
#             q_values = local_heads[i](shared_out)
#             current_q = q_values.gather(1, batch['action'].unsqueeze(1)).squeeze(1)

#             with torch.no_grad():
#                 next_shared_out = shared_layers[i](batch['next_state'])
#                 next_q = local_heads[i](next_shared_out)
#                 target_q = batch['reward'] + gamma * (1 - batch['done']) * next_q.max(1)[0]

#             loss = torch.nn.functional.mse_loss(current_q, target_q)
#             optimizers[i].zero_grad()
#             loss.backward()
#             optimizers[i].step()

#         if writer:
#             for i in range(n_agents):
#                 writer.add_scalar(f"agent_{i}/reward_individual", total_rewards[i], episode)

#         epsilon = max(epsilon_end, epsilon * epsilon_decay)
#         for i in range(n_agents):
#             episode_rewards[i, episode] = total_rewards[i]

#     plot_rewards(episode_rewards, filename="outputs/reward_plot_individual.png")
#     return np.mean(episode_rewards)
