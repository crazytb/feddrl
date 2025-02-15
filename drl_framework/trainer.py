import torch
from typing import List
import gymnasium as gym
from .utils import *
from .networks import *
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

# SummaryWriter for TensorBoard
output_path = 'outputs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
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
        return dict(
            states=torch.FloatTensor(self.state_buf[idxs]).to(device),
            next_states=torch.FloatTensor(self.next_state_buf[idxs]).to(device),
            actions=torch.LongTensor(self.action_buf[idxs]).to(device),
            rewards=torch.FloatTensor(self.reward_buf[idxs]).to(device),
            dones=torch.FloatTensor(self.done_buf[idxs]).to(device)
        )
    
    def __len__(self):
        return self.size

@measure_time
def train_single_agent(
    env: gym.Env,
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    episodes: int = 500,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update: int = 10,
    batch_size: int = 32,
    buffer_size: int = 100000,
    min_samples: int = 1000,
    hidden_dim: int = 8
) -> np.ndarray:
    """Train a single agent using DQN with replay buffer
    
    Args:
        env: Gymnasium environment
        agent: Agent network (Q-network)
        optimizer: Optimizer for Q-network
        device: Device to use for computation
        episodes: Number of episodes to train
        gamma: Discount factor
        epsilon_start: Starting value of epsilon for ε-greedy policy
        epsilon_end: Minimum value of epsilon
        epsilon_decay: Decay rate of epsilon
        target_update: Number of episodes between target network updates
        batch_size: Size of mini-batch for training
        buffer_size: Size of replay buffer
        min_samples: Minimum number of samples before training starts
        
    Returns:
        np.ndarray: Episode rewards during training
    """
    # SummaryWriter for TensorBoard
    writer = SummaryWriter(output_path + "/" + "single" + "_" + TIMESTAMP)
    
    # Initialize target network
    # state_dim = gym.spaces.utils.flatten_space(env.observation_space).shape[0]
    state_dim = len(flatten_dict_values(env.observation_space.sample()))
    # target_net = type(agent)(env.observation_space.shape[0], env.action_space.n, hidden_dim).to(device)
    target_net = type(agent)(state_dim, env.action_space.n, hidden_dim).to(device)
    target_net.load_state_dict(agent.state_dict())
    
    # Initialize replay buffer
    memory = ReplayBuffer(state_dim, buffer_size, batch_size)
    episode_rewards = np.zeros(episodes)
    epsilon = epsilon_start

    for episode in range(episodes):
        state, _ = env.reset()
        state = flatten_dict_values(state)
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select action (ε-greedy)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = agent(state_tensor).argmax().item()
            
            # Take action in environment
            next_state, reward, done, _, _ = env.step(action)
            done_float = 0.0 if done else 1.0
            episode_reward += reward
            
            # Store transition in replay buffer
            next_state = flatten_dict_values(next_state)
            memory.store(state, action, reward, next_state, done_float)
            
            # Train if we have enough samples
            if len(memory) > min_samples:
                batch = memory.sample(device)
                
                # Compute Q(s_t, a) - current Q-values
                current_q = agent(batch['states']).gather(1, batch['actions'].unsqueeze(1))
                
                # Compute Q(s_{t+1}, a) - next Q-values
                with torch.no_grad():
                    next_q = target_net(batch['next_states']).max(1)[0].unsqueeze(1)
                    target_q = batch['rewards'].unsqueeze(1) + gamma * next_q * batch['dones'].unsqueeze(1)
                
                # Compute loss and update Q-network
                loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            
        # Store episode reward
        episode_rewards[episode] = episode_reward
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(agent.state_dict())
            
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Print progress
        print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Epsilon = {epsilon:.3f}")
        
        # Log to TensorBoard
        writer.add_scalar('Reward', episode_reward, episode)
        
    return episode_rewards

@measure_time
def train_federated_agents(
    envs: List[gym.Env],
    agents: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    device: torch.device,
    episodes: int = 500,
    sync_interval: int = 10,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update: int = 10,
    batch_size: int = 32,
    buffer_size: int = 100000,
    min_samples: int = 1000,
    hidden_dim: int = 8,
    averaging_scheme: str = 'fedavg'
) -> np.ndarray:
    """Train multiple agents using federated DQN with replay buffer
    
    Args:
        Similar to train_single_agent, but with multiple agents
        
    Returns:
        np.ndarray: Rewards for each agent (shape: [n_agents, episodes])
    """
    # SummaryWriter for TensorBoard
    writer = SummaryWriter(output_path + "/" + averaging_scheme + "_" + TIMESTAMP)
    
    """
    schemes = {
        'fedavg': fedavg_shared_mlp,
        'fedprox': fedprox_shared_mlp,
        'fedadam': fedadam_shared_mlp,
        'weighted': weighted_shared_mlp
    }
    """
    averaging_scheme = get_averaging_scheme(averaging_scheme)  # or 'fedadam', 'weighted'

    
    # Initialize target networks for each agent
    # state_dim = gym.spaces.utils.flatten_space(env.observation_space).shape[0]
    state_dim = len(flatten_dict_values(envs[0].observation_space.sample()))
    # target_nets = [type(agent)(env.observation_space.shape[0], env.action_space.n, hidden_dim).to(device)
    #               for agent in agents]
    target_nets = [type(agent)(state_dim, envs[0].action_space.n, hidden_dim).to(device)
                  for agent in agents]
    for target_net, agent in zip(target_nets, agents):
        target_net.load_state_dict(agent.state_dict())
    
    # Initialize replay buffers for each agent
    memories = [ReplayBuffer(state_dim, buffer_size, batch_size) 
               for _ in range(len(agents))]
    
    episode_rewards = np.zeros((len(agents), episodes))
    epsilon = epsilon_start

    for episode in range(episodes):
        for agent_idx, (agent, optimizer, target_net, memory) in enumerate(zip(agents, optimizers, target_nets, memories)):
            state, _ = envs[agent_idx].reset()
            state = flatten_dict_values(state)
            episode_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Select action (ε-greedy)
                if np.random.random() < epsilon:
                    action = envs[agent_idx].action_space.sample()
                else:
                    with torch.no_grad():
                        action = agent(state_tensor).argmax().item()
                
                # Take action in environment
                next_state, reward, done, _, _ = envs[agent_idx].step(action)
                done_float = 0.0 if done else 1.0
                episode_reward += reward
                
                # Store transition in replay buffer
                next_state = flatten_dict_values(next_state)
                memory.store(state, action, reward, next_state, done_float)
                
                # Train if we have enough samples
                if len(memory) > min_samples:
                    batch = memory.sample(device)
                    
                    # Compute Q(s_t, a) - current Q-values
                    current_q = agent(batch['states']).gather(1, batch['actions'].unsqueeze(1))
                    
                    # Compute Q(s_{t+1}, a) - next Q-values
                    with torch.no_grad():
                        next_q = target_net(batch['next_states']).max(1)[0].unsqueeze(1)
                        target_q = batch['rewards'].unsqueeze(1) + gamma * next_q * batch['dones'].unsqueeze(1)
                    
                    # Compute loss and update Q-network
                    loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                state = next_state
            
            agent.last_state = next_state
            agent.last_velocity = envs[agent_idx].agent_velocity
            
            # Store episode reward for this agent
            episode_rewards[agent_idx, episode] = episode_reward
            
            # Update target network
            if episode % target_update == 0:
                target_net.load_state_dict(agent.state_dict())
        
        # Synchronize agents
        if episode % sync_interval == 0:
            # average_shared_mlp(agents)
            # Update performance metrics
            for agent in agents:
                agent.update_performance_metric(episode_reward)
            
            # Apply chosen averaging scheme
            averaging_scheme(agents)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Print progress
        avg_reward = episode_rewards[:, episode].mean()
        print(f"Episode {episode + 1}: Federated Sync {episode % sync_interval == 0}, "
              f"Average Reward = {avg_reward:.1f}, Epsilon = {epsilon:.3f}")
        
        # Log to TensorBoard
        writer.add_scalar('Reward', episode_reward, episode)
    
    return episode_rewards