import sys
from typing import Dict
import os

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from drl_framework.custom_env import *
from drl_framework.params import *

# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.Linear1 = nn.Linear(state_space, 64)
        self.Linear2 = nn.Linear(64, 64)
        self.Linear3 = nn.Linear(64, action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0,1)
        else:
            return self.forward(obs).argmax().item()

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


def train(q_net=None, target_q_net=None, replay_buffer=None,
          device=None, 
          optimizer = None,
          batch_size=64,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = replay_buffer.sample()

    
    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

    # Define loss
    q_target_max = target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma*q_target_max*dones
    q_out = q_net(states)
    q_a = q_out.gather(1, actions)

    # Multiply Importance Sampling weights to loss        
    loss = F.smooth_l1_loss(q_a, targets)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":

    # Env parameters
    model_name = "DQN_POMDP"
    seed = 42

    # Set gym environment
    env = CustomEnv(max_comp_units=MAX_COMP_UNITS,
                max_epoch_size=MAX_EPOCH_SIZE,
                max_queue_size=MAX_QUEUE_SIZE,
                reward_weights=REWARD_WEIGHTS)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # Summarywriter setting
    output_path = 'outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    writer = SummaryWriter(output_path + "/" + model_name + "_" + f"{TIMESTAMP}")

    state, info = env.reset()
    # n_observation = len(env.flatten_dict_values(state))
    # Hidden state settings
    target_key = ['available_computation_units',
                  'channel_quality',
                  'remain_epochs',
                  'mec_comp_units',
                  'mec_proc_times',
                  'queue_comp_units',
                  'queue_proc_times']
    state = {key: state[key] for key in target_key}
    n_observation = len(env.flatten_dict_values(state))
    # n_states_to_be_hidden = 2*MAX_QUEUE_SIZE

    # Create Q functions
    Q = Q_net(state_space=n_observation, 
              action_space=env.action_space.n).to(device)
    Q_target = Q_net(state_space=n_observation, 
                     action_space=env.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Create Replay buffer
    replay_buffer = ReplayBuffer(n_observation,
                                 size=buffer_len,
                                 batch_size=batch_size)

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start

    # Train
    for i in range(episodes):
        s, _ = env.reset()
        s = {key: s[key] for key in target_key}
        s = env.flatten_dict_values(s)
        done = False
        
        for t in range(max_step):

            # Get action
            a = Q.sample_action(torch.from_numpy(s).float().to(device), epsilon)

            # Do action
            s_prime, r, done, _, _ = env.step(a)
            s_prime = {key: s_prime[key] for key in target_key}
            s_prime = env.flatten_dict_values(s_prime)

            # make data
            done_mask = 0.0 if done else 1.0

            replay_buffer.put(s, a, r/100, s_prime, done_mask)            

            s = s_prime
            
            score += r
            score_sum += r

            if len(replay_buffer) >= min_buffer_len:
                train(Q, Q_target, replay_buffer, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
    
            if done:
                break
        
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(replay_buffer), epsilon*100))
            score_sum=0.0

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0
    
    save_model(Q, model_name+"_"+f"{TIMESTAMP}.pth")
    writer.close()
    env.close()