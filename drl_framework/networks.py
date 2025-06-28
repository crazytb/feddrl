import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .params import device

class SharedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        return x

    def get_parameters(self):
        """Get parameters for federation"""
        return [p.data.clone() for p in self.parameters()]
    
    def set_parameters(self, new_params):
        """Set parameters after federation"""
        for p, new_p in zip(self.parameters(), new_params):
            p.data.copy_(new_p)

class LocalNetwork(nn.Module):
    """Q-Network with shared MLP for DQN"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        # Shared feature extractor
        self.mlp = SharedMLP(state_dim, hidden_dim).to(device)
        
        # Q-value head
        self.q_head = nn.Linear(hidden_dim, action_dim).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values for all actions"""
        x = self.mlp(x.to(device))
        q_values = self.q_head(x)
        return q_values
    
    def get_q_value(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Get Q-value for a specific action"""
        q_values = self.forward(state)
        return q_values[:, action]
    
    def get_max_q_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get maximum Q-value and corresponding action"""
        with torch.no_grad():
            q_values = self.forward(state)
            max_q_value, max_action = q_values.max(1)
            return max_q_value, max_action
    
    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        """Select action using ε-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.randint(self.q_head.out_features)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()

def average_shared_mlp(agent_networks):
    """Average shared MLP parameters across agents"""
    with torch.no_grad():
        num_agents = len(agent_networks)
        averaged_params = [torch.zeros_like(p, device=device) 
                         for p in agent_networks[0].mlp.get_parameters()]
        
        # Sum up parameters
        for agent in agent_networks:
            params = agent.mlp.get_parameters()
            for i in range(len(averaged_params)):
                averaged_params[i] += params[i] / num_agents
        
        # Set averaged parameters
        for agent in agent_networks:
            agent.mlp.set_parameters(averaged_params)