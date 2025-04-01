import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .params import device

class SharedMLP(nn.Module):
    """Shared MLP layers for federated learning"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        # For FedProx
        self.global_params = None
        # For FedAdam
        self.m = None
        self.v = None
        self.t = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_parameters(self):
        """Get parameters for federation"""
        return [p.data.clone() for p in self.parameters()]
    
    def set_parameters(self, new_params):
        """Set parameters after federation"""
        for p, new_p in zip(self.parameters(), new_params):
            p.data.copy_(new_p)


class LocalHead(nn.Module):
    """Local head for agent-specific parameters"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.output = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x.to(device))
        

class LocalNetwork(nn.Module):
    """Q-Network with shared MLP for DQN"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        # Shared feature extractor
        self.mlp = SharedMLP(state_dim, hidden_dim).to(device)
        # Q-value head
        self.q_head1 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.q_head2 = nn.Linear(hidden_dim, action_dim).to(device)
        # Performance metric
        self.performance_metric = 0.0
        self.last_state = None
        self.last_velocity = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values for all actions"""
        x = self.mlp(x.to(device))
        x = F.relu(self.q_head1(x))
        q_values = self.q_head2(x)
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
            
    def update_performance_metric(self, reward):
        """Update performance metric for federated learning"""
        self.performance_metric = 0.9 * self.performance_metric + 0.1 * reward

def fedavg_shared_mlp(agent_networks):
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
            
def fedprox_shared_mlp(agent_networks, mu=0.01):
    """FedProx averaging with proximal term for non-IID data"""
    with torch.no_grad():
        num_agents = len(agent_networks)
        averaged_params = [torch.zeros_like(p, device=device) 
                         for p in agent_networks[0].mlp.get_parameters()]
        
        # Initialize global parameters if needed
        if agent_networks[0].mlp.global_params is None:
            agent_networks[0].mlp.global_params = [p.clone() for p in averaged_params]
        
        # Compute proximal updates
        for agent in agent_networks:
            for i, param in enumerate(agent.mlp.parameters()):
                prox_term = mu * (param.data - agent_networks[0].mlp.global_params[i])
                averaged_params[i] += (param.data - prox_term) / num_agents
        
        # Update global parameters
        agent_networks[0].mlp.global_params = [p.clone() for p in averaged_params]
        
        # Update all agents
        for agent in agent_networks:
            agent.mlp.set_parameters(averaged_params)

def fedadam_shared_mlp(agent_networks, beta1=0.9, beta2=0.999, eta=0.01):
    """FedAdam averaging for better convergence on non-IID data"""
    with torch.no_grad():
        num_agents = len(agent_networks)
        updates = [torch.zeros_like(p, device=device) 
                  for p in agent_networks[0].mlp.get_parameters()]
        
        # Initialize moments if needed
        if agent_networks[0].mlp.m is None:
            agent_networks[0].mlp.m = [torch.zeros_like(p, device=device) 
                                     for p in updates]
            agent_networks[0].mlp.v = [torch.zeros_like(p, device=device) 
                                     for p in updates]
        
        # Compute average updates
        for agent in agent_networks:
            for i, param in enumerate(agent.mlp.parameters()):
                updates[i] += param.data / num_agents
        
        # Update moments and parameters
        agent_networks[0].mlp.t += 1
        t = agent_networks[0].mlp.t
        
        for i in range(len(updates)):
            # Update biased first moment
            agent_networks[0].mlp.m[i] = beta1 * agent_networks[0].mlp.m[i] + \
                                       (1 - beta1) * updates[i]
            # Update biased second moment
            agent_networks[0].mlp.v[i] = beta2 * agent_networks[0].mlp.v[i] + \
                                       (1 - beta2) * updates[i]**2
            
            # Compute bias-corrected moments
            m_hat = agent_networks[0].mlp.m[i] / (1 - beta1**t)
            v_hat = agent_networks[0].mlp.v[i] / (1 - beta2**t)
            
            # Update parameters
            updates[i] = eta * m_hat / (torch.sqrt(v_hat) + 1e-8)
        
        # Update all agents
        for agent in agent_networks:
            agent.mlp.set_parameters(updates)

def weighted_shared_mlp(agent_networks):
    """Performance-weighted averaging for heterogeneous environments"""
    with torch.no_grad():
        # Get performance metrics
        metrics = torch.tensor([agent.performance_metric for agent in agent_networks])
        weights = F.softmax(metrics, dim=0)
        
        averaged_params = [torch.zeros_like(p, device=device) 
                         for p in agent_networks[0].mlp.get_parameters()]
        
        # Weighted averaging
        for i, agent in enumerate(agent_networks):
            params = agent.mlp.get_parameters()
            for j in range(len(averaged_params)):
                averaged_params[j] += params[j] * weights[i]
        
        # Update all agents
        for agent in agent_networks:
            agent.mlp.set_parameters(averaged_params)

def fedcustom_shared_mlp(agent_networks):
    """Custom federated averaging for UAV-MEC environment"""
    with torch.no_grad():
        num_agents = len(agent_networks)
        averaged_params = [torch.zeros_like(p, device=device) 
                         for p in agent_networks[0].mlp.get_parameters()]
        
        # Calculate weights for each agent based on environment factors
        weights = torch.zeros(num_agents, device=device)
        for i, agent in enumerate(agent_networks):
            # Channel quality score (from observation space)
            state = agent.last_state if hasattr(agent, 'last_state') else None
            channel_score = state['channel_quality']/4 if state else 0.5
            
            # Task processing score (from performance metric)
            task_score = agent.performance_metric
            
            # Velocity score (from environment)
            velocity = agent.last_velocity if hasattr(agent, 'last_velocity') else 10
            velocity_score = 1.0 - abs(velocity - 20) / 20  # Normalized around optimal velocity
            
            # Combined weight
            weight = 0.3 * channel_score + 0.3 * velocity_score + 0.4 * task_score
            weights[i] = weight

        # Normalize weights using softmax
        weights = F.softmax(weights, dim=0)
        
        # Weighted averaging with adaptive proximal term
        for i, agent in enumerate(agent_networks):
            params = agent.mlp.get_parameters()
            
            # Calculate adaptive proximal term based on channel quality
            state = agent.last_state if hasattr(agent, 'last_state') else None
            channel_quality = state['channel_quality'] if state else 2
            velocity = agent.last_velocity if hasattr(agent, 'last_velocity') else 10
            
            base_mu = 0.01
            channel_factor = 1 + (1 - channel_quality/4)  # Higher for worse channel
            velocity_factor = 1 + abs(velocity - 20)/20   # Higher for extreme velocities
            mu = base_mu * channel_factor * velocity_factor
            
            # Apply proximal term if global params exist
            if hasattr(agent.mlp, 'global_params') and agent.mlp.global_params is not None:
                for j, param in enumerate(params):
                    prox_term = mu * (param - agent.mlp.global_params[j])
                    averaged_params[j] += weights[i] * (param - prox_term)
            else:
                for j, param in enumerate(params):
                    averaged_params[j] += weights[i] * param
        
        # Update global parameters
        for agent in agent_networks:
            if not hasattr(agent.mlp, 'global_params') or agent.mlp.global_params is None:
                agent.mlp.global_params = [p.clone() for p in averaged_params]
            else:
                for i, param in enumerate(agent.mlp.global_params):
                    param.copy_(averaged_params[i])
            
            # Update agent parameters
            agent.mlp.set_parameters(averaged_params)
            
def individual_shared_mlp(agent_networks):
    """
    No federation - each agent keeps its own parameters.
    This function is a no-op placeholder that does nothing,
    allowing each agent to train independently.
    """
    # No parameter sharing or averaging
    pass

# Add custom scheme to the averaging schemes dictionary
def get_averaging_scheme(scheme_name: str):
    """Get averaging scheme by name"""
    schemes = {
        'fedavg': fedavg_shared_mlp,
        'fedprox': fedprox_shared_mlp,
        'fedadam': fedadam_shared_mlp,
        'weighted': weighted_shared_mlp,
        'fedcustom': fedcustom_shared_mlp,
        'individual': individual_shared_mlp,
    }
    return schemes.get(scheme_name, fedavg_shared_mlp)