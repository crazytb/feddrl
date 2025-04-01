import torch
import os
from drl_framework.custom_env import *
from drl_framework.networks import LocalNetwork
from drl_framework.utils import *
from drl_framework.params import device
import numpy as np
import matplotlib.pyplot as plt

# def load_agents(model_dir, model_type, n_agents, state_dim, action_dim, hidden_dim, device):
#     agents = []
#     for i in range(n_agents):
#         model = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
#         filename = f"{model_type}_shared_mlp_agent_{i}.pth"
#         path = os.path.join(model_dir, filename)
#         model.load_state_dict(torch.load(path, map_location=device))
#         model.eval()
#         agents.append(model)
#     return agents

def load_agents(model_dir, model_type, n_agents, state_dim, action_dim, hidden_dim, device):
    # averaging된 단일 모델 하나만 불러옴 (agent_0 기준)
    filename = f"{model_type}_shared_mlp_agent_0.pth"
    path = os.path.join(model_dir, filename)

    base_model = LocalNetwork(state_dim, action_dim, hidden_dim).to(device)
    base_model.load_state_dict(torch.load(path, map_location=device))
    base_model.eval()

    # 동일한 모델 복사하여 각 agent에 할당
    agents = [copy_model(base_model, device) for _ in range(n_agents)]
    return agents

def copy_model(model, device):
    import copy
    new_model = copy.deepcopy(model).to(device)
    new_model.eval()
    return new_model

def test_agents(envs, agents, episodes, cloud_controller):
    episode_rewards = np.zeros((len(agents), episodes))
    for episode in range(episodes):
        cloud_controller.reset()
        states = []
        
        for env in envs:
            state, _ = env.reset()
            states.append(state)
        
        rewards = np.zeros(len(agents))    
        for epoch in range(envs[0].max_epoch_size):
            for agent_idx, agent in enumerate(agents):
                state = states[agent_idx]
                state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_probs = agent(state_tensor)
                action = torch.argmax(action_probs, dim=1).item()
                
                # Take action
                next_state, reward, done, _, _ = envs[agent_idx].step(action)
                states[agent_idx] = next_state
                rewards[agent_idx] += reward
        
        episode_rewards[:, episode] = rewards
                
    return episode_rewards

def main():
    # 새로운 테스트 환경 설정
    num_agents = 50
    max_available_computation_units = [100]*num_agents
    agent_velocities = [100]*num_agents
    channel_patterns = ['rural']*num_agents
    cloud_controller = CloudController(max_comp_units=100)
    envs = [CustomEnv(
        max_comp_units=10,
        max_available_computation_units=max_available_computation_units[i],
        max_epoch_size=10,
        max_power=10,
        agent_velocity=agent_velocities[i],
        channel_pattern=channel_patterns[i],
        cloud_controller=cloud_controller
    ) for i in range(num_agents)]

    env = envs[0]
    state_dim = len(flatten_dict_values(env.observation_space.sample()))
    action_dim = env.action_space.n
    hidden_dim = 16
    episodes = 20

    # model_types = ['individual', 'fedavg', 'fedprox', 'fedadam', 'fedcustom']
    model_types = ['individual', 'fedavg']
    model_dir = './models'  # 예: ./saved_models/fedavg/agent_0.pth 형태
    rewards = {}

    for model_type in model_types:
        agents = load_agents(model_dir, model_type, num_agents, state_dim, action_dim, hidden_dim, device)
        rewards[model_type] = test_agents(envs, agents, episodes, cloud_controller)
        print(f"[{model_type}] 평균 리워드:", rewards[model_type].mean())
        plt.plot(rewards[model_type].mean(axis=0), label=model_type)
    plt.title("Test Results")
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.savefig("test_results.png")
    plt.show()    

if __name__ == "__main__":
    main()
