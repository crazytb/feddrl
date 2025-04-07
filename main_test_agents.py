import torch
import os
from drl_framework.custom_env import *
from drl_framework.networks import SharedMLP, PolicyHead, ValueHead
from drl_framework.utils import *
from drl_framework.params import device
import numpy as np
import matplotlib.pyplot as plt

def copy_model(model, device):
    import copy
    new_model = copy.deepcopy(model).to(device)
    new_model.eval()
    return new_model

def average_head_state_dicts(model_dir, model_type, head_class, n_models, device):
    avg_state_dict = None

    for i in range(n_models):
        path = os.path.join(model_dir, f"{model_type}_{head_class.__name__.lower()}_agent_{i}.pth")
        state_dict = torch.load(path, map_location=device)

        if avg_state_dict is None:
            avg_state_dict = {k: state_dict[k].clone().float() for k in state_dict}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float()

    for k in avg_state_dict:
        avg_state_dict[k] /= n_models

    model = head_class(avg_state_dict[list(avg_state_dict.keys())[0]].shape[0],
                       avg_state_dict[list(avg_state_dict.keys())[-1]].shape[0]).to(device)
    model.load_state_dict(avg_state_dict)
    model.eval()
    return model

def load_agents(model_dir, model_type, n_agents, state_dim, action_dim, hidden_dim, device, n_models_available=5):
    shared_path = os.path.join(model_dir, f"{model_type}_shared_mlp.pth")
    shared_model = SharedMLP(state_dim, hidden_dim).to(device)
    shared_model.load_state_dict(torch.load(shared_path, map_location=device))
    shared_model.eval()

    # 평균 policy/value head 계산
    avg_policy_head = average_head_state_dicts(model_dir, model_type, PolicyHead, n_models_available, device)
    avg_value_head = average_head_state_dicts(model_dir, model_type, ValueHead, n_models_available, device)

    agents = [
        (copy_model(shared_model, device), copy_model(avg_policy_head, device), copy_model(avg_value_head, device))
        for _ in range(n_agents)
    ]
    return agents

def test_agents(envs, agents, episodes, cloud_controller):
    episode_rewards = np.zeros((len(agents), episodes))
    episode_values = np.zeros((len(agents), episodes))

    for episode in range(episodes):
        cloud_controller.reset()
        states = []

        for env in envs:
            state, _ = env.reset()
            states.append(state)

        rewards = np.zeros(len(agents))
        values = np.zeros(len(agents))

        for epoch in range(envs[0].max_epoch_size):
            for agent_idx, (shared_model, policy_head, value_head) in enumerate(agents):
                state = states[agent_idx]
                state_tensor = torch.FloatTensor(flatten_dict_values(state)).unsqueeze(0).to(device)

                with torch.no_grad():
                    shared_out = shared_model(state_tensor)
                    action_probs = policy_head(shared_out)
                    value_est = value_head(shared_out)

                action = torch.argmax(action_probs, dim=1).item()
                next_state, reward, done, _, _ = envs[agent_idx].step(action)

                states[agent_idx] = next_state
                rewards[agent_idx] += reward
                values[agent_idx] += value_est.item()

        episode_rewards[:, episode] = rewards
        episode_values[:, episode] = values / envs[0].max_epoch_size

    return episode_rewards, episode_values

def main():
    num_agents = 10
    max_available_computation_units = [5]*num_agents
    agent_velocities = [50]*num_agents
    channel_patterns = ['urban']*num_agents
    cloud_controller = CloudController(max_comp_units=500)
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

    model_types = ['individual', 'fedavg']
    model_dir = './models'
    rewards = {}
    values = {}

    for model_type in model_types:
        agents = load_agents(model_dir, model_type, num_agents, state_dim, action_dim, hidden_dim, device)
        rewards[model_type], values[model_type] = test_agents(envs, agents, episodes, cloud_controller)
        print(f"[{model_type}] 평균 리워드: {rewards[model_type].mean():.2f}, 평균 가치: {values[model_type].mean():.2f}")
        plt.plot(rewards[model_type].mean(axis=0), label=f"{model_type} Reward")
        plt.plot(values[model_type].mean(axis=0), label=f"{model_type} Value", linestyle='--')

    plt.title("Test Results")
    plt.xlabel("Episode")
    plt.ylabel("Reward / Value")
    plt.legend()
    plt.grid()
    plt.savefig("test_results.png")
    plt.show()

if __name__ == "__main__":
    main()
