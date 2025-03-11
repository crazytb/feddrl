import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.random import default_rng
import math
import random
from .params import *

# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
# https://www.youtube.com/@cartoonsondemand

class CustomEnv(gym.Env):
    def __init__(self,
                 max_comp_units,
                 max_available_computation_units,
                 max_epoch_size,
                 max_power,
                 reward_weights=1,
                 agent_velocity=None,
                 channel_pattern=None,
                 channel_pattern_change_interval=10,
                 cloud_controller=None,
                 ):
        super().__init__()
        self.max_comp_units = max_comp_units
        # self.max_terminals = max_terminals
        self.max_epoch_size = max_epoch_size
        self.max_queue_size = max_epoch_size
        self.max_power = max_power
        self.reward_weight = reward_weights
        self.agent_velocity = agent_velocity if agent_velocity else 10
        self.channel_pattern = channel_pattern
        self.channel_pattern_change_interval = channel_pattern_change_interval
        self.max_available_computation_units = max_available_computation_units
        self.max_channel_quality = 2
        self.max_remain_epochs = max_epoch_size
        self.max_proc_times = int(np.ceil(max_epoch_size/2))
        self.energy_weight = 0.2
        
        # Cloud controller reference
        self.cloud_controller = cloud_controller
        
        # 0: process, 1: offload
        self.action_space = spaces.Discrete(2)

        self.reward = 0

        self.observation_space = spaces.Dict({
            "available_computation_units": spaces.Discrete(self.max_available_computation_units),
            # "number_of_associated_terminals": spaces.Discrete(self.max_number_of_associated_terminals),
            "channel_quality": spaces.Discrete(self.max_channel_quality),
            "remain_epochs": spaces.Discrete(self.max_remain_epochs),
            "mec_comp_units": spaces.MultiDiscrete([max_comp_units] * self.max_queue_size),
            # "mec_proc_times": spaces.MultiDiscrete([max_epoch_size] * self.max_queue_size),
            "power": spaces.Discrete(self.max_power),
            # "queue_comp_units": spaces.Discrete(max_comp_units, start=1),
            # "queue_proc_times": spaces.Discrete(max_epoch_size, start=1),
            # "queue_comp_units": spaces.MultiDiscrete([max_comp_units] * max_epoch_size),
            # "queue_proc_times": spaces.MultiDiscrete([max_epoch_size] * max_epoch_size),
        })
        self.rng = default_rng()
        self.current_obs = None
    
    def consume_cloud_resources(self, comp_units):
        if self.cloud_controller.remaining_comp_units >= comp_units:
            self.cloud_controller.remaining_comp_units -= comp_units
            return True
        else:
            self.cloud_controller.remaining_comp_units = 0
            return False
    
    def get_obs(self):
        return {"available_computation_units": self.available_computation_units,
                # "number_of_associated_terminals": self.number_of_associated_terminals,
                "channel_quality": self.channel_quality,
                "remain_epochs": self.remain_epochs,
                "mec_comp_units": self.mec_comp_units,
                # "mec_proc_times": self.mec_proc_times,
                "power": self.remaining_power,
                # "queue_comp_units": self.queue_comp_units,
                # "queue_proc_times": self.queue_proc_times
                }
    
    def stepfunc(self, thres, x):
        if x > thres:
            return 1
        else:
            return 0
    
    def change_channel_pattern(self):
        new_pattern = self.rng.choice(['urban', 'suburban', 'rural'])
        return new_pattern
    
    def change_channel_quality(self):
        if self.channel_pattern == 'urban':
            # High buildings, frequent state changes
            velocity = self.agent_velocity
            snr_thr = 20  # Higher SNR threshold in urban areas
            snr_ave = snr_thr + 15
        elif self.channel_pattern == 'suburban':
            velocity = self.agent_velocity
            snr_thr = 15
            snr_ave = snr_thr + 10
        else:  # rural
            velocity = self.agent_velocity
            snr_thr = 10  # Lower SNR threshold in rural areas
            snr_ave = snr_thr + 5
        
        f_0 = 5.9e9 # Carrier freq = 5.9GHz, IEEE 802.11bd
        speedoflight = 300000   # km/sec
        f_d = velocity/(3600*speedoflight)*f_0  # Hz
        packettime = 100*1000/MAX_EPOCH_SIZE
        # packettime = 5000    # us
        fdtp = f_d*packettime/1e6
        
        TRAN_01 = (fdtp*math.sqrt(2*math.pi*snr_thr/snr_ave))/(np.exp(snr_thr/snr_ave)-1)
        TRAN_00 = 1 - TRAN_01
        # TRAN_11 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_10 = fdtp*math.sqrt((2*math.pi*snr_thr)/snr_ave)
        TRAN_11 = 1 - TRAN_10

        if self.channel_quality == 0:  # Bad state
            return 1 if random.random() > TRAN_00 else 0
        else:   # Good state
            return 0 if random.random() > TRAN_11 else 1

    def fill_first_zero(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == 0:
                arr[i] = value
                break
        return arr

    def reset(self, seed=None, options=None):
        """
        Returns: observation
        """
        super().reset(seed=seed)

        self.available_computation_units = self.max_available_computation_units
        # self.number_of_associated_terminals = self.rng.integers(1, self.max_number_of_associated_terminals + 1, size=1)
        self.channel_quality = self.rng.integers(0, self.max_channel_quality)
        self.remain_epochs = self.max_remain_epochs
        self.mec_comp_units = np.random.randint(1, self.max_comp_units + 1, self.max_queue_size)
        # self.mec_proc_times = np.random.randint(1, self.max_proc_times + 1, self.max_queue_size)
        self.remaining_power = self.max_power
        # self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        # self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)

        self.reward = 0

        observation = self.get_obs()
        
        return observation, {}
    
    def step(self, action):
        """
        Returns: observation, reward, terminated, truncated, info
        """
        self.reward = 0
        comp_unit_ahead = self.mec_comp_units[0]
        self.mec_comp_units = np.roll(self.mec_comp_units, -1)
        self.mec_comp_units[-1] = 0
        # forwarding phase
        # 0: local process, 1: offload
        if action == 0:  # Local process
            valid_action = ((self.available_computation_units >= comp_unit_ahead) and 
                            (self.remaining_power >= 0.5))
            if valid_action:
                self.available_computation_units -= comp_unit_ahead
                power_consumed = 0.5
                self.reward = comp_unit_ahead - (self.energy_weight * power_consumed)
            else:
                self.available_computation_units = 0
                self.reward = -self.energy_weight
            self.remaining_power = np.clip(self.remaining_power - 0.5, 0, self.max_power)
        elif action == 1:   # Offload
            is_enough = self.consume_cloud_resources(comp_unit_ahead)
            valid_action = ((self.available_computation_units >= comp_unit_ahead) and 
                            (self.remaining_power >= 1) and is_enough)
            if ((self.channel_quality == 1) and valid_action):
                power_consumed = 1
                self.reward = (self.reward_weight * comp_unit_ahead) - (self.energy_weight * power_consumed)
            elif self.channel_quality == 0:
                self.reward = - self.energy_weight
            self.remaining_power = np.clip(self.remaining_power - 1, 0, self.max_power)
        elif action == 2:   # Discard
            pass
        else:
            raise ValueError("Invalid action")
        
        # self.queue_comp_units = self.rng.integers(1, self.max_comp_units + 1)
        # self.queue_proc_times = self.rng.integers(1, self.max_proc_times + 1)
            
        # Channel channel pattern
        if self.remain_epochs % self.channel_pattern_change_interval == 0:
            self.channel_pattern = self.change_channel_pattern()
        
        self.channel_quality = self.change_channel_quality()
        self.remain_epochs = self.remain_epochs - 1

        # # Processing phase
        # zeroed_index = (self.mec_proc_times == 1)
        # if zeroed_index.any():
        #     self.available_computation_units += self.mec_comp_units[zeroed_index].sum()
        #     self.reward = self.mec_comp_units[zeroed_index].sum()
        #     self.mec_proc_times = np.concatenate([self.mec_proc_times[zeroed_index == False], np.zeros(zeroed_index.sum(), dtype=int)])
        #     self.mec_comp_units = np.concatenate([self.mec_comp_units[zeroed_index == False], np.zeros(zeroed_index.sum(), dtype=int)])
        # self.mec_proc_times = np.clip(self.mec_proc_times - 1, 0, self.max_proc_times)

        next_obs = self.get_obs()

        return next_obs, self.reward, self.remain_epochs == 0, False, {}


    def render(self):
        """
        Returns: None
        """
        pass

    def close(self):
        """
        Returns: None
        """
        pass
    
    
class CloudController:
    """Manages cloud computation resources for all agents"""
    def __init__(self, max_comp_units: int):
        self.max_comp_units = max_comp_units
        self.remaining_comp_units = max_comp_units
    
    def reset(self):
        """Reset controller state at the start of each episode"""
        self.remaining_comp_units = self.max_comp_units