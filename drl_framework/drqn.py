import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    
class DRQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_length = x.size(1) if len(x.size()) > 2 else 1
        x = x.view(batch_size, seq_length, -1)
        
        if hidden is None:
            h, c = self.init_hidden(batch_size)
        else:
            h, c = hidden
        
        lstm_out, (new_h, new_c) = self.lstm(x, (h, c))
        q_values = self.fc(lstm_out[:, -1, :])
        return q_values, new_h, new_c

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))
