from collections import namedtuple
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TimeStep = namedtuple('TimeStep', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

class ReplayBuffer(object):
    def __init__(self, state_dim: tuple, action_dim: int, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size,) + state_dim)
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size,) +  state_dim)
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.extra = {}

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = np.zeros((self.max_size,) + value.shape)
                self.extra[key][self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.extra:
            extra = {key: torch.FloatTensor(value[ind]).to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        timestep = TimeStep(
            state = torch.FloatTensor(self.state[ind]).to(device),
            action = torch.FloatTensor(self.action[ind]).to(device), 
            next_state = torch.FloatTensor(self.next_state[ind]).to(device), 
            reward = torch.FloatTensor(self.reward[ind]).to(device), 
            not_done = torch.FloatTensor(self.not_done[ind]).to(device), 
            extra = extra
        )

        return timestep

