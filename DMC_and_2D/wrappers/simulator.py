import copy
from typing import Union
import dataclasses
import numpy as np

import gym
from gym import core
from .base_wrapper import EnvWrapper

@dataclasses.dataclass
class Checkpoint:
    """Holds the checkpoint state for the environment simulator."""
    needs_reset: bool
    env: gym.Env

class SimulatorWrapper(EnvWrapper):
    """A simulator model, which wraps a copy of the true environment.
    
    Assumptions:
    - The environment (including RNG) is fully copyable via `deepcopy`.
    - Environment dynamics (modulo episode resets) are deterministic.
    """
    # _checkpoint: Checkpoint
    # _env: gym.Env

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._env = copy.deepcopy(env)
        assert isinstance(env, gym.Env)
        self._needs_reset = True

        self.save_checkpoint()


    def save_checkpoint(self):
        self._checkpoint = Checkpoint(
            needs_reset=self._needs_reset,
            env = copy.deepcopy(self._env)
        )

        return self._checkpoint

    def load_checkpoint(self, checkpoint:Checkpoint=None):
        if checkpoint is not None:
            self._env = copy.deepcopy(checkpoint.env)
            self._needs_reset = checkpoint.needs_reset

        else: # load checkpoint from internal checkpoint
            self._env = copy.deepcopy(self._checkpoint.env)
            self._needs_reset = self._checkpoint.needs_reset


    def step(self, action):

        if self._needs_reset:
            raise ValueError("This model needs to be explicitly reset.")
        obs, r, d, info = self._env.step(action) # won't reset after > max_steps (1000)

        self._needs_reset = d

        return obs, r, d, info
    

    def reset(self, ):
        self._needs_reset = False
        return self._env.reset()


    @property
    def needs_reset(self,):
        return self._needs_reset