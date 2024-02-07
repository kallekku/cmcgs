import gym
import numpy as np
from .base_wrapper import EnvWrapper


class ActionRepeatWrapper(EnvWrapper):
  def __init__(self, env: gym.Env, action_repeat: int):
    super().__init__(env)
    self._action_repeat = action_repeat

  def step(self, action):
    reward = 0
    for _ in range(self._action_repeat):
        obs, r, done, info = self._env.step(action)
        reward += r or 0
        if done:
            break

    return obs, reward, done, info