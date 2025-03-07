from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class CarRacing(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('CarRacing-v3', render_mode=self.render_mode)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 96, 96), dtype=np.float32)
        self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)
        observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
        return observation, reward / 100, done, truncated, information

    def render(self, mode='human'):
        return self.environment.render()

    def close(self):
        self.environment.close()
