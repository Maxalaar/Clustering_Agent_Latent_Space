from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class LunarLander(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('LunarLander-v2', render_mode=self.render_mode)

        # self.observation_space = self.environment.observation_space
        self.observation_space = Box(np.array([-2.5, -2.5, -10., -10., -6.2831855, -10., -0., -0.]), np.array([2.5, 2.5, 10., 10., 6.2831855, 10., 1., 1.]), (8,), np.float32)
        self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        return observation, reward, done, truncated, information

    def render(self, mode='human'):
        return self.environment.render()

    def close(self):
        self.environment.close()
