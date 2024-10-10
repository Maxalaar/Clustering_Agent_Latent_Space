from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class BipedalWalker(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('BipedalWalker-v3', render_mode=self.render_mode)

        # self.observation_space = self.environment.observation_space
        # self.observation_space = Box(
        #     np.array([-2*3.1415927, -5., -5., -5., -3.1415927, -5., -3.1415927, -5., -0., -3.1415927, -5., -3.1415927, -5., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
        #     np.array([2*3.1415927, 5., 5., 5., 3.1415927, 5., 3.1415927, 5., 5., 3.1415927, 5., 3.1415927, 5., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        #     (24,),
        #     np.float32
        # )
        self.observation_space = Box(
            low=np.NINF,
            high=np.PINF,
            shape=(24,),
            dtype=np.float32
        )
        self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        return observation, reward, done, truncated, information

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
