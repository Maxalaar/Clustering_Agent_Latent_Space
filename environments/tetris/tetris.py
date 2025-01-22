from typing import Optional
import numpy as np
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation


class Tetris(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('tetris_gymnasium/Tetris', render_mode=self.render_mode)

        self.observation_rgb = environment_configuration.get('observation_rgb', False)
        if self.observation_rgb:
            self.environment = RgbObservation(self.environment)
            self.environment.observation_space.high[:] = 255
            self.environment = gym.wrappers.ResizeObservation(self.environment, (84, 84))
            self.environment = gym.wrappers.GrayscaleObservation(self.environment)
            self.environment = gym.wrappers.FrameStackObservation(self.environment, 4)

            self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.environment.observation_space.shape, dtype=np.float32)
            self.action_space = self.environment.action_space

        else:
            self.observation_space = self.environment.observation_space
            self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)

        if self.observation_rgb:
            observation = observation.astype(np.float32)

        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        if self.observation_rgb:
            observation = observation.astype(np.float32)

        return observation, reward/100, done, truncated, information

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
