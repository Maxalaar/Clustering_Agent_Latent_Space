from typing import Optional
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

        self.observation_space = self.environment.observation_space
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
