from typing import Optional
import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from gymnasium.wrappers import TransformObservation

gym.register_envs(ale_py)


class TetrisAtari(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('ALE/Tetris-v5', render_mode=self.render_mode)
        self.environment = ResizeObservation(self.environment, (96, 96))
        # self.environment = GrayscaleObservation(self.environment, keep_dim=True)

        new_observation_shape = np.array([self.environment.observation_space.shape[2], self.environment.observation_space.shape[0], self.environment.observation_space.shape[1]])
        new_observation_space = gym.spaces.Box(low=0, high=255, shape=new_observation_shape, dtype=np.float32)
        self.environment = TransformObservation(
            env=self.environment,
            func=lambda observation: np.transpose(observation,(2, 0, 1)).astype(np.float32),
            observation_space=new_observation_space
        )

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
