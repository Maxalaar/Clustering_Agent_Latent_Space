import os
from pathlib import Path
from typing import Optional
import gymnasium as gym
import numpy as np

try:
    import craftium
except ImportError:
    print('craftium is not installed')

class Craftium(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        run_dir_prefix='./temporary/craftium'
        os.makedirs(run_dir_prefix, exist_ok=True)
        kwargs = dict(
            frameskip=4,
            rgb_observations=True,
            gray_scale_keepdim=False,
            run_dir_prefix=run_dir_prefix,
        )
        self.environment = gym.make('Craftium/SmallRoom-v0', render_mode=self.render_mode, **kwargs)
        # self.environment = gym.wrappers.FrameStackObservation(self.environment, 3)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.float32)
        self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        # observation = observation.astype(np.float32)
        observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)
        # observation = observation.astype(np.float32)
        observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
        return observation, reward, done, truncated, information

    def render(self, mode='human'):
        return self.environment.render()

    def close(self):
        self.environment.close()
