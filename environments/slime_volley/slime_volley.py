from typing import Optional

import gymnasium
import numpy as np
from environments.slime_volley.old_slime_volley import SlimeVolleyEnv


class SlimeVolley(gymnasium.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 60,
        }

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = SlimeVolleyEnv()

        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = gymnasium.spaces.MultiDiscrete(np.array([2, 2, 2]))

    def reset(self, seed=None, options=None):
        observation = self.environment.reset()
        information = {}
        return observation, information

    def step(self, action):
        observation, reward, done, information = self.environment.step(action)
        truncated = False
        return observation, reward, done, truncated, information

    def render(self):
        return self.environment.render(mode='rgb_array')

    def close(self):
        self.environment.close()
