from typing import Any

import gymnasium as gym
from gymnasium.core import WrapperObsType


class InformationWrapper(gym.Wrapper):
    def __init__(self, environment, save_rendering: bool = False):
        super(InformationWrapper, self).__init__(environment)
        self.save_rendering: bool = save_rendering

    def reset(
        self, **kwargs
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        observation, information = self.env.reset(**kwargs)
        if self.save_rendering:
            information['render'] = self.env.render()

        return observation, information

    def step(self, action):
        observation, reward, done, truncated, information = self.env.step(action)

        if self.save_rendering:
            information['render'] = self.env.render()
        information['action'] = action

        return observation, reward, done, truncated, information
