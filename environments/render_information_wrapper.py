from typing import Any

import gymnasium as gym
from gymnasium.core import WrapperObsType


class RenderInformationWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RenderInformationWrapper, self).__init__(env)

    def reset(
        self, **kwargs
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        observation, information = self.env.reset(**kwargs)
        information['render'] = self.env.render()

        return observation, information

    def step(self, action):
        observation, reward, done, truncated, information = self.env.step(action)

        information['render'] = self.env.render()

        return observation, reward, done, truncated, information
