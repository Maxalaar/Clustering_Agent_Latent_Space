import gymnasium as gym


class CartPole(gym.Env):
    def __init__(self, environment_configuration=None):
        self.environment = gym.make('CartPole-v1')

        self.observation_space = self.environment.observation_space
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
