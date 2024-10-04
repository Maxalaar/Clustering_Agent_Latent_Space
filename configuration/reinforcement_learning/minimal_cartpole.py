from configuration.reinforcement_learning.reinforcement_learning_configuration import ReinforcementLearningConfiguration


minimal_cartpole = ReinforcementLearningConfiguration('CartPole-v1')
minimal_cartpole.environment_configuration = {
    'action_space': 'discrete',
    'observation_space': 'continuous',
}
