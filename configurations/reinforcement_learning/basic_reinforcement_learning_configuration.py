import torch

from configurations.structure.reinforcement_learning_configuration import ReinforcementLearningConfiguration

basic_reinforcement_learning_configuration = ReinforcementLearningConfiguration()
basic_reinforcement_learning_configuration.architecture_name = 'dense'
basic_reinforcement_learning_configuration.architecture_configuration = {
    'activation_function': torch.nn.ReLU(),
    'configuration_hidden_layers': [64, 64],
}

basic_reinforcement_learning_configuration.number_gpu = 1
basic_reinforcement_learning_configuration.number_gpus_per_learner = 1
basic_reinforcement_learning_configuration.number_environment_runners = 15

basic_reinforcement_learning_configuration.evaluation_interval = 50
basic_reinforcement_learning_configuration.evaluation_duration = 50

basic_reinforcement_learning_configuration.checkpoint_frequency = 50
basic_reinforcement_learning_configuration.checkpoint_score_attribute = 'evaluation/env_runners/episode_reward_mean'
basic_reinforcement_learning_configuration.number_checkpoint_to_keep = 10