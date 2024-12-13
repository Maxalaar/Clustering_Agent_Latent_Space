from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.tetris_ppo import TetrisPPOTransformer

tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)

tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'

tetris.reinforcement_learning_configuration.number_environment_runners = 20
tetris.reinforcement_learning_configuration.number_environment_per_environment_runners = 1

tetris.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
tetris.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris.reinforcement_learning_configuration.minibatch_size = 32
tetris.reinforcement_learning_configuration.train_batch_size = 1024
tetris.reinforcement_learning_configuration.number_epochs = 64

tetris.reinforcement_learning_configuration.architecture = TetrisPPOTransformer
tetris.reinforcement_learning_configuration.architecture_configuration = {
    'dimension_token': 32,
    'dimension_feedforward': 64,
    'number_heads': 4,
    'number_transformer_layers': 2,
}
