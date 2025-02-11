from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_dqn import DenseDQN
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.tetris_ppo import TetrisPPOTransformer, TetrisPPOCNN
from rllib_repertory.architectures.transformer_ppo import TransformerPPO

tetris_cnn = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)

tetris_cnn.ray_local_mode = False

tetris_cnn.reinforcement_learning_configuration.number_environment_runners = 2
tetris_cnn.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris_cnn.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1 / tetris_cnn.reinforcement_learning_configuration.number_environment_runners
tetris_cnn.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris_cnn.reinforcement_learning_configuration.training_name = 'cnn_ppo_v12'
tetris_cnn.reinforcement_learning_configuration.algorithm_name = 'PPO'
tetris_cnn.environment_configuration = {'observation_rgb': True}
tetris_cnn.reinforcement_learning_configuration.flatten_observations = False
tetris_cnn.reinforcement_learning_configuration.compress_observations = True
# tetris_cnn.reinforcement_learning_configuration.architecture = TetrisPPOCNN
tetris_cnn.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris_cnn.reinforcement_learning_configuration.train_batch_size = 1024 * 4
tetris_cnn.reinforcement_learning_configuration.minibatch_size = 1024 * 4
# tetris_cnn.reinforcement_learning_configuration.number_epochs = 8
tetris_cnn.reinforcement_learning_configuration.batch_mode = 'complete_episodes'

# tetris_cnn.reinforcement_learning_configuration.learning_rate = 3e-4
# tetris_cnn.reinforcement_learning_configuration.lambda_gae = 0.95
# tetris_cnn.reinforcement_learning_configuration.entropy_coefficient = 0.01

# tetris_cnn.reinforcement_learning_configuration.gradient_clip = 0.1
tetris_cnn.reinforcement_learning_configuration.clip_policy_parameter = 0.1
# tetris_cnn.reinforcement_learning_configuration.use_kullback_leibler_loss = True
# tetris_cnn.reinforcement_learning_configuration.kullback_leibler_coefficient = 0.5
# tetris_cnn.reinforcement_learning_configuration.value_function_loss_coefficient = 2
# tetris_cnn.reinforcement_learning_configuration.clip_value_function_parameter = 20

tetris_cnn.reinforcement_learning_configuration.checkpoint_frequency = 1000
tetris_cnn.reinforcement_learning_configuration.evaluation_interval = 1000