from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

slime_volley = ExperimentationConfiguration(
    experimentation_name='slime_volley',
    environment_name='SlimeVolleyRllib',
)

# Ray
slime_volley.ray_local_mode = True

# Reinforcement Learning
slime_volley.reinforcement_learning_configuration.training_name = 'base'
slime_volley.reinforcement_learning_configuration.architecture = DensePPO
slime_volley.reinforcement_learning_configuration.number_environment_runners = 16
slime_volley.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
slime_volley.reinforcement_learning_configuration.number_gpus_per_learner = 1
slime_volley.reinforcement_learning_configuration.train_batch_size = 40_000
slime_volley.reinforcement_learning_configuration.minibatch_size = 10_000
slime_volley.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
slime_volley.reinforcement_learning_configuration.number_epochs = 16
slime_volley.reinforcement_learning_configuration.clip_policy_parameter = 0.1

# Video Episodes
slime_volley.video_episodes_generation_configuration.number_environment_runners = 1