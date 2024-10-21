from torch.nn import LeakyReLU
from configurations.structure.experimentation_configuration import ExperimentationConfiguration


marvine_ant = ExperimentationConfiguration(
    experimentation_name='marvine_ant',
    environment_name='AntRllib',
)

# marvine_ant.reinforcement_learning_configuration.grad_clip = 0.5
marvine_ant.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
marvine_ant.reinforcement_learning_configuration.learning_rate = 3e-4
marvine_ant.reinforcement_learning_configuration.lambda_gae = 0.95
# ant.reinforcement_learning_configuration.train_batch_size = 2048 * 4
# ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 64 * 4

marvine_ant.reinforcement_learning_configuration.train_batch_size = 80_000
marvine_ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
# ant.reinforcement_learning_configuration.number_gpus_per_learner = 0

marvine_ant.reinforcement_learning_configuration.clip_all_parameter = 2.0
marvine_ant.reinforcement_learning_configuration.clip_policy_parameter = 2.0
marvine_ant.reinforcement_learning_configuration.clip_value_function_parameter = 2.0
