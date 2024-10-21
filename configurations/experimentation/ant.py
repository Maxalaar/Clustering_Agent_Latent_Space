from configurations.structure.experimentation_configuration import ExperimentationConfiguration


ant = ExperimentationConfiguration(
    experimentation_name='ant',
    environment_name='AntRllib',
)

ant.reinforcement_learning_configuration.grad_clip = 0.5
ant.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
ant.reinforcement_learning_configuration.learning_rate = 3e-4
ant.reinforcement_learning_configuration.lambda_gae = 0.95
# ant.reinforcement_learning_configuration.train_batch_size = 2048 * 4
# ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 64 * 4

ant.reinforcement_learning_configuration.train_batch_size = 80_000
ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
# ant.reinforcement_learning_configuration.number_gpus_per_learner = 0
