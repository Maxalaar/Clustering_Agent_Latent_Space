from configurations.structure.experimentation_configuration import ExperimentationConfiguration


ant = ExperimentationConfiguration(
    experimentation_name='ant',
    environment_name='AntRllib',
)

ant.reinforcement_learning_configuration.train_batch_size = 80_000
ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
# ant.reinforcement_learning_configuration.number_gpus_per_learner = 0
