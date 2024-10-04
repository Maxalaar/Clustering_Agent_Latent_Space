from configuration.reinforcement_learning.reinforcement_learning_configuration import ReinforcementLearningConfiguration


class ExperimentationConfiguration:
    def __init__(self, experimentation_name: str, reinforcement_learning_configuration: ReinforcementLearningConfiguration = None):
        self.experimentation_name: str = experimentation_name
        self.experimentation_path: str = './experiments' + '/' + self.experimentation_name
        self.reinforcement_learning_configuration: ReinforcementLearningConfiguration = reinforcement_learning_configuration
