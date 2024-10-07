from pathlib import Path
from typing import Optional

from configurations.reinforcement_learning.reinforcement_learning_configuration import ReinforcementLearningConfiguration


class ExperimentationConfiguration:
    def __init__(self, experimentation_name: str, environment_name: str = 'CartPole', environment_configuration=None):

        if environment_configuration is None:
            environment_configuration = {}

        self.experimentation_name: str = experimentation_name
        self.experimentation_storage_path: Path = Path.cwd().parent / 'experiments' / self.experimentation_name
        self.reinforcement_learning_storage_path: Path = self.experimentation_storage_path / 'reinforcement_learning'

        self.environment_name: str = environment_name
        self.environment_configuration: dict = environment_configuration

        self.reinforcement_learning_configuration: Optional[ReinforcementLearningConfiguration] = None
