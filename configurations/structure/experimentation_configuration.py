from pathlib import Path
from typing import Optional

from configurations.structure.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from configurations.structure.video_episodes_generation_configuration import VideoEpisodesGenerationConfiguration


class ExperimentationConfiguration:
    def __init__(self, experimentation_name: str, environment_name: str = 'CartPole', environment_configuration=None):

        if environment_configuration is None:
            environment_configuration = {}

        self.experimentation_name: str = experimentation_name
        self.experimentation_storage_path: Path = Path.cwd().parent / 'experiments' / self.experimentation_name
        self.video_path: Path = self.experimentation_storage_path / 'video'

        self.environment_name: str = environment_name
        self.environment_configuration: dict = environment_configuration

        self.reinforcement_learning_storage_path: Path = self.experimentation_storage_path / 'reinforcement_learning'
        self.reinforcement_learning_configuration: Optional[ReinforcementLearningConfiguration] = ReinforcementLearningConfiguration()

        self.video_episodes_storage_path: Path = self.video_path / 'episodes'
        self.video_episodes_generation_configuration: Optional[VideoEpisodesGenerationConfiguration] = VideoEpisodesGenerationConfiguration()
