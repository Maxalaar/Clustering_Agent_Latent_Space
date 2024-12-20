from pathlib import Path
from typing import Optional

from configurations.structure.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from configurations.structure.surrogate_policy_training_configuration import SurrogatePolicyTrainingConfiguration
from configurations.structure.trajectory_dataset_generation_configuration import \
    TrajectoryDatasetGenerationConfiguration
from configurations.structure.video_episodes_generation_configuration import VideoEpisodesGenerationConfiguration


class ExperimentationConfiguration:
    def __init__(self, experimentation_name: str, environment_name: str = 'CartPole', environment_configuration=None):

        if environment_configuration is None:
            environment_configuration = {}

        self.experimentation_name: str = experimentation_name
        self.experimentation_storage_path: Path = Path.cwd() / 'experiments' / self.experimentation_name
        self.video_path: Path = self.experimentation_storage_path / 'videos'
        self.dataset_path: Path = self.experimentation_storage_path / 'datasets'

        self.environment_name: str = environment_name
        self.environment_configuration: dict = environment_configuration
        self.ray_local_mode: bool = False

        # Reinforcement Learning
        self.reinforcement_learning_storage_path: Path = self.experimentation_storage_path / 'reinforcement_learning'
        self.reinforcement_learning_configuration: Optional[ReinforcementLearningConfiguration] = ReinforcementLearningConfiguration()

        # Video
        self.video_episodes_storage_path: Path = self.video_path / 'episodes'
        self.video_episodes_generation_configuration: Optional[VideoEpisodesGenerationConfiguration] = VideoEpisodesGenerationConfiguration()

        # Trajectory Dataset
        self.trajectory_dataset_generation_configuration: Optional[TrajectoryDatasetGenerationConfiguration] = TrajectoryDatasetGenerationConfiguration()

        # Surrogate Policy
        self.surrogate_policy_storage_path: Path = self.experimentation_storage_path / 'surrogate_policy'
        self.surrogate_policy_training_configuration: Optional[SurrogatePolicyTrainingConfiguration] = SurrogatePolicyTrainingConfiguration()

        # Latent Space Analysis
        self.latent_space_analysis_storage_path: Path = self.experimentation_storage_path / 'latent_space_analysis'
