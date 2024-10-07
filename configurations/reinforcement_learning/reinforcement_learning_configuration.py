from pathlib import Path
from typing import Optional
import yaml

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.from_config import NotProvided


class ReinforcementLearningConfiguration:
    def __init__(self, algorithm: str = 'PPO'):
        # Generale
        self.ray_local_mode: bool = False
        self.algorithm: str = algorithm

        # Framework
        self.framework: str = 'torch'

        # Resources
        self.number_gpu: int = NotProvided

        # Training
        self.architecture_name: str = NotProvided
        self.architecture_configuration: dict = NotProvided
        self.train_batch_size: int = NotProvided
        self.learning_rate: float = NotProvided
        # PPO only
        self.mini_batch_size_per_learner: int = NotProvided
        self.sgd_minibatch_size: int = NotProvided
        self.num_sgd_iter: int = NotProvided

        # Environment runners
        self.batch_mode: str = NotProvided
        self.number_environment_runners: int = NotProvided
        self.number_environment_per_environment_runners: int = NotProvided
        self.number_cpus_per_environment_runners: int = NotProvided
        self.number_gpus_per_environment_runners: int = NotProvided

        # Learners
        self.number_learners: int = NotProvided
        self.number_cpus_per_learner: int = NotProvided
        self.number_gpus_per_learner: int = NotProvided

        # Evaluation
        self.evaluation_interval: int = NotProvided
        self.evaluation_num_environment_runners: int = NotProvided
        self.evaluation_duration_unit: str = NotProvided
        self.evaluation_duration: int = NotProvided
        self.evaluation_parallel_to_training: bool = NotProvided

        # Callbacks
        self.callback: DefaultCallbacks = NotProvided

        # Stopping Criterion
        self.stopping_criterion: Optional[dict] = None

        # Checkpoint Configuration
        self.number_checkpoint_to_keep: Optional[int] = None
        self.checkpoint_score_attribute: Optional[str] = None
        self.checkpoint_score_order: str = 'max'
        self.checkpoint_frequency: int = 0
        self.checkpoint_at_end: Optional[bool] = None

    def to_yaml_file(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / 'reinforcement_learning_configuration.yaml'
        configuration_dictionary = {key: value for key, value in self.__dict__.items() if value is not NotProvided}

        with open(file_path, 'w') as file:
            yaml.dump(configuration_dictionary, file)


