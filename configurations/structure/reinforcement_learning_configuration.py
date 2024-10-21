from copy import deepcopy
from pathlib import Path
from typing import Optional, List
import yaml

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.from_config import NotProvided


class ReinforcementLearningConfiguration:
    def __init__(self, algorithm_name: str = 'PPO'):
        # Generale
        self.algorithm_name: str = algorithm_name

        # Framework
        self.framework: str = 'torch'

        # Resources
        self.number_gpu: int = NotProvided

        # Training
        self.architecture_name: str = NotProvided
        self.architecture_configuration: dict = NotProvided
        self.train_batch_size: int = NotProvided
        self.learning_rate: float = NotProvided
        self.grad_clip: float = NotProvided
        self.exploration_configuration: dict = NotProvided
        # PPO only
        self.use_generalized_advantage_estimator: bool = NotProvided
        self.mini_batch_size_per_learner: int = NotProvided
        self.num_sgd_iter: int = NotProvided
        self.lambda_gae: float = NotProvided
        self.clip_policy_parameter: float = NotProvided
        self.clip_value_function_parameter: float = NotProvided
        self.clip_all_parameter: float = NotProvided

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

    def clone(self):
        deep_copy = deepcopy(self)

        for attribute_name in dir(deep_copy):
            if not attribute_name.startswith('_'):
                attribute_value = getattr(deep_copy, attribute_name)
                if type(attribute_value) == type(NotProvided):
                    setattr(deep_copy, attribute_name, NotProvided)

        return deep_copy






