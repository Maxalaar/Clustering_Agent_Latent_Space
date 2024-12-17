from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Callable, Dict
import yaml

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.typing import LearningRateOrSchedule


class ReinforcementLearningConfiguration:
    def __init__(self, algorithm_name: str = 'PPO'):
        # Generale
        self.training_name: str = 'base'
        self.algorithm_name: str = algorithm_name

        # Framework
        self.framework: str = 'torch'

        # Resources
        self.number_gpu: int = NotProvided

        # Training
        self.gamma: float = NotProvided
        self.flatten_observations: bool = True
        self.architecture: Optional[RLModule] = None
        self.architecture_configuration: Optional[DefaultModelConfig, Dict] = None
        self.train_batch_size: int = NotProvided
        self.learning_rate: float = NotProvided
        self.exploration_configuration: dict = NotProvided
        self.learner_connector: Callable = NotProvided
        self.number_epochs: int = NotProvided
        self.gradient_clip: float = NotProvided
        self.gradient_clip_by: str = NotProvided

        # DQN only
        self.tau: float = NotProvided
        self.target_network_update_frequency: int = NotProvided
        self.training_intensity: int = NotProvided
        self.replay_buffer_configuration: dict = NotProvided
        self.epsilon: Optional[LearningRateOrSchedule] = NotProvided
        self.number_steps_sampled_before_learning_starts: int = NotProvided

        # PPO only
        self.use_generalized_advantage_estimator: bool = NotProvided
        self.minibatch_size: int = NotProvided
        self.lambda_gae: float = NotProvided
        self.clip_all_parameter: float = NotProvided
        self.clip_value_function_parameter: float = NotProvided
        self.entropy_coefficient: float = NotProvided

        # Environment runners
        self.batch_mode: str = 'complete_episodes'
        self.number_environment_runners: int = NotProvided
        self.number_environment_per_environment_runners: int = NotProvided
        self.number_cpus_per_environment_runners: int = NotProvided
        self.number_gpus_per_environment_runners: float = NotProvided
        self.compress_observations: bool = NotProvided

        # Learners
        self.number_learners: int = NotProvided
        self.number_cpus_per_learner: int = NotProvided
        self.number_gpus_per_learner: int = NotProvided

        # Evaluation
        self.evaluation_interval: int = 50
        self.evaluation_num_environment_runners: int = NotProvided
        self.evaluation_duration_unit: str = NotProvided
        self.evaluation_duration: int = 100
        self.evaluation_parallel_to_training: bool = NotProvided

        # Callbacks
        self.callback: DefaultCallbacks = NotProvided

        # Stopping Criterion
        self.stopping_criterion: Optional[dict] = None

        # Checkpoint Configuration
        self.number_checkpoint_to_keep: Optional[int] = 10
        self.checkpoint_score_attribute: Optional[str] = 'evaluation/env_runners/episode_return_mean'
        self.checkpoint_score_order: str = 'max'
        self.checkpoint_frequency: int = 50
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






