import argparse

import ray
import warnings
from pathlib import Path

from ray.rllib.algorithms import Algorithm
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib_repertory.find_best_checkpoint_path import find_best_reinforcement_learning_checkpoint_path
from rllib_repertory.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration
from rllib_repertory.save_trajectory_callback import SaveTrajectoryCallback
from utilities.get_configuration_class import get_configuration_class


def trajectory_dataset_generation(experimentation_configuration: ExperimentationConfiguration, reinforcement_learning_path: Path):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    ray.init(local_mode=experimentation_configuration.ray_local_mode)

    register_environments()
    
    path_file: Path = experimentation_configuration.dataset_path / reinforcement_learning_path.name / 'trajectory_dataset_with_rending.h5'

    def create_save_trajectory_callback():
        return SaveTrajectoryCallback(
            h5_file_path=path_file,
            save_rendering=True,
            image_compression_function=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.image_compression_function,
            image_compression_configuration=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.image_compression_configuration,
            number_rendering_to_stack=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.number_rendering_to_stack,
        )

    best_checkpoints_path: Path = find_best_reinforcement_learning_checkpoint_path(reinforcement_learning_path)
    algorithm_configuration = get_checkpoint_algorithm_configuration(best_checkpoints_path)

    algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})

    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
        num_cpus_per_learner=0,
    )

    algorithm_configuration.evaluation(
        evaluation_num_env_runners=0,
    )

    algorithm_configuration.env_runners(
        explore=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.explore,
        num_env_runners=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.number_environment_runners,
        num_envs_per_env_runner=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.number_gpus_per_environment_runners,
    )

    algorithm_configuration.callbacks(create_save_trajectory_callback)
    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(best_checkpoints_path))

    def sample_for_trajectory_dataset_generation(worker: SingleAgentEnvRunner):
        for _ in range(experimentation_configuration.rendering_trajectory_dataset_generation_configuration.number_iterations):
            worker.sample(num_timesteps=experimentation_configuration.rendering_trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners)

    algorithm.env_runner_group.foreach_worker(sample_for_trajectory_dataset_generation, local_env_runner=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate trajectory dataset with rendering from policy.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    parser.add_argument(
        '--reinforcement_learning_path',
        type=str,
        help="The path of repository with the reinforcement learning checkpoint (e.g., './experiments/cartpole/reinforcement_learning/base')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_path = Path(arguments.reinforcement_learning_path)
    if not reinforcement_learning_path.is_absolute():
        reinforcement_learning_path = Path.cwd() / reinforcement_learning_path

    trajectory_dataset_generation(configuration_class, reinforcement_learning_path)
