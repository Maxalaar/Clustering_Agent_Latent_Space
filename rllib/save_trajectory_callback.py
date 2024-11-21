from pathlib import Path

from gymnasium.vector import SyncVectorEnv
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
import gymnasium as gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Optional, Dict, List
import h5py
import numpy as np
from filelock import FileLock


def get_last_episode_id(h5_file):
    if 'episode_id' in h5_file:
        return h5_file['episode_id'][-1]
    else:
        return -1


class SaveTrajectoryCallback(DefaultCallbacks):
    def __init__(
            self,
            h5_file_path: Path,
            save_rendering: bool,
            image_compression_function=None,
            image_compression_configuration=None,
            number_rendering_to_stack: int = 0,
    ):

        self.h5_file_path: Path = h5_file_path
        self.save_rendering: bool = save_rendering
        self.rendering_by_episode_id: dict = {}
        self.image_compression_function = image_compression_function
        self.image_compression_configuration = image_compression_configuration
        self.number_rendering_to_stack: int = number_rendering_to_stack

    def on_episode_step(
            self,
            episode: SingleAgentEpisode,
            env_index: int,
            env: SyncVectorEnv,
            **kwargs,
    ) -> None:
        if self.save_rendering:
            if episode.id_ not in self.rendering_by_episode_id:
                self.rendering_by_episode_id[episode.id_] = []

            rending = env.envs[env_index].render()

            if self.image_compression_function is not None:
                rending = self.image_compression_function(image=rending, **self.image_compression_configuration)

            self.rendering_by_episode_id[episode.id_].append(rending)

    def on_sample_end(
            self,
            samples: List[SingleAgentEpisode],
            env_runner=None,
            metrics_logger=None,
            **kwargs,
    ):
        local_episodes_id = []
        observations = []
        actions = []
        action_distribution_inputs = []
        rewards = []
        renderings = []

        for index, trajectory in enumerate(samples):
            trajectory: SingleAgentEpisode
            local_episodes_id.append(np.full(trajectory.get_actions().shape[0], index))
            observations.append(trajectory.get_observations()[:-1])
            actions.append(trajectory.get_actions())
            action_distribution_inputs.append(trajectory.get_extra_model_outputs('action_dist_inputs'))
            rewards.append(trajectory.get_rewards())

            if self.save_rendering:
                renderings_trajectory = np.stack(self.rendering_by_episode_id[trajectory.id_])

                if self.number_rendering_to_stack > 0:
                    # weights = np.linspace(1, 0, self.number_rendering_to_stack + 1)
                    n = self.number_rendering_to_stack + 1
                    weights = np.exp(-np.linspace(0, 5, n))
                    weights /= weights.sum()
                    mean_renderings_trajectory = np.zeros_like(renderings_trajectory)
                    for i in range(renderings_trajectory.shape[0]):
                        for j in range(self.number_rendering_to_stack):
                            index = i - j
                            if index >= 0:
                                mean_renderings_trajectory[i] += (weights[j] * renderings_trajectory[index]).astype(np.uint8)

                    renderings_trajectory = mean_renderings_trajectory

                renderings.append(renderings_trajectory)

                del self.rendering_by_episode_id[trajectory.id_]

        local_episodes_id = np.concatenate(local_episodes_id, axis=0)
        dataset_dictionary = {
            'observations': np.concatenate(observations, axis=0),
            'actions': np.concatenate(actions, axis=0),
            'action_distribution_inputs': np.concatenate(action_distribution_inputs, axis=0),
            'rewards': np.concatenate(rewards, axis=0),
        }
        if self.save_rendering:
            renderings = np.concatenate(renderings, axis=0)

            dataset_dictionary['renderings'] = renderings

        self.write(local_episodes_id, dataset_dictionary)

    def write(self, episodes_id, dataset_dictionary: Dict[str, np.ndarray]):
        lock = FileLock(self.h5_file_path.with_suffix(self.h5_file_path.suffix + ".lock"))

        with lock:
            with h5py.File(self.h5_file_path, 'a') as h5_file:
                global_episode_id = episodes_id + get_last_episode_id(h5_file) + 1
                dataset_dictionary['episodes_id'] = global_episode_id

                for dataset_name in dataset_dictionary.keys():
                    data = dataset_dictionary[dataset_name]

                    if dataset_name in h5_file:
                        dataset = h5_file[dataset_name]
                        dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                        dataset[-data.shape[0]:] = data
                    else:
                        max_shape = (None,) + data.shape[1:]
                        h5_file.create_dataset(dataset_name, data=data, maxshape=max_shape)