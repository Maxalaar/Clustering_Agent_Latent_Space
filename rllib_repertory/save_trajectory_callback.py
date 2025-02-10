from pathlib import Path
import numpy as np
import h5py
from filelock import FileLock
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.single_agent_episode import SingleAgentEpisode


def add_value(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


class SaveTrajectoryCallback(DefaultCallbacks):
    def __init__(
            self,
            h5_file_path: Path,
            save_rendering: bool = False,
    ):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.save_rendering = save_rendering
        self.rendering_by_episode = {}

    def on_episode_step(self, *, episode, env, **kwargs):
        if self.save_rendering:
            # Capture du rendu graphique
            rendering = env.render()
            if episode.id_ not in self.rendering_by_episode:
                self.rendering_by_episode[episode.id_] = []
            self.rendering_by_episode[episode.id_].append(rendering)

    def on_sample_end(self, *, samples, **kwargs):
        data_dictionary = {}

        for episode in samples:
            episode: SingleAgentEpisode
            add_value(data_dictionary, 'observations', episode.get_observations()[:-1])
            add_value(data_dictionary, 'action_distribution_inputs', episode.get_extra_model_outputs('action_dist_inputs'))
            add_value(data_dictionary, 'actions', episode.get_actions())
            # add_value(data_dictionary, 'episodes_id', np.full(episode.get_actions().shape[0], episode.id_))
            add_value(data_dictionary, 'rewards', episode.get_rewards())
            add_value(data_dictionary, 'critic_values', episode.get_extra_model_outputs('critic_values'))

            activations = episode.get_extra_model_outputs('activations')
            for activations_key in activations.keys():
                add_value(data_dictionary, activations_key, activations[activations_key])

            if self.save_rendering:
                add_value(data_dictionary, 'renderings', self.rendering_by_episode.get(episode.id_, []))

        for key in data_dictionary.keys():
            if data_dictionary[key]:
                data_dictionary[key] = np.concatenate(data_dictionary[key], axis=0)
            else:
                del data_dictionary[key]

        self._save_to_hdf5(data_dictionary)

    def _save_to_hdf5(self, data_dictionary: dict):
        lock = FileLock(str(self.h5_file_path) + ".lock")
        with lock:
            with h5py.File(self.h5_file_path, "a") as h5_file:
                for key, array in data_dictionary.items():
                    if key in h5_file:
                        h5_file[key].resize((h5_file[key].shape[0] + array.shape[0]), axis=0)
                        h5_file[key][-array.shape[0]:] = array
                    else:
                        h5_file.create_dataset(
                            key,
                            data=array,
                            maxshape=(None,) + array.shape[1:],
                            chunks=True,
                            compression="gzip"
                        )
