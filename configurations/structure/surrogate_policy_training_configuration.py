from datetime import timedelta
from pathlib import Path

import yaml

from lightning.clusterization_function.kmeans import Kmeans
from lightning.clusterization_loss.distance_centroid_loss import DistanceCentroidLoss


class SurrogatePolicyTrainingConfiguration:
    def __init__(self):
        self.architecture_configuration: dict = {
            'shape_layers': [128, 64, 32, 16, 32, 64, 128],
            'indexes_latent_space_to_clusterize': [6],
        }
        self.learning_rate: float = 1e-4
        self.batch_size: int = 20_000
        self.use_clusterization_loss = True
        self.clusterization_function = Kmeans
        self.clusterization_function_configuration: dict = {}
        self.clusterization_loss = DistanceCentroidLoss
        self.clusterization_loss_configuration: dict = {}

        self.number_mini_chunks: int = 2
        self.mini_chunk_size: int = 100_000
        self.data_loader_number_workers: int = 2

        self.model_checkpoint_time_interval = timedelta(minutes=10)
        self.evaluation_every_n_epoch = 10

    def to_yaml_file(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / 'surrogate_policy_training_configuration.yaml'
        configuration_dictionary = {key: value for key, value in self.__dict__.items()}

        with open(file_path, 'w') as file:
            yaml.dump(configuration_dictionary, file)

