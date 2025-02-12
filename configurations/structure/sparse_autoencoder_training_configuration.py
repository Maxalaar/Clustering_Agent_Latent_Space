from datetime import timedelta
from pathlib import Path
from typing import Optional

import yaml

from lightning_repertory.clusterization_function.kmeans import Kmeans
from lightning_repertory.clusterization_loss.distance_centroid_loss import DistanceCentroidLoss


class SparseAutoencoderTrainingConfiguration:
    def __init__(self):
        self.training_name: str = 'base'
        self.latent_dimension: int = 32
        self.sparsity_loss_coefficient: float = 0.01
        self.learning_rate: float = 1e-4
        self.batch_size: int = 20_000


        self.number_mini_chunks: int = 2
        self.mini_chunk_size: int = 100_000
        self.data_loader_number_workers: int = 2

        self.model_checkpoint_time_interval = timedelta(minutes=10)
        self.evaluation_every_n_epoch = 10

    def to_yaml_file(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / 'sparse_autoencoder_training_configuration.yaml'
        configuration_dictionary = {key: value for key, value in self.__dict__.items()}

        with open(file_path, 'w') as file:
            yaml.dump(configuration_dictionary, file)

