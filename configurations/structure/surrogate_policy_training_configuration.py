from datetime import timedelta
from pathlib import Path
import yaml

from lightning.kmeans_loss import KmeansLoss


class SurrogatePolicyTrainingConfiguration:
    def __init__(self):
        self.architecture_configuration: dict = {
            'shape_layers': [128, 64, 32, 16, 32, 64, 128],
            'latent_space_to_clusterize': [False, False, False, True, False, False, False],
        }
        self.learning_rate: float = 1e-4
        self.batch_size: int = 20_000
        self.clusterization_loss = KmeansLoss
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

