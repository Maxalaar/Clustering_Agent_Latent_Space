from datetime import timedelta

from lightning.clustering_loss.kmeans_loss import KMeansLoss


class SurrogatePolicyTrainingConfiguration:
    def __init__(self):
        self.architecture_configuration: dict = {
            'cluster_space_size': 16,
            'projection_clustering_space_shape': [128, 64, 32],
            'projection_action_space_shape': [32, 64, 128],
        }
        self.learning_rate: float = 1e-4
        self.batch_size: int = 20_000
        self.clusterization_loss = KMeansLoss
        self.clusterization_loss_configuration = {}

        self.number_mini_chunks: int = 2
        self.mini_chunk_size: int = 100_000
        self.data_loader_number_workers: int = 4

        self.model_checkpoint_time_interval = timedelta(minutes=10)
        self.evaluation_every_n_epoch = 10

