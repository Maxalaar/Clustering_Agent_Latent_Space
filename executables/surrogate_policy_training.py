from datetime import timedelta

import numpy as np
import pytorch_lightning as pl
import ray
from ray.rllib.evaluation.rollout_worker import torch
from ray.tune.registry import _Registry

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from lightning.clustering_loss_functions.kmeans_loss import kmeans_loss
from lightning.h5_data_module import H5DataModule
from lightning.surrogate_policy import SurrogatePolicy
from utilities.display_h5_file_information import display_h5_file_information


def surrogate_policy_training(experimentation_configuration: ExperimentationConfiguration):
    display_h5_file_information(experimentation_configuration.trajectory_dataset_file_path)
    ray.init()
    register_environments()
    environment_creator = _Registry().get('env_creator', experimentation_configuration.environment_name)
    torch.set_float32_matmul_precision('medium')

    environment = environment_creator(experimentation_configuration.environment_configuration)

    data_module = H5DataModule(
        h5_file_path=experimentation_configuration.trajectory_dataset_file_path,
        input_dataset_name='observation',
        output_dataset_name='action_distribution_inputs',
        batch_size=20_000,
        chunk_size=100_000,
        number_workers=10,
    )
    data_module.setup()

    surrogate_policy = SurrogatePolicy(
        input_dimension=np.prod(data_module.input_shape),
        output_dimension=np.prod(data_module.output_shape),
        cluster_space_size=16,
        projection_clustering_space_shape=[128, 64, 32],
        projection_action_space_shape=[32, 64, 128],
        learning_rate=1e-4,
        clusterization_loss_function=kmeans_loss,
        clusterization_loss_function_arguments={
            'number_cluster': 4,
        },
    )

    logger = TensorBoardLogger(
        save_dir=experimentation_configuration.surrogate_policy_storage_path.parent,
        prefix='pytorch_lightning/',
        name=experimentation_configuration.surrogate_policy_storage_path.name,
    )
    checkpoint_callback = ModelCheckpoint(
        # every_n_epochs=50,
        train_time_interval=timedelta(minutes=10)
    )
    trainer = pl.Trainer(
        max_epochs=-1,
        logger=logger,
        check_val_every_n_epoch=50,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=surrogate_policy,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.validation_dataloader(),
    )


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    surrogate_policy_training(configurations.list_experimentation_configurations.pong_survivor_two_balls)
