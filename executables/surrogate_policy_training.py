from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import ray
from ray.rllib.evaluation.rollout_worker import torch
from ray.tune.registry import _Registry

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
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
        batch_size=experimentation_configuration.surrogate_policy_training_configuration.batch_size,
        mini_chunk_size=experimentation_configuration.surrogate_policy_training_configuration.mini_chunk_size,
        number_mini_chunks=experimentation_configuration.surrogate_policy_training_configuration.number_mini_chunks,
        number_workers=experimentation_configuration.surrogate_policy_training_configuration.data_loader_number_workers,
    )
    data_module.setup()

    surrogate_policy = SurrogatePolicy(
        input_dimension=np.prod(data_module.input_shape),
        output_dimension=np.prod(data_module.output_shape),
        architecture_configuration=experimentation_configuration.surrogate_policy_training_configuration.architecture_configuration,
        learning_rate=experimentation_configuration.surrogate_policy_training_configuration.learning_rate,
        clusterization_loss_coefficient=experimentation_configuration.surrogate_policy_training_configuration.clusterization_loss_coefficient,
        clusterization_loss=experimentation_configuration.surrogate_policy_training_configuration.clusterization_loss,
        clusterization_loss_configuration=experimentation_configuration.surrogate_policy_training_configuration.clusterization_loss_configuration,
    )

    logger = TensorBoardLogger(
        save_dir=experimentation_configuration.surrogate_policy_storage_path.parent,
        prefix='pytorch_lightning/',
        name=experimentation_configuration.surrogate_policy_storage_path.name,
    )
    experimentation_configuration.surrogate_policy_training_configuration.to_yaml_file(Path(logger.log_dir))

    checkpoint_callback = ModelCheckpoint(
        train_time_interval=experimentation_configuration.surrogate_policy_training_configuration.model_checkpoint_time_interval,
    )
    trainer = pl.Trainer(
        max_epochs=-1,
        logger=logger,
        check_val_every_n_epoch=experimentation_configuration.surrogate_policy_training_configuration.evaluation_every_n_epoch,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=surrogate_policy,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.validation_dataloader(),
    )


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    surrogate_policy_training(configurations.list_experimentation_configurations.ant)
