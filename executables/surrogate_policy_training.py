import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import ray
from ray.rllib.evaluation.rollout_worker import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from lightning_repertory.h5_data_module import H5DataModule
from lightning_repertory.surrogate_policy import SurrogatePolicy
from utilities.display_h5_file_information import display_h5_file_information
from utilities.get_configuration_class import get_configuration_class


def surrogate_policy_training(
        experimentation_configuration: ExperimentationConfiguration,
        trajectory_dataset_path: Path,
        surrogate_policy_checkpoint_path: Optional[Path] = None,
):
    trajectory_dataset_file_path = trajectory_dataset_path / 'trajectory_dataset.h5'
    display_h5_file_information(trajectory_dataset_file_path)
    ray.init()
    torch.set_float32_matmul_precision('medium')

    data_module = H5DataModule(
        h5_file_path=trajectory_dataset_file_path,
        dataset_names=['observations', 'action_distribution_inputs'],
        batch_size=experimentation_configuration.surrogate_policy_training_configuration.batch_size,
        mini_chunk_size=experimentation_configuration.surrogate_policy_training_configuration.mini_chunk_size,
        number_mini_chunks=experimentation_configuration.surrogate_policy_training_configuration.number_mini_chunks,
        number_workers=experimentation_configuration.surrogate_policy_training_configuration.data_loader_number_workers,
    )
    data_module.setup()

    surrogate_policy = SurrogatePolicy(
        input_dimension=np.prod(data_module.input_shape),
        output_dimension=np.prod(data_module.output_shape),
        learning_rate=experimentation_configuration.surrogate_policy_training_configuration.learning_rate,
        use_clusterization_loss=experimentation_configuration.surrogate_policy_training_configuration.use_clusterization_loss,
        clusterization_function=experimentation_configuration.surrogate_policy_training_configuration.clusterization_function,
        clusterization_function_configuration=experimentation_configuration.surrogate_policy_training_configuration.clusterization_function_configuration,
        clusterization_loss=experimentation_configuration.surrogate_policy_training_configuration.clusterization_loss,
        clusterization_loss_configuration=experimentation_configuration.surrogate_policy_training_configuration.clusterization_loss_configuration,
        **experimentation_configuration.surrogate_policy_training_configuration.architecture_configuration,
        action_loss_coefficient=experimentation_configuration.surrogate_policy_training_configuration.action_loss_coefficient,
        clusterization_loss_coefficient=experimentation_configuration.surrogate_policy_training_configuration.clusterization_loss_coefficient,
    )

    logger = TensorBoardLogger(
        save_dir=experimentation_configuration.surrogate_policy_storage_path,
        prefix='pytorch_lightning/',
        name=experimentation_configuration.surrogate_policy_training_configuration.training_name,
    )
    experimentation_configuration.surrogate_policy_training_configuration.to_yaml_file(Path(logger.log_dir))

    checkpoint_callback = ModelCheckpoint(
        train_time_interval=experimentation_configuration.surrogate_policy_training_configuration.model_checkpoint_time_interval,
        save_top_k=1,
    )
    trainer = pl.trainer.Trainer(
        max_epochs=-1,
        logger=logger,
        check_val_every_n_epoch=experimentation_configuration.surrogate_policy_training_configuration.evaluation_every_n_epoch,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=surrogate_policy,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.validation_dataloader(),
        ckpt_path=surrogate_policy_checkpoint_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train surrogate policy from trajectory dataset.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    parser.add_argument(
        '--trajectory_dataset_path',
        type=str,
        help="The path of trajectory dataset directory (e.g., './experiments/cartpole/datasets/base/')"
    )

    parser.add_argument(
        '--surrogate_policy_checkpoint_path',
        type=str,
        help="Optional argument that allows resuming training by specifying a checkpoint of a substitute policy (e.g., './experiments/cartpole/surrogate_policy/base/version_[...]/checkpoints/[...].ckpt')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    trajectory_dataset_path = Path(arguments.trajectory_dataset_path)
    if not trajectory_dataset_path.is_absolute():
        trajectory_dataset_path = Path.cwd() / trajectory_dataset_path

    if arguments.surrogate_policy_checkpoint_path is not None:
        surrogate_policy_checkpoint_path = Path(arguments.surrogate_policy_checkpoint_path)
        if not surrogate_policy_checkpoint_path.is_absolute():
            surrogate_policy_checkpoint_path = Path.cwd() / surrogate_policy_checkpoint_path
    else:
        surrogate_policy_checkpoint_path = None

    surrogate_policy_training(configuration_class, trajectory_dataset_path, surrogate_policy_checkpoint_path)
