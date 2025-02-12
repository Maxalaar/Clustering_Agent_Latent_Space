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
from lightning_repertory.sparse_autoencoder import SparseAutoencoder
from utilities.display_h5_file_information import display_h5_file_information
from utilities.get_configuration_class import get_configuration_class


def sparse_autoencoder_training(
        experimentation_configuration: ExperimentationConfiguration,
        trajectory_dataset_path: Path,
        sparse_autoencoder_checkpoint_path: Optional[Path] = None,
):
    trajectory_dataset_file_path = trajectory_dataset_path / 'trajectory_dataset.h5'
    display_h5_file_information(trajectory_dataset_file_path)
    ray.init()
    torch.set_float32_matmul_precision('medium')

    data_module = H5DataModule(
        h5_file_path=trajectory_dataset_file_path,
        dataset_names=['actor_layers.1_ReLU()'],
        batch_size=experimentation_configuration.sparse_autoencoder_training_configuration.batch_size,
        mini_chunk_size=experimentation_configuration.sparse_autoencoder_training_configuration.mini_chunk_size,
        number_mini_chunks=experimentation_configuration.sparse_autoencoder_training_configuration.number_mini_chunks,
        number_workers=experimentation_configuration.sparse_autoencoder_training_configuration.data_loader_number_workers,
    )
    data_module.setup()

    surrogate_policy = SparseAutoencoder(
        input_dimension=np.prod(data_module.input_shape),
        latent_dimension=experimentation_configuration.sparse_autoencoder_training_configuration.latent_dimension,
        sparsity_loss_coefficient=experimentation_configuration.sparse_autoencoder_training_configuration.sparsity_loss_coefficient,
    )

    logger = TensorBoardLogger(
        save_dir=experimentation_configuration.sparse_autoencoder_storage_path,
        prefix='sparse_autoencoder_training/',
        name=experimentation_configuration.sparse_autoencoder_training_configuration.training_name,
    )
    experimentation_configuration.sparse_autoencoder_training_configuration.to_yaml_file(Path(logger.log_dir))

    checkpoint_callback = ModelCheckpoint(
        train_time_interval=experimentation_configuration.sparse_autoencoder_training_configuration.model_checkpoint_time_interval,
        save_top_k=1,
    )
    trainer = pl.trainer.Trainer(
        max_epochs=-1,
        logger=logger,
        check_val_every_n_epoch=experimentation_configuration.sparse_autoencoder_training_configuration.evaluation_every_n_epoch,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=surrogate_policy,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.validation_dataloader(),
        ckpt_path=sparse_autoencoder_checkpoint_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train sparse autoencoder from trajectory dataset.')
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
        '--sparse_autoencoder_checkpoint_path',
        type=str,
        help="Optional argument that allows resuming training by specifying a checkpoint of a sparse autoencoder (e.g., './experiments/cartpole/sparse_autoencoder/base/version_[...]/checkpoints/[...].ckpt')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    trajectory_dataset_path = Path(arguments.trajectory_dataset_path)
    if not trajectory_dataset_path.is_absolute():
        trajectory_dataset_path = Path.cwd() / trajectory_dataset_path

    if arguments.sparse_autoencoder_checkpoint_path is not None:
        sparse_autoencoder_checkpoint_path = Path(arguments.sparse_autoencoder_checkpoint_path)
        if not sparse_autoencoder_checkpoint_path.is_absolute():
            sparse_autoencoder_checkpoint_path = Path.cwd() / sparse_autoencoder_checkpoint_path
    else:
        sparse_autoencoder_checkpoint_path = None

    sparse_autoencoder_training(configuration_class, trajectory_dataset_path, sparse_autoencoder_checkpoint_path)
