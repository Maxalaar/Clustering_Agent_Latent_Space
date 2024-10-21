from time import sleep
from typing import Optional
import torch.nn.functional as F

import h5py
import numpy as np
import pytorch_lightning as pl
import ray
from ray.rllib.evaluation.rollout_worker import torch
from ray.tune.registry import _Registry
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Sampler

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from models.architectures.pytorch_lightning.surrogate_policy import SurrogatePolicy


def get_h5_shapes(h5_file_path: Path, dataset_name: str):
    shape = None
    with h5py.File(h5_file_path, 'r') as file:
        if dataset_name in file:
            dataset = file[dataset_name]
            if len(dataset.shape) > 1:
                shape = dataset.shape[1:]
            else:
                shape = (np.max(dataset) + 1,)
        else:
            print('Dataset ' + str(dataset_name) + ' not found in the file.')
    return shape


class H5Dataset(Dataset):
    def __init__(self, file_path: Path, chunk_size: int, input_dataset_name: str, output_dataset_name: Optional[str] = None):
        self.file_path: Path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')
        self.chunk_size: int = chunk_size

        self.input_dataset = self.h5_file[input_dataset_name]
        self.output_dataset = None
        if output_dataset_name is not None:
            self.output_dataset = self.h5_file[output_dataset_name]

        self.idx_current_chunk_start = None
        self.idx_current_chunk_stop = None

        if len(self.input_dataset) != len(self.output_dataset):
            raise ValueError(f"Error: input_dataset length ({len(self.input_dataset)}) does not match output_dataset length ({len(self.output_dataset)}).")
        self.dataset_size = len(self.input_dataset)

        self.input_current_chunk_data: Optional[np.ndarray] = None
        self.output_current_chunk_data: Optional[np.ndarray] = None

    def load_chunk(self, idx):
        # self.idx_current_chunk_start = (idx[0] // self.chunk_size) * self.chunk_size
        self.idx_current_chunk_start = (idx // self.chunk_size) * self.chunk_size
        self.idx_current_chunk_stop = min(self.idx_current_chunk_start + self.chunk_size, self.dataset_size)

        self.input_current_chunk_data = np.array(self.input_dataset[self.idx_current_chunk_start:self.idx_current_chunk_stop])
        if self.output_dataset is not None:
            self.output_current_chunk_data = np.array(self.output_dataset[self.idx_current_chunk_start:self.idx_current_chunk_stop])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # idx = np.array(idx)

        if self.idx_current_chunk_start is None:
            self.load_chunk(idx)

        # idx_in_current_chunk = (self.idx_current_chunk_start <= idx) & (idx < self.idx_current_chunk_stop)
        # if not np.all(idx_in_current_chunk):
        #     self.load_chunk(idx)
        if not self.idx_current_chunk_start <= idx < self.idx_current_chunk_stop:
            self.load_chunk(idx)

        local_idx = idx - self.idx_current_chunk_start
        if self.output_dataset is not None:
            return self.input_current_chunk_data[local_idx], self.output_current_chunk_data[local_idx]
        else:
            return self.input_current_chunk_data[local_idx]


class BatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __iter__(self):
        return (range(i, min(i + self.batch_size, self.dataset_size)) for i in range(0, self.dataset_size, self.batch_size))

    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


# def collate_fn_padding(batch):
#     # print('oki')
#     # # Trouve la taille maximale dans le batch
#     # batch_size = len(batch)
#     #
#     # # Applique du padding pour que tous les éléments aient la même taille
#     # batch_padded = [F.pad(item, (0, max_size[1] - item.size(1), 0, max_size[0] - item.size(0))) for item in batch]
#     print(type(batch))
#     print(batch)
#     return torch.stack(batch)


class H5DataModule(pl.LightningDataModule):
    def __init__(
            self,
            h5_file_path: Path,
            input_dataset_name: str,
            output_dataset_name: Optional[str] = None,
            batch_size: int = 64,
            chunk_size: int = 64,
            validation_split: float = 0.2,
            number_workers: int = 1,
    ):
        super().__init__()

        if chunk_size < batch_size:
            raise ValueError(f"Error: chunk_size ({chunk_size}) cannot be smaller than batch_size ({batch_size}).")

        self.h5_file_path: Path = h5_file_path
        self.input_dataset_name: str = input_dataset_name
        self.output_dataset_name: Optional[str] = output_dataset_name
        self.batch_size: int = batch_size
        self.validation_split = validation_split
        self.number_workers: int = number_workers
        self.chunk_size: int = chunk_size

        self.input_shape: Optional[tuple] = None
        self.output_shape: Optional[tuple] = None
        self.train_dataset: Optional[H5Dataset] = None
        self.test_dataset: Optional[H5Dataset] = None

    def setup(self, stage=None):
        self.input_shape = get_h5_shapes(self.h5_file_path, self.input_dataset_name)
        if self.output_dataset_name is not None:
            self.output_shape = get_h5_shapes(self.h5_file_path, self.output_dataset_name)

        self.train_dataset = H5Dataset(
            file_path=self.h5_file_path,
            input_dataset_name=self.input_dataset_name,
            output_dataset_name=self.output_dataset_name,
            chunk_size=self.chunk_size,
        )

        self.test_dataset = H5Dataset(
            file_path=self.h5_file_path,
            input_dataset_name=self.input_dataset_name,
            output_dataset_name=self.output_dataset_name,
            chunk_size=self.chunk_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.number_workers,
            # sampler=BatchSampler(dataset_size=len(self.train_dataset), batch_size=self.batch_size),
            # collate_fn=collate_fn_padding,
            # drop_last=True,
        )

    def validation_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.number_workers,
            # sampler=BatchSampler(dataset_size=len(self.test_dataset), batch_size=self.batch_size),
            # collate_fn=collate_fn_padding,
            # drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.number_workers,
            # sampler=BatchSampler(dataset_size=len(self.test_dataset), batch_size=self.batch_size),
            # collate_fn=collate_fn_padding,
            # drop_last=True,
        )


def surrogate_policy_training(experimentation_configuration: ExperimentationConfiguration):
    ray.init()
    register_environments()
    environment_creator = _Registry().get('env_creator', experimentation_configuration.environment_name)

    environment = environment_creator(experimentation_configuration.environment_configuration)

    data_module = H5DataModule(
        h5_file_path=experimentation_configuration.trajectory_dataset_file_path,
        input_dataset_name='observation',
        output_dataset_name='action_logit',
        chunk_size=100_000,
        batch_size=20_000,
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
    )

    logger = TensorBoardLogger(
        save_dir=experimentation_configuration.surrogate_policy_storage_path.parent,
        prefix='pytorch_lightning/',
        name=experimentation_configuration.surrogate_policy_storage_path.name,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1000,
    )
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=10000000,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=surrogate_policy,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.validation_dataloader(),
    )


if __name__ == '__main__':
    from configurations.experimentation.cartpole import cartpole
    from configurations.experimentation.bipedal_walker import bipedal_walker
    from configurations.experimentation.lunar_lander import lunar_lander
    from configurations.experimentation.ant import ant
    from configurations.experimentation.pong_survivor_two_balls import pong_survivor_two_balls

    surrogate_policy_training(lunar_lander)
