from time import sleep
from typing import Optional

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from models.architectures.pytorch_lightning.dense import Dense


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
    def __init__(self, h5_file_path: Path, input_dataset_name: str, output_dataset_name: Optional[str] = None):
        super().__init__()
        print('New H5Dataset')
        self.h5_file_path: Path = h5_file_path
        self.input_dataset_name: str = input_dataset_name
        self.output_dataset_name: Optional[str] = output_dataset_name

        # with h5py.File(self.h5_file_path, 'r') as h5_file:
        #     self.data_len = h5_file[self.input_dataset_name].shape[0]

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            self.data_len = h5_file[self.input_dataset_name].shape[0]
            self.input_data = np.array(h5_file[self.input_dataset_name])
            self.output_data = np.array(h5_file[self.output_dataset_name])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # with h5py.File(self.h5_file_path, 'r') as h5_file:
        #     input_data = h5_file[self.input_dataset_name][idx]
        #
        #     if self.output_dataset_name is not None:
        #         output_data = h5_file[self.output_dataset_name][idx]
        #         return input_data, output_data
        #
        #     return input_data
        return self.input_data[idx], self.output_data[idx]

# class H5Dataset(Dataset):
#     def __init__(self, h5_file_path: Path, input_dataset_name: str, output_dataset_name: Optional[str] = None):
#         super().__init__()
#         self.h5_file_path: Path = h5_file_path
#         self.input_dataset_name: str = input_dataset_name
#         self.output_dataset_name: Optional[str] = output_dataset_name
#
#         # Ouverture du fichier HDF5
#         self.h5_file = h5py.File(self.h5_file_path, 'r')
#         self.data_len = self.h5_file[self.input_dataset_name].shape[0]
#
#     def __len__(self):
#         return self.data_len
#
#     def __getitem__(self, idx):
#         # Accéder aux données directement à partir du fichier HDF5
#         input_data = self.h5_file[self.input_dataset_name][idx]
#
#         if self.output_dataset_name is not None:
#             output_data = self.h5_file[self.output_dataset_name][idx]
#             return input_data, output_data
#
#         return input_data
#
#     def __del__(self):
#         # Fermer le fichier HDF5 lorsque l'objet est détruit
#         self.h5_file.close()


class H5DataModule(pl.LightningDataModule):
    def __init__(self, h5_file_path: Path, input_dataset_name: str, output_dataset_name: Optional[str] = None, batch_size: int = 64, validation_split: float = 0.2, number_workers: int = 1):
        super().__init__()
        self.h5_file_path: Path = h5_file_path
        self.input_dataset_name: str = input_dataset_name
        self.output_dataset_name: Optional[str] = output_dataset_name
        self.batch_size: int = batch_size
        self.validation_split = validation_split
        self.number_workers: int = number_workers

        self.input_shape: Optional[tuple] = None
        self.output_shape: Optional[tuple] = None
        self.train_dataset: Optional[H5Dataset] = None
        self.test_dataset: Optional[H5Dataset] = None

    def setup(self, stage=None):
        self.input_shape = get_h5_shapes(self.h5_file_path, self.input_dataset_name)
        self.train_dataset = H5Dataset(
            h5_file_path=self.h5_file_path,
            input_dataset_name=self.input_dataset_name,
            output_dataset_name=self.output_dataset_name,
        )

        if self.output_dataset_name is not None :
            self.output_shape = get_h5_shapes(self.h5_file_path, self.output_dataset_name)
            self.test_dataset = H5Dataset(
                h5_file_path=self.h5_file_path,
                input_dataset_name=self.input_dataset_name,
                output_dataset_name=self.output_dataset_name,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.number_workers)

    def validation_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.number_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.number_workers)


def surrogate_policy_training(experimentation_configuration: ExperimentationConfiguration):
    data_module = H5DataModule(
        h5_file_path=experimentation_configuration.trajectory_dataset_file_path,
        input_dataset_name='observation',
        output_dataset_name='action',
        batch_size=100_000,
        number_workers=5,
    )
    data_module.setup()

    architecture = Dense(
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
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=architecture,
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
