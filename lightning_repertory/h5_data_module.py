from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lightning_repertory.h5_dataset import H5Dataset
from lightning_repertory.h5_mini_chunk_dataset import H5MiniChunkDataset
from utilities.get_h5_shapes import get_h5_shapes


class H5DataModule(pl.LightningDataModule):
    def __init__(
            self,
            h5_file_path: Path,
            input_dataset_name: str,
            output_dataset_name: Optional[str] = None,
            batch_size: int = 64,
            number_mini_chunks: int = 2,
            mini_chunk_size: int = 64,
            validation_split: float = 0.2,
            number_workers: int = 1,
            shuffle: bool = True,
    ):
        super().__init__()

        if mini_chunk_size < batch_size:
            raise ValueError(f"Error: mini_chunk_size ({mini_chunk_size}) cannot be smaller than batch_size ({batch_size}).")

        self.h5_file_path: Path = h5_file_path
        self.input_dataset_name: str = input_dataset_name
        self.output_dataset_name: Optional[str] = output_dataset_name
        self.batch_size: int = batch_size
        self.validation_split = validation_split
        self.number_workers: int = number_workers
        self.number_mini_chunks = number_mini_chunks
        self.mini_chunk_size: int = mini_chunk_size
        self.shuffle = shuffle

        self.input_shape: Optional[tuple] = None
        self.output_shape: Optional[tuple] = None
        self.dataset: Optional[H5Dataset] = None
        # self.train_dataset: Optional[H5Dataset] = None
        # self.validation_dataset: Optional[H5Dataset] = None

    def setup(self, stage=None):
        self.input_shape = get_h5_shapes(self.h5_file_path, self.input_dataset_name)
        if self.output_dataset_name is not None:
            self.output_shape = get_h5_shapes(self.h5_file_path, self.output_dataset_name)

        self.dataset = H5MiniChunkDataset(
            file_path=self.h5_file_path,
            input_dataset_name=self.input_dataset_name,
            output_dataset_name=self.output_dataset_name,
            mini_chunk_size=self.mini_chunk_size,
            number_mini_chunk=self.number_mini_chunks,
            shuffle=self.shuffle,
        )
        # dataset_size = len(self.dataset)
        # indices = np.arange(dataset_size)
        # split = int(np.floor(self.validation_split * dataset_size))
        # train_indices, val_indices = indices[split:], indices[:split]
        # self.train_dataset = Subset(self.dataset, train_indices)
        # self.validation_dataset = Subset(self.dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=self.number_workers,
            batch_size=self.batch_size,
        )

    def validation_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=self.number_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=self.number_workers,
            batch_size=self.batch_size,
        )
