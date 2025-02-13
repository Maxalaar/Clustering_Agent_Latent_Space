from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from lightning_repertory.h5_mini_chunk_dataset import H5MiniChunkDataset


class H5DataModule(pl.LightningDataModule):
    def __init__(
            self,
            h5_file_path: Path,
            dataset_names: List[str],
            batch_size: int = 64,
            number_mini_chunks: int = 2,
            mini_chunk_size: int = 64,
            validation_split: float = 0.2,
            number_workers: int = 1,
            shuffle: bool = True,
    ):
        super().__init__()

        if mini_chunk_size < batch_size:
            raise ValueError(
                f"mini_chunk_size ({mini_chunk_size}) must be >= batch_size ({batch_size})."
            )
        if not dataset_names:
            raise ValueError("dataset_names cannot be empty.")

        self.h5_file_path = h5_file_path
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.number_workers = number_workers
        self.number_mini_chunks = number_mini_chunks
        self.mini_chunk_size = mini_chunk_size
        self.shuffle = shuffle

        self.input_shape: Optional[tuple] = None
        self.output_shape: Optional[tuple] = None
        self.full_dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.validation_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # Get shapes from the first and second datasets (if available)
        with h5py.File(self.h5_file_path, "r") as h5_file:
            self.input_shape = h5_file[self.dataset_names[0]].shape[1:]
            if len(self.dataset_names) > 1:
                self.output_shape = h5_file[self.dataset_names[1]].shape[1:]

        # Initialize the full dataset
        self.full_dataset = H5MiniChunkDataset(
            file_path=self.h5_file_path,
            dataset_names=self.dataset_names,
            mini_chunk_size=self.mini_chunk_size,
            number_mini_chunk=self.number_mini_chunks,
            shuffle=self.shuffle,
        )

        # Split into train and validation
        dataset_size = len(self.full_dataset)
        indices = np.arange(dataset_size)
        split_idx = int(np.floor(self.validation_split * dataset_size))

        if self.shuffle:
            np.random.shuffle(indices)

        train_indices = indices[split_idx:]
        val_indices = indices[:split_idx]

        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.validation_dataset = Subset(self.full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.number_workers,
            shuffle=False,  # Shuffle is handled by H5MiniChunkDataset
        )

    def validation_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.number_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.validation_dataset,  # Or use a separate test dataset if available
            batch_size=self.batch_size,
            num_workers=self.number_workers,
            shuffle=False,
        )

    def load_data(self, data_number: int):
        return self.full_dataset.load_data(data_number)