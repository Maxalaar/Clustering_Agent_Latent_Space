from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from torch.utils.data import Dataset


def padding(chunk: np.ndarray, target_size: int):
    duplications = chunk.shape[0] // target_size
    remainder = chunk.shape[0] % target_size

    expanded_chunk = np.tile(chunk, (duplications, 1))
    if remainder > 0:
        expanded_chunk = np.vstack([expanded_chunk, chunk[:remainder]])

    return expanded_chunk


class H5Dataset(Dataset):
    def __init__(self, file_path: Path, chunk_size: int, input_dataset_name: str, output_dataset_name: Optional[str] = None):
        self.file_path: Path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')
        self.chunk_size: int = chunk_size

        self.input_dataset = self.h5_file[input_dataset_name]
        self.output_dataset = None
        if output_dataset_name is not None:
            self.output_dataset = self.h5_file[output_dataset_name]

        self.idx_chunk_start = None
        self.idx_chunk_stop = None

        if len(self.input_dataset) != len(self.output_dataset):
            raise ValueError(f"Error: input_dataset length ({len(self.input_dataset)}) does not match output_dataset length ({len(self.output_dataset)}).")
        self.dataset_size = len(self.input_dataset)

        self.input_chunk: Optional[np.ndarray] = None
        self.output_chunk: Optional[np.ndarray] = None

    def load_chunk(self, idx):
        self.idx_chunk_start = (idx[0] // self.chunk_size) * self.chunk_size

        need_to_padding = False
        if self.idx_chunk_start + self.chunk_size > self.dataset_size:
            need_to_padding = True

        self.idx_chunk_stop = min(self.idx_chunk_start + self.chunk_size, self.dataset_size)

        self.input_chunk = np.array(self.input_dataset[self.idx_chunk_start:self.idx_chunk_stop])
        if self.output_dataset is not None:
            self.output_chunk = np.array(self.output_dataset[self.idx_chunk_start:self.idx_chunk_stop])

        if need_to_padding:
            batch_size = idx[1] - idx[0] + 1
            self.input_chunk = padding(self.input_chunk, batch_size)
            if self.output_dataset is not None:
                self.output_chunk = padding(self.output_chunk, batch_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if self.idx_chunk_start is None:
            self.load_chunk(idx)

        idx_in_current_chunk = (self.idx_chunk_start <= idx) & (idx < self.idx_chunk_stop)
        if not np.all(idx_in_current_chunk):
            self.load_chunk(idx)

        local_idx = idx - self.idx_chunk_start
        if self.output_dataset is not None:
            return self.input_chunk[local_idx], self.output_chunk[local_idx]
        else:
            return self.input_chunk[local_idx]
