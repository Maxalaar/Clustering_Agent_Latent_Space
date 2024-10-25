from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(
            self, file_path: Path,
            chunk_size: int, input_dataset_name: str,
            output_dataset_name: Optional[str] = None,
            # shuffle_in_chunk: bool = True
    ):
        self.file_path: Path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')
        self.chunk_size: int = chunk_size
        # self.shuffle_in_chunk: bool = shuffle_in_chunk

        self.input_dataset = self.h5_file[input_dataset_name]
        self.output_dataset = None
        if output_dataset_name is not None:
            self.output_dataset = self.h5_file[output_dataset_name]

        self.idx_chunk_start = None
        self.idx_chunk_stop = None

        if self.output_dataset is not None and len(self.input_dataset) != len(self.output_dataset):
            raise ValueError(f"Error: input_dataset length ({len(self.input_dataset)}) does not match output_dataset length ({len(self.output_dataset)}).")
        self.dataset_size = len(self.input_dataset)

        self.input_chunk: Optional[torch.Tensor] = None
        self.output_chunk: Optional[torch.Tensor] = None

    def load_chunk(self, idx):
        self.idx_chunk_start = (idx // self.chunk_size) * self.chunk_size

        self.idx_chunk_stop = min(self.idx_chunk_start + self.chunk_size, self.dataset_size)

        self.input_chunk = torch.tensor(self.input_dataset[self.idx_chunk_start:self.idx_chunk_stop])
        if self.output_dataset is not None:
            self.output_chunk = torch.tensor(self.output_dataset[self.idx_chunk_start:self.idx_chunk_stop])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.idx_chunk_start is None:
            self.load_chunk(idx)

        idx_in_current_chunk = (self.idx_chunk_start <= idx) & (idx < self.idx_chunk_stop)
        if not idx_in_current_chunk:
            self.load_chunk(idx)

        local_idx = idx - self.idx_chunk_start
        if self.output_dataset is not None:
            return self.input_chunk[local_idx], self.output_chunk[local_idx]
        else:
            return self.input_chunk[local_idx]

        # if self.idx_chunk_start is None:
        #     self.load_chunk(idx)
        #
        # idx_in_current_chunk = (self.idx_chunk_start <= idx) & (idx < self.idx_chunk_stop)
        # if not np.all(idx_in_current_chunk):
        #     self.load_chunk(idx)
        #
        # local_idx = idx - self.idx_chunk_start
        # start, end = local_idx[0], local_idx[1]
        #
        # if not self.shuffle_in_chunk:
        #     if self.output_dataset is not None:
        #         return self.input_chunk[start:end], self.output_chunk[start:end]
        #     else:
        #         return self.input_chunk[start:end]
        # else:
        #     indices = torch.sort(torch.randint(0, self.input_chunk.size(0), (local_idx[1]-local_idx[0],))).values
        #     if self.output_dataset is not None:
        #         return self.input_chunk[indices], self.output_chunk[indices]
        #     else:
        #         return self.input_chunk[indices]
