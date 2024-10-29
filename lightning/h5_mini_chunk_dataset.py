from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5MiniChunkDataset(Dataset):
    def __init__(
            self, file_path: Path,
            mini_chunk_size: int, input_dataset_name: str,
            number_mini_chunk: int = 2,
            output_dataset_name: Optional[str] = None,
    ):
        self.file_path: Path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')
        self.mini_chunk_size: int = mini_chunk_size
        self.number_mini_chunk: int = number_mini_chunk

        self.input_dataset = self.h5_file[input_dataset_name]
        self.output_dataset = None
        if output_dataset_name is not None:
            self.output_dataset = self.h5_file[output_dataset_name]

        if self.output_dataset is not None and len(self.input_dataset) != len(self.output_dataset):
            raise ValueError(f"Error: input_dataset length ({len(self.input_dataset)}) does not match output_dataset length ({len(self.output_dataset)}).")
        self.dataset_size = len(self.input_dataset)

        self.number_call_current_chunk: Optional[int] = None
        self.number_data_in_chunk: Optional[int] = None
        self.input_chunk: Optional[torch.Tensor] = None
        self.output_chunk: Optional[torch.Tensor] = None

    def load_mini_chunks(self):
        input_mini_chunks = []
        if self.output_dataset is not None:
            output_mini_chunks = []

        for i in range(self.number_mini_chunk):
            idx_mini_chunk_start = np.random.randint(0, self.dataset_size)
            idx_mini_chunk_stop = min(idx_mini_chunk_start + self.mini_chunk_size, self.dataset_size)

            input_mini_chunks.append(torch.tensor(self.input_dataset[idx_mini_chunk_start:idx_mini_chunk_stop]))
            if self.output_dataset is not None:
                output_mini_chunks.append(torch.tensor(self.output_dataset[idx_mini_chunk_start:idx_mini_chunk_stop]))

        self.input_chunk = torch.cat(input_mini_chunks, dim=0)
        if self.output_dataset is not None:
            self.output_chunk = torch.cat(output_mini_chunks, dim=0)

        self.number_data_in_chunk = self.input_chunk.size(0)
        self.number_call_current_chunk = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.input_chunk is None or self.number_call_current_chunk >= self.number_data_in_chunk:
            self.load_mini_chunks()

        self.number_call_current_chunk += 1

        local_idx = np.random.randint(0, self.number_data_in_chunk)

        if self.output_dataset is not None:
            return self.input_chunk[local_idx], self.output_chunk[local_idx]
        else:
            return self.input_chunk[local_idx]
