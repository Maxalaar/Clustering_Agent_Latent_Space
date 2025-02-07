from pathlib import Path
from typing import Optional, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5MiniChunkDataset(Dataset):
    def __init__(
            self,
            file_path: Path,
            mini_chunk_size: int,
            dataset_names: List[str],
            number_mini_chunk: int = 2,
            shuffle: bool = True,
    ):
        self.file_path: Path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')
        self.mini_chunk_size: int = mini_chunk_size
        self.number_mini_chunk: int = number_mini_chunk
        self.shuffle: bool = shuffle

        self.dataset_names = dataset_names
        self.datasets = [self.h5_file[name] for name in self.dataset_names]

        # Check all datasets have the same length
        first_len = len(self.datasets[0])
        for ds in self.datasets[1:]:
            if len(ds) != first_len:
                raise ValueError(
                    f"Dataset {ds.name} has length {len(ds)}, which does not match the first dataset length {first_len}."
                )
        self.dataset_size = first_len

        self.number_call_current_chunk: Optional[int] = None
        self.number_data_in_chunk: Optional[int] = None
        self.dataset_chunks: Optional[List[torch.Tensor]] = None

    def load_mini_chunks(self):
        # Initialize a list to hold mini chunks for each dataset
        mini_chunks = [[] for _ in range(len(self.datasets))]

        for _ in range(self.number_mini_chunk):
            idx_mini_chunk_start = np.random.randint(0, self.dataset_size)
            idx_mini_chunk_stop = min(idx_mini_chunk_start + self.mini_chunk_size, self.dataset_size)

            for i, dataset in enumerate(self.datasets):
                data = torch.tensor(dataset[idx_mini_chunk_start:idx_mini_chunk_stop])
                mini_chunks[i].append(data)

        # Concatenate mini chunks for each dataset
        self.dataset_chunks = [torch.cat(chunks, dim=0) for chunks in mini_chunks]
        self.number_data_in_chunk = self.dataset_chunks[0].size(0)
        self.number_call_current_chunk = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset_chunks is None or self.number_call_current_chunk >= self.number_data_in_chunk:
            self.load_mini_chunks()

        if self.shuffle:
            local_idx = np.random.randint(0, self.number_data_in_chunk)
        else:
            local_idx = self.number_call_current_chunk

        self.number_call_current_chunk += 1

        return tuple(chunk[local_idx] for chunk in self.dataset_chunks)
