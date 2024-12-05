from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from torch.utils.data import Dataset


class ClassicDataset(Dataset):
    def __init__(
            self,
            file_path: Path,
            input_dataset_name: str,
            output_dataset_name: Optional[str] = None,
    ):
        self.file_path: Path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')

        self.input_dataset = np.array(self.h5_file[input_dataset_name])
        self.output_dataset = None
        if output_dataset_name is not None:
            self.output_dataset = np.array(self.h5_file[output_dataset_name])

        if self.output_dataset is not None and len(self.input_dataset) != len(self.output_dataset):
            raise ValueError(f"Error: input_dataset length ({len(self.input_dataset)}) does not match output_dataset length ({len(self.output_dataset)}).")
        self.dataset_size = len(self.input_dataset)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.dataset_size)
        if self.output_dataset is not None:
            return self.input_dataset[idx], self.output_dataset[idx]
        else:
            return self.input_dataset[idx]
