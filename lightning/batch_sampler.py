from torch.utils.data import Sampler
import numpy as np


class BatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __iter__(self):
        return (np.array([i, i + self.batch_size - 1]) for i in range(0, self.dataset_size, self.batch_size))

    def __len__(self):
        return ((self.dataset_size + self.batch_size - 1) // self.batch_size) + 1
