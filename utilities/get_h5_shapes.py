from pathlib import Path
import h5py
import numpy as np


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
