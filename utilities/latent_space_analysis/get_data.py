from pathlib import Path
from typing import List
import torch

from lightning_repertory.h5_data_module import H5DataModule
from utilities.display_h5_file_information import display_h5_file_information


def get_data(
        dataset_names:List[str],
        data_number:int,
        trajectory_dataset_file_path: Path,
        device=torch.device('cpu'),
        batch_size=1000,
        number_mini_chunks=3,
        mini_chunk_size=1000,
        number_workers=2
):
    display_h5_file_information(trajectory_dataset_file_path)
    data_module = H5DataModule(
        h5_file_path=trajectory_dataset_file_path,
        dataset_names=dataset_names,
        batch_size=batch_size,
        number_mini_chunks=number_mini_chunks,
        mini_chunk_size=mini_chunk_size,
        number_workers=number_workers,
    )
    data_module.setup()
    data = data_module.load_data(data_number)

    for i in range(len(data)):
        data[i] = data[i].clone().detach().to(device)

    return data