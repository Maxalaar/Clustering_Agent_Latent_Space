import os
import shutil
from pathlib import Path
from typing import List

from configurations.structure.experimentation_configuration import ExperimentationConfiguration


def create_latent_space_analysis_directories(surrogate_policy_checkpoint_paths: List[Path], experimentation_configuration: ExperimentationConfiguration) -> List[Path]:
    latent_space_analysis_storage_paths = []

    for surrogate_policy_checkpoint_path in surrogate_policy_checkpoint_paths:
        latent_space_analysis_storage_path = experimentation_configuration.latent_space_analysis_storage_path / surrogate_policy_checkpoint_path.parents[2].name / surrogate_policy_checkpoint_path.parents[1].name

        if latent_space_analysis_storage_path.exists() and latent_space_analysis_storage_path.is_dir():
            shutil.rmtree(latent_space_analysis_storage_path)

        os.makedirs(latent_space_analysis_storage_path, exist_ok=True)
        latent_space_analysis_storage_paths.append(latent_space_analysis_storage_path)

        # message = 'Surrogate policy checkpoint path: ' + str(surrogate_policy_checkpoint_path) + '\n'
        # print(message)
        #
        # with open(latent_space_analysis_storage_path / 'information.txt', 'a') as file:
        #     file.write(message)

    return latent_space_analysis_storage_paths