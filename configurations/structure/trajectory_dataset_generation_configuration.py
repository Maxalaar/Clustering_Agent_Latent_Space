from ray.rllib.utils.from_config import NotProvided

from utilities.image_compression import image_compression


class TrajectoryDatasetGenerationConfiguration:
    def __init__(self):
        self.save_rendering: bool = False
        self.number_iterations: int = 5
        self.minimal_steps_per_iteration_per_environment_runners: int = 10_000
        self.number_environment_runners: int = 1
        self.number_environment_per_environment_runners: int = 1
        self.number_cpus_per_environment_runners: int = NotProvided
        self.number_gpus_per_environment_runners: int = NotProvided
        self.image_compression_function = image_compression
        self.image_compression_configuration = {
            'new_width': 300,
            'new_height': 300,
            'grayscale': True,
        }
