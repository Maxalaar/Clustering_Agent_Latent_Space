from ray.rllib.utils.from_config import NotProvided


class VideoEpisodesGenerationConfiguration:
    def __init__(self):
        self.minimal_number_videos: int = 50
        self.number_environment_runners: int = 1
        self.number_environment_per_environment_runners: int = 1
        self.number_cpus_per_environment_runners: int = NotProvided
        self.number_gpus_per_environment_runners: int = NotProvided




