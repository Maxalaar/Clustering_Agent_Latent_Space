from ray.rllib.utils.from_config import NotProvided


class SurrogatePolicyEvaluationConfiguration:
    def __init__(self):
        self.number_environment_runners: int = 1
        self.evaluation_duration : int = 100
        self.number_cpus_per_environment_runners: int = NotProvided
        self.number_gpus_per_environment_runners: float = NotProvided