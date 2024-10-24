from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.pong_survivor.configurations import classic_one_ball


pong_survivor_one_ball = ExperimentationConfiguration(
    experimentation_name='pong_survivor_one_ball',
    environment_name='PongSurvivor',
)
pong_survivor_one_ball.environment_configuration = classic_one_ball
