import pathlib

import numpy as np
import pygame
from ray.rllib import Policy
from ray.rllib.algorithms import PPO, AlgorithmConfig
import ray
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core.rl_module import RLModule
from ray.rllib.algorithms.algorithm import Algorithm
import torch
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

from environments.register_environments import register_environments
from environments.slime_volley.slime_volley import BaselinePolicy, SlimeVolley
from rllib_repertory.find_best_checkpoint_path import find_best_reinforcement_learning_checkpoint_path
from rllib_repertory.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration

if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    FPS = 50
    environment = SlimeVolley({'time_step': 1 / 30})

    # Initialisation des manettes
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        joystick.init()

    left_agent_manual = False
    right_agent_manual = False

    left_agent_action = [0, 0, 0]
    right_agent_action = [0, 0, 0]

    left_policy = BaselinePolicy()
    right_policy = BaselinePolicy()

    ray.init(local_mode=True)
    register_environments()

    checkpoint_path: pathlib.Path = find_best_reinforcement_learning_checkpoint_path(pathlib.Path('/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/slime_volley/reinforcement_learning/debug'))
    # checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/slime_volley/reinforcement_learning/debug/PPO_SlimeVolleyRllib_96e3c_00000_0_2025-01-31_11-31-54/checkpoint_000015'
    # checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/slime_volley/reinforcement_learning/base/PPO_SlimeVolleyRllib_af080_00000_0_2025-01-30_16-34-24/checkpoint_000067'

    algorithm_configuration = get_checkpoint_algorithm_configuration(checkpoint_path)
    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
    )
    algorithm_configuration.env_runners(
        num_env_runners=1,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0,
    )

    algorithm: Algorithm = algorithm_configuration.build()
    algorithm.restore(str(checkpoint_path))
    module_to_env_connector = algorithm.config.build_module_to_env_connector(environment)
    module_to_env_connector.connectors.pop()
    rl_module = algorithm.get_module()

    environment.render_mode = 'human'
    obs, _ = environment.reset()
    done = False

    while not done:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()

        # Contrôle via manette si disponible
        if left_agent_manual and len(joysticks) > 0:
            joystick = joysticks[0]

            # num_buttons = joystick.get_numbuttons()
            # for i in range(num_buttons):
            #     if joystick.get_button(i):
            #         print(i)

            left_agent_action[0] = 1 if (joystick.get_axis(0) > 0.05 or joystick.get_button(7) or joystick.get_button(9)) else 0  # Droite
            left_agent_action[1] = 1 if (joystick.get_axis(0) < -0.05 or joystick.get_button(6) or joystick.get_button(8)) else 0  # Gauche
            left_agent_action[2] = 1 if (joystick.get_button(0) or joystick.get_button(1) or joystick.get_button(3) or joystick.get_button(4))else 0  # Saut (Bouton 0)
        else:
            left_agent_action[0] = 1 if keys[pygame.K_d] else 0
            left_agent_action[1] = 1 if keys[pygame.K_q] else 0
            left_agent_action[2] = 1 if keys[pygame.K_SPACE] else 0

        if right_agent_manual and len(joysticks) > 1:
            joystick = joysticks[1]


            right_agent_action[1] = 1 if (joystick.get_axis(0) > 0.05 or joystick.get_button(7) or joystick.get_button(9)) else 0
            right_agent_action[0] = 1 if (joystick.get_axis(0) < -0.05 or joystick.get_button(6) or joystick.get_button(8)) else 0
            right_agent_action[2] = 1 if (joystick.get_button(0) or joystick.get_button(1) or joystick.get_button(3) or joystick.get_button(4)) else 0  # Saut (Bouton 0)

        else:
            right_agent_action[0] = 1 if keys[pygame.K_LEFT] else 0
            right_agent_action[1] = 1 if keys[pygame.K_RIGHT] else 0
            right_agent_action[2] = 1 if keys[pygame.K_KP0] else 0

        if not left_agent_manual:
            # left_agent_action = right_policy.predict(environment.game.agent_left.getObservation())

            obs = environment.getObs()
            fwd_ins = {"obs": torch.Tensor([obs])}
            fwd_outputs = rl_module.forward_inference(fwd_ins)
            left_agent_action = module_to_env_connector(rl_module=rl_module, batch=fwd_outputs, episodes=[None])['actions_for_env'][0]

        if not right_agent_manual:
            right_agent_action = right_policy.predict(environment.game.agent_right.getObservation())

        environment.step(action=left_agent_action, otherAction=right_agent_action)
        environment.render()

    environment.close()
    pygame.joystick.quit()
