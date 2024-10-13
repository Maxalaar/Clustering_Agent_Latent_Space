from ray.tune.registry import register_env

from environments.ant.ant import Ant
from environments.cart_pole.cart_pole import CartPole
from environments.lunar_lander.lunar_lander import LunarLander
from environments.bipedal_walker.bipedal_walker import BipedalWalker
from environments.pong_survivor.pong_survivor import PongSurvivor


def register_environments():
    register_env(name='CartPoleRllib', env_creator=CartPole)
    register_env(name='LunarLanderRllib', env_creator=LunarLander)
    register_env(name='BipedalWalkerRllib', env_creator=BipedalWalker)
    register_env(name='AntRllib', env_creator=Ant)
    register_env(name='PongSurvivor', env_creator=PongSurvivor)

