from ray.tune.registry import register_env

from environments.ant.ant import Ant
from environments.car_racing.car_racing import CarRacing
from environments.cart_pole.cart_pole import CartPole
from environments.craftium.craftium import Craftium
from environments.flappy_bird.flappy_bird import FlappyBird
from environments.lunar_lander.lunar_lander import LunarLander
from environments.bipedal_walker.bipedal_walker import BipedalWalker
from environments.pong_survivor.pong_survivor import PongSurvivor
from environments.slime_volley.slime_volley import SlimeVolley
from environments.taxi.taxi import Taxi
from environments.tetris.tetris import Tetris
from environments.tetris_atari.tetris_atari import TetrisAtari


def register_environments():
    register_env(name='CartPoleRllib', env_creator=CartPole)
    register_env(name='LunarLanderRllib', env_creator=LunarLander)
    register_env(name='FlappyBirdRllib', env_creator=FlappyBird)
    register_env(name='BipedalWalkerRllib', env_creator=BipedalWalker)
    register_env(name='AntRllib', env_creator=Ant)
    register_env(name='TaxiRllib', env_creator=Taxi)
    register_env(name='PongSurvivor', env_creator=PongSurvivor)
    register_env(name='TetrisRllib', env_creator=Tetris)
    register_env(name='TetrisAtariRllib', env_creator=TetrisAtari)
    register_env(name='SlimeVolleyRllib', env_creator=SlimeVolley)
    register_env(name='CarRacingRllib', env_creator=CarRacing)
    register_env(name='CraftiumRllib', env_creator=Craftium)
