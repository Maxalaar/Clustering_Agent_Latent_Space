"""
Port of Neural Slime Volleyball to Python Gymnasium Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gymnasium
"""

import logging
import math
from typing import Optional, Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
import pygame
from pygame.locals import *
from collections import deque

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

class SlimeVolley(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = {}):
        # Game settings
        self.REF_W = 24 * 2
        self.REF_H = self.REF_W
        self.REF_U = 1.5  # ground height
        self.REF_WALL_WIDTH = 1.0  # wall width
        self.REF_WALL_HEIGHT = 3.5
        self.PLAYER_SPEED_X = 10 * 1.75
        self.PLAYER_SPEED_Y = 10 * 1.35
        self.MAX_BALL_SPEED = 15 * 1.5
        self.TIMESTEP = environment_configuration.get('time_step', 1 / 30.)
        self.NUDGE = 0.1
        self.FRICTION = 1.0  # 1 means no FRICTION, less means FRICTION
        self.INIT_DELAY_FRAMES = 30
        self.GRAVITY = -9.8 * 2 * 1.5
        self.MAXLIVES = 5  # game ends when one agent loses this many games
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 500
        self.FACTOR = self.WINDOW_WIDTH / self.REF_W

        # Colors
        self.BALL_COLOR = (217, 79, 0)
        self.AGENT_LEFT_COLOR = (35, 93, 188)
        self.AGENT_RIGHT_COLOR = (255, 236, 0)
        self.BACKGROUND_COLOR = (11, 16, 19)
        self.FENCE_COLOR = (102, 56, 35)
        self.COIN_COLOR = self.FENCE_COLOR
        self.GROUND_COLOR = (116, 114, 117)

        self.metadata = {
            'render_modes': ['human', 'rgb_array'],
            'render_fps': 50,
        }
        self.render_mode = environment_configuration.get('render_mode', 'rgb_array')

        self.t = 0
        self.t_limit = 4000

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # self.action_space = spaces.MultiDiscrete(np.array([2, 2, 2]))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.game = Game(self)
        self.viewer = None
        self.screen = None
        self.policy = BaselinePolicy()
        self.otherAction = None

        self.observation_labels = [
            # Self
            'self_position_x',
            'self_position_y',
            'self_velocity_x',
            'self_velocity_y',
            # Ball
            'ball_position_x',
            'ball_position_y',
            'ball_velocity_x',
            'ball_velocity_y',
            # Other
            'other_position_x',
            'other_position_y',
            'other_velocity_x',
            'other_velocity_y',
        ]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game = Game(self, np_random=self.np_random)
        return [seed]

    def getObs(self):
        return self.game.agent_left.getObservation()

    def step(self, action, otherAction=None):
        done = False
        self.t += 1

        if self.otherAction is not None:
            otherAction = self.otherAction

        if otherAction is None:
            obs = self.game.agent_right.getObservation()
            otherAction = self.policy.predict(obs)

        self.game.agent_left.setAction(action)
        self.game.agent_right.setAction(otherAction)

        reward = self.game.step()

        obs = self.getObs()

        if self.t >= self.t_limit:
            done = True

        if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
            done = True

        info = {
            'left_agent_lives': self.game.agent_left.lives(),
            'right_agent_lives': self.game.agent_right.lives(),
            'left_agent_observation': self.game.agent_left.getObservation(),
            'right_agent_observation': self.game.agent_right.getObservation(),
        }

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.t = 0
        self.game.reset()
        return self.getObs(), {}

    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                pygame.display.set_caption('Slime Volleyball')
            surface = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.game.display(surface)
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        elif self.render_mode == 'rgb_array':
            surface = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.game.display(surface)
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

class DelayScreen:
    """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """

    def __init__(self, env, life=None):
        self.env = env
        self.life = life if life is not None else env.INIT_DELAY_FRAMES

    def reset(self, life=None):
        self.life = life if life is not None else self.env.INIT_DELAY_FRAMES

    def status(self):
        if self.life == 0:
            return True
        self.life -= 1
        return False

def toX(env, x):
    return (x + env.REF_W / 2) * env.FACTOR

def toP(env, x):
    return x * env.FACTOR

def toY(env, y):
    return y * env.FACTOR

def create_canvas(env, canvas, c):
    canvas.fill(c)
    return canvas

def rect(env, canvas, x, y, width, height, color):
    if height < 0:
        pygame_height = -height
        pygame_y = env.WINDOW_HEIGHT - y
    else:
        pygame_height = height
        pygame_y = env.WINDOW_HEIGHT - y - height
    pygame.draw.rect(canvas, color, (int(x), int(pygame_y), int(width), int(pygame_height)))

def half_circle(env, canvas, x, y, r, color):
    res = 20
    points = []
    for i in range(res + 1):
        ang = math.pi - math.pi * i / res
        px = x + math.cos(ang) * r
        py = y + math.sin(ang) * r
        points.append((int(px), int(env.WINDOW_HEIGHT - py)))
    pygame.draw.polygon(canvas, color, points)

def circle(env, canvas, x, y, r, color):
    pygame.draw.circle(canvas, color, (int(x), int(env.WINDOW_HEIGHT - y)), int(r))

class Particle:
    def __init__(self, env, x, y, vx, vy, r, c):
        self.env = env
        self.x = x
        self.y = y
        self.prev_x = self.x
        self.prev_y = self.y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.c = c

    def display(self, canvas):
        circle(self.env, canvas, toX(self.env, self.x), toY(self.env, self.y), toP(self.env, self.r), self.c)
        return canvas

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.vx * self.env.TIMESTEP
        self.y += self.vy * self.env.TIMESTEP

    def applyAcceleration(self, ax, ay):
        self.vx += ax * self.env.TIMESTEP
        self.vy += ay * self.env.TIMESTEP

    def checkEdges(self):
        if self.x <= (self.r - self.env.REF_W / 2):
            self.vx *= -self.env.FRICTION
            self.x = self.r - self.env.REF_W / 2 + self.env.NUDGE * self.env.TIMESTEP

        if self.x >= (self.env.REF_W / 2 - self.r):
            self.vx *= -self.env.FRICTION
            self.x = self.env.REF_W / 2 - self.r - self.env.NUDGE * self.env.TIMESTEP

        if self.y <= (self.r + self.env.REF_U):
            self.vy *= -self.env.FRICTION
            self.y = self.r + self.env.REF_U + self.env.NUDGE * self.env.TIMESTEP
            if self.x <= 0:
                return -1
            else:
                return 1
        if self.y >= (self.env.REF_H - self.r):
            self.vy *= -self.env.FRICTION
            self.y = self.env.REF_H - self.r - self.env.NUDGE * self.env.TIMESTEP
        # fence:
        if (self.x <= (self.env.REF_WALL_WIDTH / 2 + self.r)) and (self.prev_x > (self.env.REF_WALL_WIDTH / 2 + self.r)) and (self.y <= self.env.REF_WALL_HEIGHT):
            self.vx *= -self.env.FRICTION
            self.x = self.env.REF_WALL_WIDTH / 2 + self.r + self.env.NUDGE * self.env.TIMESTEP

        if (self.x >= (-self.env.REF_WALL_WIDTH / 2 - self.r)) and (self.prev_x < (-self.env.REF_WALL_WIDTH / 2 - self.r)) and (self.y <= self.env.REF_WALL_HEIGHT):
            self.vx *= -self.env.FRICTION
            self.x = -self.env.REF_WALL_WIDTH / 2 - self.r - self.env.NUDGE * self.env.TIMESTEP
        return 0

    def getDist2(self, p):
        dy = p.y - self.y
        dx = p.x - self.x
        return dx * dx + dy * dy

    def isColliding(self, p):
        r = self.r + p.r
        return r * r > self.getDist2(p)

    def bounce(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abd = math.sqrt(abx * abx + aby * aby)
        abx /= abd
        aby /= abd
        nx = abx
        ny = aby
        abx *= self.env.NUDGE
        aby *= self.env.NUDGE
        while self.isColliding(p):
            self.x += abx
            self.y += aby
        ux = self.vx - p.vx
        uy = self.vy - p.vy
        un = ux * nx + uy * ny
        unx = nx * (un * 2.)
        uny = ny * (un * 2.)
        ux -= unx
        uy -= uny
        self.vx = ux + p.vx
        self.vy = uy + p.vy

    def limitSpeed(self, minSpeed, maxSpeed):
        mag2 = self.vx * self.vx + self.vy * self.vy
        if mag2 > (maxSpeed * maxSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed
        if mag2 < (minSpeed * minSpeed):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= minSpeed
            self.vy *= minSpeed

class Wall:
    def __init__(self, env, x, y, w, h, c):
        self.env = env
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def display(self, canvas):
        rect(self.env, canvas, toX(self.env, self.x - self.w / 2), toY(self.env, self.y - self.h / 2), toP(self.env, self.w), toP(self.env, self.h), self.c)
        return canvas

class RelativeState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.bx = 0
        self.by = 0
        self.bvx = 0
        self.bvy = 0
        self.ox = 0
        self.oy = 0
        self.ovx = 0
        self.ovy = 0

    def getObservation(self):
        result = [self.x, self.y, self.vx, self.vy,
                  self.bx, self.by, self.bvx, self.bvy,
                  self.ox, self.oy, self.ovx, self.ovy]
        scaleFactor = 10.0
        result = np.array(result) / scaleFactor

        # result = np.array([1.2000, 0.1500, 0.0000, 0.0000, 0.0000, 1.2000, -0.8870, 1.8969, 1.2000, 0.1500, 0.0000, 0.0000])
        # result = np.array([ 2.25 ,  0.446,  0.   , -0.414, -0.739,  1.884, -1.109,  0.096,  1.2  ,  0.15 ,  0.   ,  0.   ])
        return result

class Agent:
    def __init__(self, env, dir, x, y, c):
        self.env = env
        self.dir = dir
        self.x = x
        self.y = y
        self.r = 1.5
        self.c = c
        self.vx = 0
        self.vy = 0
        self.desired_vx = 0
        self.desired_vy = 0
        self.state = RelativeState()
        self.emotion = "happy"
        self.life = env.MAXLIVES

    def lives(self):
        return self.life

    def setAction(self, action):
        forward = action[0] > 0
        backward = action[1] > 0
        jump = action[2] > 0
        self.desired_vx = 0
        self.desired_vy = 0
        if forward and not backward:
            self.desired_vx = -self.env.PLAYER_SPEED_X
        if backward and not forward:
            self.desired_vx = self.env.PLAYER_SPEED_X
        if jump:
            self.desired_vy = self.env.PLAYER_SPEED_Y

    def move(self):
        self.x += self.vx * self.env.TIMESTEP
        self.y += self.vy * self.env.TIMESTEP

    def step(self):
        self.x += self.vx * self.env.TIMESTEP
        self.y += self.vy * self.env.TIMESTEP

    def update(self):
        self.vy += self.env.GRAVITY * self.env.TIMESTEP

        if self.y <= self.env.REF_U + self.env.NUDGE * self.env.TIMESTEP:
            self.vy = self.desired_vy

        self.vx = self.desired_vx * self.dir

        self.move()

        if self.y <= self.env.REF_U:
            self.y = self.env.REF_U
            self.vy = 0

        if self.x * self.dir <= (self.env.REF_WALL_WIDTH / 2 + self.r):
            self.vx = 0
            self.x = self.dir * (self.env.REF_WALL_WIDTH / 2 + self.r)

        if self.x * self.dir >= (self.env.REF_W / 2 - self.r):
            self.vx = 0
            self.x = self.dir * (self.env.REF_W / 2 - self.r)

    def updateState(self, ball, opponent):
        self.state.x = self.x * self.dir
        self.state.y = self.y
        self.state.vx = self.vx * self.dir
        self.state.vy = self.vy
        self.state.bx = ball.x * self.dir
        self.state.by = ball.y
        self.state.bvx = ball.vx * self.dir
        self.state.bvy = ball.vy
        self.state.ox = opponent.x * (-self.dir)
        self.state.oy = opponent.y
        self.state.ovx = opponent.vx * (-self.dir)
        self.state.ovy = opponent.vy

    def getObservation(self):
        return self.state.getObservation()

    def display(self, canvas, bx, by):
        x = toX(self.env, self.x)
        y = toY(self.env, self.y)
        r = toP(self.env, self.r)

        angle = math.pi * 60 / 180 if self.dir == -1 else math.pi * 120 / 180
        eyeX, eyeY = 0, 0

        half_circle(self.env, canvas, x, y, r, self.c)

        c = math.cos(angle)
        s = math.sin(angle)
        ballX = toX(self.env, bx) - (x + 0.6 * r * c)
        ballY = toY(self.env, by) - (y + 0.6 * r * s)

        dist = math.sqrt(ballX ** 2 + ballY ** 2)
        if dist > 0:
            eyeX = ballX / dist
            eyeY = ballY / dist

        eye_pos_x = x + 0.6 * r * c + eyeX * 0.15 * r
        eye_pos_y = y + 0.6 * r * s + eyeY * 0.15 * r
        circle(self.env, canvas, eye_pos_x, eye_pos_y, r * 0.3, (255, 255, 255))
        circle(self.env, canvas, eye_pos_x + eyeX * 0.15 * r, eye_pos_y + eyeY * 0.15 * r, r * 0.1, (0, 0, 0))

        for i in range(1, self.life):
            coin_x = toX(self.env, self.dir * (self.env.REF_W / 2 + 0.5 - i * 2.0))
            coin_y = toY(self.env, 1.5)
            circle(self.env, canvas, coin_x, self.env.WINDOW_HEIGHT - coin_y, toP(self.env, 0.5), self.env.COIN_COLOR)

        return canvas

class BaselinePolicy:
    def __init__(self):
        self.nGameInput = 8
        self.nGameOutput = 3
        self.nRecurrentState = 4
        self.nOutput = self.nGameOutput + self.nRecurrentState
        self.nInput = self.nGameInput + self.nOutput
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)
        self.weight = np.array([7.5719, 4.4285, 2.2716, -0.3598, -7.8189, -2.5422, -3.2034, 0.3935, 1.2202, -0.49, -0.0316, 0.5221, 0.7026, 0.4179, -2.1689, 1.646, -13.3639, 1.5151, 1.1175, -5.3561, 5.0442, 0.8451, 0.3987, -2.9501, -3.7811, -5.8994, 6.4167, 2.5014, 7.338, -2.9887, 2.4586, 13.4191, 2.7395, -3.9708, 1.6548, -2.7554, -1.5345, -6.4708, 9.2426, -0.7392, 0.4452, 1.8828, -2.6277, -10.851, -3.2353, -4.4653, -3.1153, -1.3707, 7.318, 16.0902, 1.4686, 7.0391, 1.7765, -1.155, 2.6697, -8.8877, 1.1958, -3.2839, -5.4425, 1.6809, 7.6812, -2.4732, 1.738, 0.3781, 0.8718, 2.5886, 1.6911, 1.2953, -9.0052, -4.6038, -6.7447, -2.5528, 0.4391, -4.9278, -3.6695, -4.8673, -1.6035, 1.5011, -5.6124, 4.9747, 1.8998, 3.0359, 6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459, 4.7134, 2.8952, -9.3627, -4.685, 0.3601, -1.3699, 9.7294, 11.5596, 0.1918, 3.0783, 0.0329, -0.1362, -0.1188, -0.7579, 0.3278, -0.977, -0.9377])
        self.bias = np.array([2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685])
        self.weight = self.weight.reshape(self.nOutput, self.nInput)

    def reset(self):
        self.inputState = np.zeros(self.nInput)
        self.outputState = np.zeros(self.nOutput)
        self.prevOutputState = np.zeros(self.nOutput)

    def _forward(self):
        self.prevOutputState = self.outputState
        self.outputState = np.tanh(np.dot(self.weight, self.inputState) + self.bias)

    def _setInputState(self, obs):
        self.inputState[0:self.nGameInput] = np.array(obs[:8])
        self.inputState[self.nGameInput:] = self.outputState

    def _getAction(self):
        forward = 1 if self.outputState[0] > 0.75 else 0
        backward = 1 if self.outputState[1] > 0.75 else 0
        jump = 1 if self.outputState[2] > 0.75 else 0
        return [forward, backward, jump]

    def predict(self, obs):
        self._setInputState(obs)
        self._forward()
        return self._getAction()

class Game:
    def __init__(self, env, np_random=np.random):
        self.env = env
        self.ball = None
        self.ground = None
        self.fence = None
        self.fenceStub = None
        self.agent_left: Optional[Agent] = None
        self.agent_right: Optional[Agent] = None
        self.delayScreen = None
        self.np_random = np_random
        self.reset()

    def reset(self):
        self.ground = Wall(self.env, 0, 0.75, self.env.REF_W, self.env.REF_U, self.env.GROUND_COLOR)
        self.fence = Wall(self.env, 0, 0.75 + self.env.REF_WALL_HEIGHT / 2, self.env.REF_WALL_WIDTH, self.env.REF_WALL_HEIGHT - 1.5, self.env.FENCE_COLOR)
        self.fenceStub = Particle(self.env, 0, self.env.REF_WALL_HEIGHT, 0, 0, self.env.REF_WALL_WIDTH / 2, self.env.FENCE_COLOR)
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(self.env, 0, self.env.REF_W / 4, ball_vx, ball_vy, 0.5, self.env.BALL_COLOR)
        self.agent_left = Agent(self.env, -1, -self.env.REF_W / 4, 1.5, self.env.AGENT_LEFT_COLOR)
        self.agent_right = Agent(self.env, 1, self.env.REF_W / 4, 1.5, self.env.AGENT_RIGHT_COLOR)
        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)
        self.delayScreen = DelayScreen(self.env)

    def newMatch(self):
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(self.env, 0, self.env.REF_W / 4, ball_vx, ball_vy, 0.5, self.env.BALL_COLOR)
        self.delayScreen.reset()

    def step(self):
        self.betweenGameControl()
        self.agent_left.update()
        self.agent_right.update()

        if self.delayScreen.status():
            self.ball.applyAcceleration(0, self.env.GRAVITY)
            self.ball.limitSpeed(0, self.env.MAX_BALL_SPEED)
            self.ball.move()

        if self.ball.isColliding(self.agent_left):
            self.ball.bounce(self.agent_left)
        if self.ball.isColliding(self.agent_right):
            self.ball.bounce(self.agent_right)
        if self.ball.isColliding(self.fenceStub):
            self.ball.bounce(self.fenceStub)

        result = -self.ball.checkEdges()
        reward = 0
        reward += 0.5 / self.env.t_limit

        if result != 0:
            self.newMatch()
            if result < 0:
                self.agent_left.emotion = "happy"
                self.agent_right.emotion = "sad"
                self.agent_right.life -= 1
                reward += +1
            else:
                self.agent_left.emotion = "sad"
                self.agent_right.emotion = "happy"
                self.agent_left.life -= 1
                reward += -1
        else:
            self.agent_left.updateState(self.ball, self.agent_right)
            self.agent_right.updateState(self.ball, self.agent_left)

        return reward

    def display(self, canvas):
        canvas = create_canvas(self.env, canvas, self.env.BACKGROUND_COLOR)
        self.fence.display(canvas)
        self.fenceStub.display(canvas)
        self.agent_left.display(canvas, self.ball.x, self.ball.y)
        self.agent_right.display(canvas, self.ball.x, self.ball.y)
        self.ball.display(canvas)
        self.ground.display(canvas)
        return canvas

    def betweenGameControl(self):
        if self.delayScreen.life > 0:
            pass