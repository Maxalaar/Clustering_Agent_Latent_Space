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

# game settings:
REF_W = 24 * 2
REF_H = REF_W
REF_U = 1.5  # ground height
REF_WALL_WIDTH = 1.0  # wall width
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 10 * 1.75
PLAYER_SPEED_Y = 10 * 1.35
MAX_BALL_SPEED = 15 * 1.5
TIMESTEP = 1 / 30.
NUDGE = 0.1
FRICTION = 1.0  # 1 means no FRICTION, less means FRICTION
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8 * 2 * 1.5

MAXLIVES = 5  # game ends when one agent loses this many games

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 500

FACTOR = WINDOW_WIDTH / REF_W

# Colors
BALL_COLOR = (217, 79, 0)
AGENT_LEFT_COLOR = (35, 93, 188)
AGENT_RIGHT_COLOR = (255, 236, 0)
BACKGROUND_COLOR = (11, 16, 19)
FENCE_COLOR = (102, 56, 35)
COIN_COLOR = FENCE_COLOR
GROUND_COLOR = (116, 114, 117)

class DelayScreen:
    """ initially the ball is held still for INIT_DELAY_FRAMES(30) frames """

    def __init__(self, life=INIT_DELAY_FRAMES):
        self.life = 0
        self.reset(life)

    def reset(self, life=INIT_DELAY_FRAMES):
        self.life = life

    def status(self):
        if (self.life == 0):
            return True
        self.life -= 1
        return False

def toX(x):
    return (x + REF_W / 2) * FACTOR

def toP(x):
    return (x) * FACTOR

def toY(y):
    return y * FACTOR

def create_canvas(canvas, c):
    canvas.fill(c)
    return canvas

def rect(canvas, x, y, width, height, color):
    if height < 0:
        pygame_height = -height
        pygame_y = WINDOW_HEIGHT - y
    else:
        pygame_height = height
        pygame_y = WINDOW_HEIGHT - y - height
    pygame.draw.rect(canvas, color, (int(x), int(pygame_y), int(width), int(pygame_height)))

def half_circle(canvas, x, y, r, color):
    res = 20
    points = []
    for i in range(res + 1):
        ang = math.pi - math.pi * i / res
        px = x + math.cos(ang) * r
        py = y + math.sin(ang) * r
        points.append((int(px), int(WINDOW_HEIGHT - py)))
    pygame.draw.polygon(canvas, color, points)

def circle(canvas, x, y, r, color):
    pygame.draw.circle(canvas, color, (int(x), int(WINDOW_HEIGHT - y)), int(r))

class Particle:
    def __init__(self, x, y, vx, vy, r, c):
        self.x = x
        self.y = y
        self.prev_x = self.x
        self.prev_y = self.y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.c = c

    def display(self, canvas):
        circle(canvas, toX(self.x), toY(self.y), toP(self.r), self.c)
        return canvas

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def applyAcceleration(self, ax, ay):
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP

    def checkEdges(self):
        if (self.x <= (self.r - REF_W / 2)):
            self.vx *= -FRICTION
            self.x = self.r - REF_W / 2 + NUDGE * TIMESTEP

        if (self.x >= (REF_W / 2 - self.r)):
            self.vx *= -FRICTION
            self.x = REF_W / 2 - self.r - NUDGE * TIMESTEP

        if (self.y <= (self.r + REF_U)):
            self.vy *= -FRICTION
            self.y = self.r + REF_U + NUDGE * TIMESTEP
            if (self.x <= 0):
                return -1
            else:
                return 1
        if (self.y >= (REF_H - self.r)):
            self.vy *= -FRICTION
            self.y = REF_H - self.r - NUDGE * TIMESTEP
        # fence:
        if ((self.x <= (REF_WALL_WIDTH / 2 + self.r)) and (self.prev_x > (REF_WALL_WIDTH / 2 + self.r)) and (
                self.y <= REF_WALL_HEIGHT)):
            self.vx *= -FRICTION
            self.x = REF_WALL_WIDTH / 2 + self.r + NUDGE * TIMESTEP

        if ((self.x >= (-REF_WALL_WIDTH / 2 - self.r)) and (self.prev_x < (-REF_WALL_WIDTH / 2 - self.r)) and (
                self.y <= REF_WALL_HEIGHT)):
            self.vx *= -FRICTION
            self.x = -REF_WALL_WIDTH / 2 - self.r - NUDGE * TIMESTEP
        return 0

    def getDist2(self, p):
        dy = p.y - self.y
        dx = p.x - self.x
        return (dx * dx + dy * dy)

    def isColliding(self, p):
        r = self.r + p.r
        return (r * r > self.getDist2(p))

    def bounce(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abd = math.sqrt(abx * abx + aby * aby)
        abx /= abd
        aby /= abd
        nx = abx
        ny = aby
        abx *= NUDGE
        aby *= NUDGE
        while (self.isColliding(p)):
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
        if (mag2 > (maxSpeed * maxSpeed)):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed
        if (mag2 < (minSpeed * minSpeed)):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vx *= minSpeed
            self.vy *= minSpeed

class Wall:
    def __init__(self, x, y, w, h, c):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c

    def display(self, canvas):
        rect(canvas, toX(self.x - self.w / 2), toY(self.y - self.h / 2), toP(self.w), toP(self.h), self.c)
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
        return result

class Agent:
    def __init__(self, dir, x, y, c):
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
        self.life = MAXLIVES

    def lives(self):
        return self.life

    def setAction(self, action):
        forward = action[0] > 0
        backward = action[1] > 0
        jump = action[2] > 0
        self.desired_vx = 0
        self.desired_vy = 0
        if forward and not backward:
            self.desired_vx = -PLAYER_SPEED_X
        if backward and not forward:
            self.desired_vx = PLAYER_SPEED_X
        if jump:
            self.desired_vy = PLAYER_SPEED_Y

    def move(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def step(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP

    def update(self):
        self.vy += GRAVITY * TIMESTEP

        if (self.y <= REF_U + NUDGE * TIMESTEP):
            self.vy = self.desired_vy

        self.vx = self.desired_vx * self.dir

        self.move()

        if (self.y <= REF_U):
            self.y = REF_U
            self.vy = 0

        if (self.x * self.dir <= (REF_WALL_WIDTH / 2 + self.r)):
            self.vx = 0
            self.x = self.dir * (REF_WALL_WIDTH / 2 + self.r)

        if (self.x * self.dir >= (REF_W / 2 - self.r)):
            self.vx = 0
            self.x = self.dir * (REF_W / 2 - self.r)

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
        x = toX(self.x)
        y = toY(self.y)
        r = toP(self.r)

        angle = math.pi * 60 / 180 if self.dir == -1 else math.pi * 120 / 180
        eyeX, eyeY = 0, 0

        half_circle(canvas, x, y, r, self.c)

        c = math.cos(angle)
        s = math.sin(angle)
        ballX = toX(bx) - (x + 0.6 * r * c)
        ballY = toY(by) - (y + 0.6 * r * s)

        if self.emotion == "sad":
            ballX = -self.dir * 10
            ballY = -3 * 10

        dist = math.sqrt(ballX ** 2 + ballY ** 2)
        if dist > 0:
            eyeX = ballX / dist
            eyeY = ballY / dist

        eye_pos_x = x + 0.6 * r * c + eyeX * 0.15 * r
        eye_pos_y = y + 0.6 * r * s + eyeY * 0.15 * r
        circle(canvas, eye_pos_x, eye_pos_y, r * 0.3, (255, 255, 255))
        circle(canvas, eye_pos_x + eyeX * 0.15 * r, eye_pos_y + eyeY * 0.15 * r, r * 0.1, (0, 0, 0))

        for i in range(1, self.life):
            coin_x = toX(self.dir * (REF_W / 2 + 0.5 - i * 2.0))
            coin_y = toY(1.5)
            circle(canvas, coin_x, WINDOW_HEIGHT - coin_y, toP(0.5), COIN_COLOR)

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
    def __init__(self, np_random=np.random):
        self.ball = None
        self.ground = None
        self.fence = None
        self.fenceStub = None
        self.agent_left = None
        self.agent_right = None
        self.delayScreen = None
        self.np_random = np_random
        self.reset()

    def reset(self):
        self.ground = Wall(0, 0.75, REF_W, REF_U, GROUND_COLOR)
        self.fence = Wall(0, 0.75 + REF_WALL_HEIGHT / 2, REF_WALL_WIDTH, REF_WALL_HEIGHT - 1.5, FENCE_COLOR)
        self.fenceStub = Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2, FENCE_COLOR)
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5, BALL_COLOR)
        self.agent_left = Agent(-1, -REF_W / 4, 1.5, AGENT_LEFT_COLOR)
        self.agent_right = Agent(1, REF_W / 4, 1.5, AGENT_RIGHT_COLOR)
        self.agent_left.updateState(self.ball, self.agent_right)
        self.agent_right.updateState(self.ball, self.agent_left)
        self.delayScreen = DelayScreen()

    def newMatch(self):
        ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        self.ball = Particle(0, REF_W / 4, ball_vx, ball_vy, 0.5, BALL_COLOR)
        self.delayScreen.reset()

    def step(self):
        self.betweenGameControl()
        self.agent_left.update()
        self.agent_right.update()

        if self.delayScreen.status():
            self.ball.applyAcceleration(0, GRAVITY)
            self.ball.limitSpeed(0, MAX_BALL_SPEED)
            self.ball.move()

        if self.ball.isColliding(self.agent_left):
            self.ball.bounce(self.agent_left)
        if self.ball.isColliding(self.agent_right):
            self.ball.bounce(self.agent_right)
        if self.ball.isColliding(self.fenceStub):
            self.ball.bounce(self.fenceStub)

        # result = -self.ball.checkEdges()
        #
        # if result != 0:
        #     self.newMatch()
        #     if result < 0:
        #         self.agent_left.emotion = "happy"
        #         self.agent_right.emotion = "sad"
        #         self.agent_right.life -= 1
        #     else:
        #         self.agent_left.emotion = "sad"
        #         self.agent_right.emotion = "happy"
        #         self.agent_left.life -= 1
        #     return result
        #
        # self.agent_left.updateState(self.ball, self.agent_right)
        # self.agent_right.updateState(self.ball, self.agent_left)
        #
        # return result

        result = -self.ball.checkEdges()
        reward = 0
        reward += 0.5 / 3000

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
        canvas = create_canvas(canvas, BACKGROUND_COLOR)
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

class SlimeVolley(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = {}):
        self.metadata = {
            'render_modes': ['human', 'rgb_array'],
            'render_fps': 50,
        }
        self.render_mode = environment_configuration.get('render_mode', 'rgb_array')

        self.t = 0
        self.t_limit = 4000
        self.action_space = spaces.MultiDiscrete(np.array([2, 2, 2]))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.game = Game()
        self.viewer = None
        self.screen = None
        self.policy = BaselinePolicy()
        self.otherAction = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game = Game(np_random=self.np_random)
        return [seed]

    def getObs(self):
        return self.game.agent_right.getObservation()

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
            # 'ale.lives': self.game.agent_right.lives(),
            # 'ale.otherLives': self.game.agent_left.lives(),
            # 'state': self.game.agent_right.getObservation(),
            # 'otherState': self.game.agent_left.getObservation(),
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
                self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption('Slime Volleyball')
            surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.game.display(surface)
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        elif self.render_mode == 'rgb_array':
            surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.game.display(surface)
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    FPS = 50
    manualAction = [0, 0, 0]
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False
    policy = BaselinePolicy()

    env = SlimeVolley()
    env.render_mode = 'human'
    obs, _ = env.reset()  # Adapter pour la nouvelle signature de reset
    done = False

    while not done:
        # clock.tick(FPS)
        #
        # # Gestion des événements
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         done = True
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             manualAction[0] = 1
        #         elif event.key == pygame.K_RIGHT:
        #             manualAction[1] = 1
        #         elif event.key == pygame.K_UP:
        #             manualAction[2] = 1
        #         elif event.key == pygame.K_d:
        #             otherManualAction[0] = 1
        #         elif event.key == pygame.K_a:
        #             otherManualAction[1] = 1
        #         elif event.key == pygame.K_w:
        #             otherManualAction[2] = 1
        #     elif event.type == pygame.KEYUP:
        #         if event.key == pygame.K_LEFT:
        #             manualAction[0] = 0
        #         elif event.key == pygame.K_RIGHT:
        #             manualAction[1] = 0
        #         elif event.key == pygame.K_UP:
        #             manualAction[2] = 0
        #         elif event.key == pygame.K_d:
        #             otherManualAction[0] = 0
        #         elif event.key == pygame.K_a:
        #             otherManualAction[1] = 0
        #         elif event.key == pygame.K_w:
        #             otherManualAction[2] = 0
        #
        # # Logique du jeu
        # if manualMode:
        #     action = manualAction
        # else:
        #     action = env.policy.predict(obs)
        #
        # if otherManualMode:
        #     otherAction = otherManualAction
        #     obs, reward, done, _, _ = env.step(action, otherAction)
        # else:
        #     obs, reward, done, _, _ = env.step(action)
        clock.tick(FPS)
        action = policy.predict(env.game.agent_left.getObservation())
        # action = env.action_space.sample()
        env.step(action)
        env.render()

    env.close()