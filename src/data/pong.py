import time
from math import sin, cos, copysign, sqrt, pi
from random import uniform, random as rand, choice
import numpy as np

import gym
from gym import spaces

from src.utils.renderer import Renderer
from src.utils import register_gym_env


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vec):
        self.x += vec.x
        self.y += vec.y

        return self

    @property
    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)

    @property
    def noramalized(self):
        length = self.length
        return Vector(self.x / length, self.y / length)

    def __repr__(self):
        return 'vec(%f, %f)' % (self.x, self.y)


def vec(x=0, y=0):
    return Vector(x, y)


def polar(angle, magnitude):
    return vec(cos(angle) * magnitude, sin(angle) * magnitude)


class Plank:
    def __init__(self, x, y, width, height):
        self.pos = vec(x, y)
        self.width = width
        self.height = height
        self.speed = 1

    def render(self, renderer):
        renderer.rect(self.pos.x, self.pos.y, self.width, self.height)


class Ball:
    def __init__(self, x, y, size, direction):
        self.pos = vec(x, y)
        self.velocity = polar(direction, 1)
        self.size = size

    def render(self, renderer):
        renderer.rect(self.pos.x - self.size, self.pos.y + self.size,
                      self.size * 2, self.size * 2)


class PONG:
    def __init__(self, w, h, pw, ph, bs, b_dir):
        self.width = w
        self.height = h

        self.plank_width = pw
        self.plank_height = ph
        self.ball_size = bs
        self.game_over = False

        self.left_plank = Plank(-self.width / 2, 0, self.plank_width,
                                self.plank_height)

        self.right_plank = Plank(self.width / 2 - self.plank_width, 0,
                                 self.plank_width, self.plank_height)

        self.ball = Ball(0, 0, self.ball_size, b_dir)

    def update_plank(self, plank, inp):
        plank.pos.y += inp * plank.speed

        if plank.pos.y > self.height / 2:
            plank.pos.y = self.height / 2

        if plank.pos.y < -self.height / 2 + plank.height:
            plank.pos.y = -self.height / 2 + plank.height

    def update_ball(self, ball):
        ball.pos.add(ball.velocity)

        left_wall = -self.width / 2 + (ball.size + 1) + self.plank_width
        right_wall = self.width / 2 - (ball.size + 1) - self.plank_width

        top_wall = self.height / 2 - ball.size
        bottom_wall = -self.height / 2 + ball.size + 1

        left_h_constraint = ball.pos.y - ball.size < self.left_plank.pos.y and \
            ball.pos.y + ball.size > self.left_plank.pos.y - self.left_plank.height

        right_h_constraint = ball.pos.y - ball.size < self.right_plank.pos.y and \
            ball.pos.y + ball.size > self.right_plank.pos.y - self.right_plank.height

        if not left_h_constraint and ball.pos.x < left_wall or \
                not right_h_constraint and ball.pos.x > right_wall:
            self.game_over = True

        if ball.pos.x > right_wall:
            ball.pos.x = right_wall
            ball.velocity.x *= -1

        if ball.pos.x < left_wall:
            ball.pos.x = left_wall
            ball.velocity.x *= -1

        if ball.pos.y > top_wall:
            ball.pos.y = top_wall
            ball.velocity.y *= -1

        if ball.pos.y < bottom_wall:
            ball.pos.y = bottom_wall
            ball.velocity.y *= -1

    def tick(self, left_input, right_input):
        self.update_plank(self.left_plank, left_input)
        self.update_plank(self.right_plank, right_input)
        self.update_ball(self.ball)


# This generates action mappings
# From milti-discrete([3, 3]) to discrete(9) and the reversed
def _generate_action_mappers():
    multi_to_single = dict()
    single_to_multi = dict()
    idx = 0
    for y in range(-1, 2):
        for x in range(-1, 2):
            multi_to_single[x, y] = idx
            single_to_multi[idx] = x, y
            idx += 1

    return multi_to_single, single_to_multi


ACTION_MAP_MULTI_TO_SINGLE, ACTION_MAP_SINGLE_TO_MULTI = \
    _generate_action_mappers()


class PONGGym(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, W, H, direction):
        super().__init__()
        self.W = W
        self.H = H

        self.direction = direction
        self.meta = {'direction': [np.sin(direction), np.cos(direction)]}

        self.window_created = False
        self.reset()

    def step(self, action):
        if not self.pong.game_over:
            action = ACTION_MAP_SINGLE_TO_MULTI[action]
            self.pong.tick(*action)

        obs = self.render('rgb_array')
        reward = not self.pong.game_over
        done = self.pong.game_over

        return obs, reward, done, dict()

    def reset(self):
        # 9 actions, 3 for each player - [up, stay, down]
        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.H, self.W, 3),
            dtype=np.uint8,
        )

        self.R = Renderer(self.W, self.W)
        self.pong = PONG(
            w=self.W,
            h=self.H,
            pw=5,
            ph=15,
            bs=2,
            b_dir=self.direction,
        )

        return self.render('rgb_array')

    def render(self, mode='rgb_array', close=False):
        assert mode in PONGGym.metadata['render.modes'], \
            f'invalid render mode `{mode}`'

        self.R.clear()
        self.pong.left_plank.render(self.R)
        self.pong.right_plank.render(self.R)
        self.pong.ball.render(self.R)

        obs = self.R.canvas

        if mode == 'human':
            if not self.window_created:
                self.window_created = True
                Renderer.init_window(self.W, self.H)
            Renderer.show_frame(obs)

        return obs * 255


class PONGAgent:
    def __init__(self, pong_env, stochasticity=0.5):
        self.pong_env = pong_env
        self.stochasticity = stochasticity
        self.f = uniform(0, 2 * pi)

    def __call__(self, _obs):
        self.f += 0.1
        pong = self.pong_env.pong

        left_y_diff = pong.ball.pos.y - pong.left_plank.pos.y + \
            pong.plank_height // 2
        left_dir = left_y_diff if pong.ball.pos.x <= 0 else sin(self.f)

        right_y_diff = pong.ball.pos.y - pong.right_plank.pos.y + \
            pong.plank_height // 2
        right_dir = right_y_diff if pong.ball.pos.x >= 0 else sin(self.f)

        random_movement_left = choice([-1, 0, 1])
        random_movement_right = choice([-1, 0, 1])

        left_plank_dir = random_movement_left if \
            uniform(0, 1) < self.stochasticity else \
            copysign(1, left_dir)

        right_plank_dir = random_movement_right if \
            uniform(0, 1) < self.stochasticity else \
            copysign(1, right_dir)

        action = ACTION_MAP_MULTI_TO_SINGLE[left_plank_dir, right_plank_dir]
        return action


# class PONGUserAgent:
#     def __call__(self, _obs):
#         action = ACTION_MAP_MULTI_TO_SINGLE[left_plank_dir, right_plank_dir]
#         return action


# REGISTER PONG ENVIRONMENTS
def pong_ctor(W, H):
    return lambda: PONGGym(W, H, direction=0.3)


def random_pong_ctor(W, H):
    return lambda: PONGGym(W, H, uniform(0, 2 * pi))


for screen_size in [32, 40, 50, 64]:
    register_gym_env(
        id=f'DeterministicTwoPlayerPong-{screen_size}-v0',
        cls=pong_ctor(screen_size, screen_size),
    )

    register_gym_env(
        id=f'TwoPlayerPong-{screen_size}-v0',
        cls=random_pong_ctor(screen_size, screen_size),
    )


def sanity_check():
    import gym

    env = gym.make('TwoPlayerPong-40-v0')
    agent = PONGAgent(env, stochasticity=0.0)
    Renderer.init_window(200, 200)

    for _i in range(10):
        obs = env.reset()
        done = False

        while not done:
            Renderer.show_frame(obs)

            action = agent(obs)
            obs, _reward, done, _info = env.step(action)

            if not Renderer.can_render():
                return


if __name__ == '__main__':
    sanity_check()
