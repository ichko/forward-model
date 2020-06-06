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


class PONGGym(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, W, H, direction):
        super().__init__()
        self.W = W
        self.H = H
        self.direction = direction
        self.window_created = False
        self.reset()

    def step(self, action):
        if not self.pong.game_over:
            self.pong.tick(*action)

        obs = self.render('rgb_array')
        reward = 0
        done = self.pong.game_over

        return obs, reward, done, {}

    def reset(self):
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.uint8,
        )

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

    def render(self, mode='human', close=False):
        assert mode in PONGGym.metadata['render.modes'], \
            f'invalid render mode `{mode}`'

        self.R.clear()
        self.pong.left_plank.render(self.R)
        self.pong.right_plank.render(self.R)
        self.pong.ball.render(self.R)

        obs = self.R.canvas[:, :, 0]

        if mode == 'human':
            if not self.window_created:
                self.window_created = True
                Renderer.init_window(self.W, self.H)
            Renderer.show_frame(obs)

        return obs


class StatefulPongGenerator:
    def __init__(self, W, H, seq_len, stochasticity=0.5):
        self.seq_len = seq_len
        self.W = W
        self.H = H
        self.stochasticity = stochasticity

    def __iter__(self):
        return self

    def __next__(self):
        direction = uniform(0, 2 * pi)
        simulation = PONGGym(self.W, self.H, direction)
        pong = simulation.pong
        controls = []
        frames = []
        outcomes = []

        f = uniform(0, 2 * pi)
        for _ in range(self.seq_len):
            f += 0.1

            left_y_diff = pong.ball.pos.y - pong.left_plank.pos.y + \
                pong.plank_height // 2
            left_dir = left_y_diff if pong.ball.pos.x <= 0 else sin(f)

            right_y_diff = pong.ball.pos.y - pong.right_plank.pos.y + \
                pong.plank_height // 2
            right_dir = right_y_diff if pong.ball.pos.x >= 0 else sin(f)

            random_movement_left = choice([-1, 0, 1])
            random_movement_right = choice([-1, 0, 1])

            left_plank_dir = random_movement_left if \
                uniform(0, 1) < self.stochasticity else \
                copysign(1, left_dir)

            right_plank_dir = random_movement_right if \
                uniform(0, 1) < self.stochasticity else \
                copysign(1, right_dir)

            control = [left_plank_dir, right_plank_dir]
            frame, _reward, game_over, _info = simulation.step(control)

            controls.append(control)
            frames.append(frame)
            outcomes.append(game_over)

        return (np.array(direction), np.array(controls)), \
               (np.array(frames), np.array(outcomes))


# REGISTER PONG ENVIRONMENTS
for screen_size in [50, 64, 128]:
    register_gym_env(
        id=f'DeterministicTwoPlayerPong-{screen_size}-v0',
        cls=lambda: PONGGym(screen_size, screen_size, 0.3),
    )

    register_gym_env(
        id=f'TwoPlayerPong-{screen_size}-v0',
        cls=lambda: PONGGym(screen_size, screen_size, uniform(0, 2 * pi)),
    )


def test_games_generator():
    generator = StatefulPongGenerator(32, 32, 128)
    for _ in range(10_000):
        _X, _Y = next(generator)

    (d, c), (f, go) = next(generator)

    print(d.shape)
    print(c.shape)
    print(f.shape)
    print(go.shape)


def test_simulate_single_game():
    _, (frames,
        _) = next(StatefulPongGenerator(
            32,
            32,
            128,
            stochasticity=0.2,
        ))
    print(frames.shape)

    Renderer.init_window(150, 150)

    for frame in frames:
        Renderer.show_frame(frame)


if __name__ == '__main__':
    env = gym.make('DeterministicTwoPlayerPong-50-v0')

    # test_games_generator()
    test_simulate_single_game()
