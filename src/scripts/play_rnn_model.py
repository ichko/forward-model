from src.pipelines.main import get_model, hparams
from src.data.pong import PONGAgent

import src.utils.nn as unn
from src.utils import make_preprocessed_env
from src.utils.renderer import Renderer

import time

import numpy as np
import cv2

from matplotlib import cm

win_name = 'window'


def main():
    env = make_preprocessed_env(
        hparams.env_name,
        frame_size=hparams.frame_size,
    )
    model = get_model(hparams)
    model.eval()
    model.preload_weights()
    model = model.to('cuda')

    obs = env.reset()
    input_frames = []
    precondition_actions = []
    for i in range(hparams.precondition_size):
        action = env.action_space.sample()

        input_frames.append(obs)
        precondition_actions.append(action)

        obs, reward, done, _info = env.step(action)
        if done:
            raise Exception('env done too early')

    precondition = input_frames
    if hasattr(env, 'meta') and 'direction' in env.meta:
        precondition = env.meta['direction']

    pred_obs = model.reset(precondition, precondition_actions, input_frames)

    Renderer.init_window(900, 300)

    agent = PONGAgent(env, stochasticity=0.9)

    while not done:
        # time.sleep(1 / 5)

        frame = np.concatenate([obs, pred_obs, abs(obs - pred_obs)], axis=2)
        frame = frame.transpose(1, 2, 0)

        frame = cm.viridis(np.mean(frame, axis=2))[:, :, :3]
        Renderer.show_frame(frame)

        # action = -1
        # while action < 0:
        #     if is_pressed('w'): action = 0
        #     if is_pressed('d'): action = 1
        #     if is_pressed('s'): action = 2
        #     if is_pressed('a'): action = 3

        action = agent(obs)
        print('ACTION', action)

        obs, reward, done, _info = env.step(action)
        pred_obs = model.step(action)


if __name__ == '__main__':
    while True:
        main()
