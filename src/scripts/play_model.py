import time
import numpy as np
from matplotlib import cm
import random
import torch as T
import os

from src.pipeline.train import get_model
from src.pipeline.config import get_hparams

from src.data.pong import PONGAgent, PONGUserAgent
from src.utils import make_preprocessed_env, get_example_rollout
from src.utils.renderer import Renderer

import matplotlib.pyplot as plt


def main():
    hparams = get_hparams('rnn_spatial_transformer_pong')

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

    precondition = input_frames[::]
    if hparams.precondition_type == 'meta':
        precondition = env.meta['direction']

    pred_obs = model.reset(precondition, precondition_actions, input_frames)

    Renderer.init_window(900, 300)

    random_agent = lambda _: env.action_space.sample()

    # pong_agent = PONGAgent(env, stochasticity=0.8)
    pong_agent = PONGUserAgent()

    agent = pong_agent if 'TwoPlayerPong' in hparams.env_name else random_agent

    y = []
    y_pred = []

    while not done:
        # time.sleep(1 / 1)
        y.append(obs)
    while not done:
        # time.sleep(1 / 1)
        y.append(obs)
        frame = np.concatenate([obs, pred_obs, abs(obs - pred_obs)], axis=2)
        frame = frame.transpose(1, 2, 0)

        # frame = cm.bwr(np.mean(frame, axis=2))[:, :, :3]
        Renderer.show_frame(frame)
        action = agent(obs)

        if 'TwoPlayerPong' in hparams.env_name:
            multi_to_str = {0: '■', 1: '▲', -1: '▼'}
            left_action = multi_to_str[agent.multi_action[0]]
            right_action = multi_to_str[agent.multi_action[1]]

            print(f'ACTIONS: {left_action} {right_action}')
        else:
            print('ACTION', action)

        obs, reward, done, _info = env.step(action)
        pred_obs = model.step(action)

    # get_example_rollout({
    #     'y': [y],
    #     'y_pred': [y_pred],
    # }, id=0, show=True)


if __name__ == '__main__':
    while True:
        main()
