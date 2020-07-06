import time
import numpy as np
from matplotlib import cm
import random
import torch as T
import os

from src.pipeline.main import get_model
from src.pipeline.config import get_hparams

from src.data.pong import PONGAgent
from src.utils import make_preprocessed_env, get_example_rollout
from src.utils.renderer import Renderer

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    hparams = get_hparams('rnn_spatial_asset_transformer_pong')

    env = make_preprocessed_env(
        hparams.env_name,
        frame_size=hparams.frame_size,
    )

    model = get_model(hparams)
    model.eval()
    model.preload_weights()
    model = model.to('cpu')

    Renderer.init_window(900, 300)
    i = 0
    while True:
        i += 0.01
        assets = model.assets[0, 0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.concatenate(
            [assets[:, :, i] for i in range(assets.shape[2])],
            axis=1,
        )

        precond = model.direction_precondition(
            T.Tensor([[np.sin(i), np.cos(i)]]))

        precond = precond.unsqueeze(0)

        transformed_assets = model.transform_frame([precond, model.assets])
        transformed_assets = transformed_assets[0, 0].detach().cpu().numpy()
        transformed_assets = transformed_assets.transpose(1, 2, 0)
        transformed_assets = transformed_assets.sum(2)
        transformed_assets = cm.viridis(transformed_assets)[:, :, :3]

        Renderer.show_frame(transformed_assets)

        plt.imshow(img, cmap='viridis')
        plt.show()
        exit(0)

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
    if hasattr(env, 'meta') and 'direction' in env.meta:
        precondition = env.meta['direction']

    pred_obs = model.reset(precondition, precondition_actions, input_frames)

    Renderer.init_window(900, 300)

    random_agent = lambda _: env.action_space.sample()
    pong_agent = PONGAgent(env, stochasticity=1)
    agent = pong_agent if 'TwoPlayerPong' in hparams.env_name else random_agent

    y = []
    y_pred = []

    while not done:
        # time.sleep(1 / 5)
        y.append(obs)
        y_pred.append(pred_obs)

        frame = np.concatenate([obs, pred_obs, abs(obs - pred_obs)], axis=2)
        frame = frame.transpose(1, 2, 0)

        # frame = cm.bwr(np.mean(frame, axis=2))[:, :, :3]
        Renderer.show_frame(frame)

        # action = -1
        # while action < 0:
        #     if is_pressed('w'): action = 0
        #     if is_pressed('d'): action = 1
        #     if is_pressed('s'): action = 2
        #     if is_pressed('a'): action = 3

        action = agent(obs)
        # action = 2
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
