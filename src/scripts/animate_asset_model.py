import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

import time
from matplotlib import cm
import random
import torch as T
import os

from src.pipeline.common import get_model
from src.pipeline.config import get_hparams

from src.data.pong import PONGAgent, ACTION_MAP_SINGLE_TO_MULTI
from src.utils import make_preprocessed_env, get_example_rollout, random_seed

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


f = 0
plt.ioff()


def main():
    hparams = get_hparams('rnn_spatial_asset_transformer_pong')

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
    if hasattr(env, 'meta') and 'direction' in env.meta:
        precondition = env.meta['direction']

    pred_obs = model.reset(precondition, precondition_actions, input_frames)

    agent = PONGAgent(env, stochasticity=0.2)

    global f
    i = 0
    while not done:
        fig = plt.figure(figsize=(5, 8))

        ax1 = plt.subplot(511)
        ax2 = plt.subplot(513)
        ax3 = plt.subplot(514)
        ax4 = plt.subplot(515)
        ax5 = plt.subplot(512)

        f += 1
        i += 1
        if i >= 150:
            break

        if f >= 256:
            raise Exception('DONE')

        assets = model.assets[0, -1].detach().cpu().numpy().transpose(1, 2, 0)
        merged_assets = np.concatenate(
            [assets[:, :, i] for i in range(assets.shape[2])],
            axis=1,
        )
        transformed_assets = model.transformed_assets[0, -1].detach().cpu()
        transformed_assets = transformed_assets.numpy().transpose(1, 2, 0)
        merged_transformed_assets = np.concatenate(
            [transformed_assets[:, :, i] for i in range(assets.shape[2])],
            axis=1,
        )

        frame = np.concatenate([obs, pred_obs, abs(obs - pred_obs)], axis=2)
        frame = frame.transpose(1, 2, 0)

        cmap = 'viridis'
        ax1.imshow(merged_assets, cmap=cmap)
        ax2.imshow(merged_transformed_assets, cmap=cmap)
        ax3.imshow(np.sum(transformed_assets, axis=2), cmap=cmap)
        ax4.imshow(frame[:, :, 0], cmap=cmap)

        def arrow(ox, oy, dx, dy):
            ax5.arrow(ox, oy, dx, dy, head_width=0.05, head_length=0.1, ec='k')

        theta = model.theta[-3:]
        # for i in range(3):
        #     scale_X = theta[i, :, 0]
        #     scale_Y = theta[i, :, 1]
        #     translate = theta[i, :, 2]

        #     ox, oy = 0.5 + i, 0.5
        #     arrow(ox, oy, scale_X[0], scale_X[1])
        #     arrow(ox, oy, scale_Y[0], scale_Y[1])
        #     arrow(ox, oy, translate[0], translate[1])

        # ax5.set_aspect(1 / 3)
        # ax5.set_xlim(0, 3)
        # ax5.set_ylim(0, 1)

        ax5.imshow(theta.reshape(3, 6))

        ax1.set_title('Assets')
        ax2.set_title('Assets after spatial transformation')
        ax3.set_title('Assets overlayed')
        ax4.set_title('Rollout (true frame, predicted frame, difference)')
        ax5.set_title('Transformations')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])

        fig.tight_layout()
        # plt.pause(0.0005)

        # imagemagc to convert to gif <convert img_*.png ../movie.gif>
        plt.savefig(f'.data/rollouts/img_{f:03}.png')
        # plt.savefig(f'.data/rollouts/last.png')
        # plt.cla()
        plt.close(fig)

        action = agent(obs)

        multi_action = ACTION_MAP_SINGLE_TO_MULTI[action]
        multi_to_str = {0: '■', 1: '▲', -1: '▼'}

        left_action = multi_to_str[multi_action[0]]
        right_action = multi_to_str[multi_action[1]]

        ax4.set_xlabel(
            f'Actions\n{left_action} {right_action}',
            size=20,
        )

        print('ACTION', action)

        obs, reward, done, _info = env.step(action)
        pred_obs = model.step(action)


if __name__ == '__main__':
    random_seed()
    while True:
        main()
