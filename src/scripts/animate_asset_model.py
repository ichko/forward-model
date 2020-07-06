import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

import time
from matplotlib import cm
import random
import torch as T
import os

from src.pipeline.main import get_model
from src.pipeline.config import get_hparams

from src.data.pong import PONGAgent
from src.utils import make_preprocessed_env, get_example_rollout

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


fig = plt.figure(figsize=(5, 8))

ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)


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

    agent = lambda _: 2

    i = 0
    while not done:
        i += 1
        if i >= 50:
            break

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

        ax1.set_title('Assets')
        ax2.set_title('Assets after spatial transformation')
        ax3.set_title('Assets overlayed')
        ax4.set_title('Rollout (true frame, predicted frame, difference)')

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])

        fig.tight_layout()
        plt.pause(0.05)
        plt.savefig(f'.data/rollouts/last.png')
        plt.cla()

        action = agent(obs)

        print('ACTION', action)

        obs, reward, done, _info = env.step(action)
        pred_obs = model.step(action)


if __name__ == '__main__':
    while True:
        main()
