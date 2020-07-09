import time
import numpy as np
import random
import torch as T
from matplotlib import cm
from tqdm.auto import tqdm
import os

from src.pipeline.train import get_model, get_hparams
from src.data.pong import PONGAgent
from src.utils import make_preprocessed_env


def evaluate(hparams):
    env = make_preprocessed_env(
        hparams.env_name,
        frame_size=hparams.frame_size,
    )

    model = get_model(hparams)
    model = model.to('cuda')
    model.preload_weights()
    model.eval()

    max_ep_len = 64
    min_ep_len = 32
    num_episodes = 100
    ERRORS = []

    for _ in tqdm(range(num_episodes)):
        while True:
            env = make_preprocessed_env(
                hparams.env_name,
                frame_size=hparams.frame_size,
            )

            step_id = 1
            EP_ERROR = 0

            obs = env.reset()
            input_frames = []
            precondition_actions = []
            for i in range(hparams.precondition_size):
                action = env.action_space.sample()

                input_frames.append(obs)
                precondition_actions.append(action)

                obs, _reward, done, _info = env.step(action)
                if done:
                    raise Exception('env done too early')

            precondition = input_frames[::]
            if hparams.precondition_type == 'meta':
                precondition = env.meta['direction']

            pred_obs = model.reset(
                precondition,
                precondition_actions,
                input_frames,
            )

            random_agent = lambda _: env.action_space.sample()
            pong_agent = PONGAgent(env, stochasticity=0.9)
            agent = pong_agent if 'TwoPlayerPong' in hparams.env_name else random_agent

            while not done:
                step_id += 1
                EP_ERROR += np.mean((obs - pred_obs)**2)
                frame = np.concatenate(
                    [obs, pred_obs, abs(obs - pred_obs)], axis=2)
                frame = frame.transpose(1, 2, 0)

                frame = cm.viridis(np.mean(frame, axis=2))[:, :, :3]

                action = agent(obs)
                # print(action)

                obs, reward, done, _info = env.step(action)
                pred_obs = model.step(action)

                if step_id >= max_ep_len:
                    break

            if step_id >= min_ep_len:
                # print(f'EP ERROR: {EP_ERROR}, EP LEN: {step_id}')
                ERRORS.append(EP_ERROR / step_id)
                break

    mean = np.mean(ERRORS)
    std = np.std(ERRORS)
    print(f'MSE: {mean:.6f} ± {std:.6f}')

    return mean


if __name__ == '__main__':
    hparams = get_hparams('rnn_dense_pong')
    evaluate(hparams)
