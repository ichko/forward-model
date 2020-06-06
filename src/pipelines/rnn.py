import sys

from src.models.rnn_deconvolve import sanity_check, make_model
from src.data.mp_rollout_generator import random_mp_generator
from src.utils.trainer import fit_generator
from src.utils import add_virtual_display
from src.loggers.wandb import WAndBLogger

import argparse

import gym
import sneks

hparams = argparse.Namespace(
    env_name='CubeCrash-v0',
    # env_name='snek-rgb-16-v1',
    # env_name='CartPole-v1',
    # env_name='LunarLander-v2',
    precondition_size=2,
    dataset_size=5000,
    frame_size=(32, 32),
    its=1_000_000,
    bs=64,
    log_interval=300,
    lr=0.0001,
    device='cuda',
    max_seq_len=32,
    min_seq_len=2,
)


def get_env():
    return gym.make(hparams.env_name)


def get_model(hparams, env):
    model = make_model(
        precondition_size=hparams.precondition_size,
        frame_size=hparams.frame_size,
        num_actions=env.action_space.n,
    )
    model.make_persisted('.models/rnn.h5')

    return model


def get_data_generator(env, agent=None):
    agent = (lambda _: env.action_space.sample()) if agent is None else agent

    return random_mp_generator(
        env_name=hparams.env_name,
        bs=hparams.bs,
        min_seq_len=hparams.min_seq_len,
        max_seq_len=hparams.max_seq_len,
        frame_size=hparams.frame_size,
        num_processes=16,
    )


def main():
    add_virtual_display()
    sanity_check()

    env = get_env()
    train_data_generator = get_data_generator(env)
    val_data_generator = get_data_generator(env)

    model = get_model(hparams, env)

    if '--from-scratch' not in sys.argv:
        try:
            model.preload_weights()
            print('>>> MODEL PRELOADED')
        except Exception as _e:
            print('>>> Could not preload! Starting from scratch.')

    model.configure_optim(lr=hparams.lr)
    model = model.to(hparams.device)

    logger = WAndBLogger(
        info_log_interval=hparams.log_interval,
        model=model,
        hparams=hparams,
        type='video',
    )

    print(model.summary())

    fit_generator(
        model,
        train_data_generator,
        val_data_generator,
        its=hparams.its,
        logger=logger,
        persist_frequency=hparams.log_interval,
        log_info_interval=hparams.log_interval,
    )


if __name__ == '__main__':
    main()
