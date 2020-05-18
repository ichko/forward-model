from src.models.rnn_deconvolve import sanity_check, make_model
from src.data.rollout_generator import RolloutGenerator
from src.utils.trainer import fit_generator
from src.loggers.wandb import WAndBLogger

import argparse

import gym
import sneks

# CubeCrash-v0
# snek-rgb-16-v1

# parser = argparse.ArgumentParser('RUN Experiment')
# parser.add_argument()

hparams = argparse.Namespace(
    env_name='snek-rgb-16-v1',
    precondition_size=2,
    dataset_size=200_000,
    frame_size=(16, 16),
    its=1_000_000,
    bs=64,
    log_interval=300,
    lr=0.001,
    device='cuda',
    max_seq_len=32,
)


def get_env():
    return gym.make(hparams.env_name)


def get_model(env):
    model = make_model(
        precondition_size=hparams.precondition_size,
        frame_size=hparams.frame_size,
        num_actions=env.action_space.n,
    )
    model.make_persisted('.models/rnn.h5')

    return model


def get_data_generator(env, agent=None):
    agent = (lambda _: env.action_space.sample()) if agent is None else agent

    gen = RolloutGenerator(
        env=env,
        agent=agent,
        bs=hparams.bs,
        max_seq_len=hparams.max_seq_len,
        buffer_size=hparams.dataset_size,
        frame_size=hparams.frame_size,
    )

    for _ep_id, actions, frames in gen:
        x = frames[:, :hparams.precondition_size], \
            actions[:, hparams.precondition_size:]
        y = frames[:, hparams.precondition_size:]

        yield x, y


def main():
    sanity_check()

    env = get_env()
    train_data_generator = get_data_generator(env)
    val_data_generator = get_data_generator(env)

    model = get_model(env)
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
