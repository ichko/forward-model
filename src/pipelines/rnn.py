from src.models.rnn_deconvolve import sanity_check as rnn_sanity_check, RNN
from src.data.rollout_generator import RolloutGenerator
from src.utils.trainer import fit_generator
from src.loggers.wandb import WAndBVideoLogger

import argparse

import gym
import sneks

# CubeCrash-v0
# snek-rgb-16-v1

# parser = argparse.ArgumentParser('RUN Experiment')
# parser.add_argument()

hparams = argparse.Namespace(
    env_name='CubeCrash-v0',
    precondition_size=1,
    num_rollouts=10_000,
    frame_size=(64, 64),
    its=50_000,
    bs=32,
    log_interval=40,
    lr=0.001,
    device='cuda',
    max_seq_len=32,
)


def get_env():
    return gym.make(hparams.env_name)


def get_model(env):
    model = RNN(
        num_precondition_frames=hparams.precondition_size,
        frame_size=hparams.frame_size,
        num_rnn_layers=3,
        num_actions=env.action_space.n,
        action_embedding_size=32,
        rnn_hidden_size=32,
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
        buffer_size=hparams.num_rollouts,
        frame_size=hparams.frame_size,
    )

    for _ep_id, actions, frames in gen:
        x = frames[:, :hparams.precondition_size], \
            actions[:, hparams.precondition_size:]
        y = frames[:, hparams.precondition_size:]

        yield x, y


def main():
    rnn_sanity_check()

    env = get_env()
    train_data_generator = get_data_generator(env)
    val_data_generator = get_data_generator(env)

    model = get_model(env)
    model.configure_optim(lr=hparams.lr)
    model = model.to(hparams.device)

    logger = WAndBVideoLogger(
        info_log_interval=hparams.log_interval,
        model=model,
        hparams=hparams,
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
