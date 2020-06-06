import sys

from src.models.rnn_deconvolve import sanity_check as model_sanity_check, make_model
from src.utils.trainer import fit_generator
from src.loggers.wandb import WAndBLogger

import argparse

import gym
import sneks

hparams = argparse.Namespace(
    env_name='DeterministicTwoPlayerPong-64-v0',
    # env_name='CubeCrash-v0',
    # env_name='snek-rgb-16-v1',
    # env_name='CartPole-v1',
    # env_name='LunarLander-v2',
    precondition_size=2,
    dataset_size=5000,
    frame_size=(32, 32),
    its=1_000_000,
    bs=32,
    log_interval=300,
    lr=0.0001,
    device='cuda',
    max_seq_len=128,
    min_seq_len=64,
)


def get_model(hparams):
    env = gym.make(hparams.env_name)

    model = make_model(
        precondition_size=hparams.precondition_size,
        frame_size=hparams.frame_size,
        num_actions=env.action_space.n,
    )
    model.make_persisted(f'.models/{model.name}_{hparams.env_name}.h5')

    return model


def get_data_generator(hparams):
    from src.data.mp_rollout_generator import preprocessed_mp_generator

    def pong_generator():
        import random
        from src.data.pong import PONGAgent

        def agent_ctor(env):
            return PONGAgent(env, stochasticity=random.random())

        return preprocessed_mp_generator(
            env_name='DeterministicTwoPlayerPong-64-v0',
            bs=hparams.bs,
            min_seq_len=hparams.min_seq_len,
            max_seq_len=hparams.max_seq_len,
            agent_ctor=agent_ctor,
            frame_size=hparams.frame_size,
            num_processes=15,
        )

    if hparams.env_name == 'DeterministicTwoPlayerPong-64-v0':
        return pong_generator()

    return preprocessed_mp_generator(
        env_name=hparams.env_name,
        bs=hparams.bs,
        min_seq_len=hparams.max_seq_len,
        max_seq_len=hparams.min_seq_len,
        agent_ctor=None,  # random agent
        frame_size=hparams.env_name,
        num_processes=16,
    )


def main():
    model_sanity_check()

    train_data_generator = get_data_generator(hparams)
    val_data_generator = get_data_generator(hparams)

    model = get_model(hparams)

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
