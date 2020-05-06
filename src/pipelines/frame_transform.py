from src.models.kernel_regressor import sanity_check, make_model
from src.data.tuple_generator import get_tuple_data
from src.utils.trainer import fit
from src.utils import persist
from src.loggers.wandb import WAndBLogger

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
    dataset_size=100_000,
    frame_size=(64, 64),
    bs=64,
    log_interval=40,
    lr=0.001,
    device='cuda',
    epochs=5000,
)


def get_env():
    return gym.make(hparams.env_name)


def get_model(env):
    model = make_model(
        num_precondition_frames=hparams.precondition_size,
        frame_size=hparams.frame_size,
        num_actions=env.action_space.n,
    )
    model.make_persisted('.models/frame_transform.h5')

    return model


def get_data_generator(env, agent=None):
    agent = (lambda _: env.action_space.sample()) if agent is None else agent

    return get_tuple_data(
        env=env,
        agent=agent,
        dataset_size=hparams.dataset_size,
        frame_size=hparams.frame_size,
        precondition_size=1,
        bs=hparams.bs,
        val_frac=0.2,
    )


def main():
    sanity_check()

    env = get_env()
    train_data = get_data_generator(env)

    model = get_model(env)
    model.configure_optim(lr=hparams.lr)
    model = model.to(hparams.device)

    logger = WAndBLogger(
        info_log_interval=hparams.log_interval,
        model=model,
        hparams=hparams,
        type='image',
    )

    fit(
        model=model,
        data=train_data,
        epochs=hparams.epochs,
        logger=logger,
        log_interval=hparams.log_interval,
    )


if __name__ == '__main__':
    main()
