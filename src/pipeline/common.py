import importlib
import gym
import sneks

import random
from src.data.mp_rollout_generator import preprocessed_mp_generator
from src.data.pong import PONGAgent


def get_model(hparams):
    env = gym.make(hparams.env_name)
    model_module = importlib.import_module(f'src.models.{hparams.model}')
    # model_module.sanity_check()

    hparams_dict = vars(hparams)
    model = model_module.make_model({
        **hparams_dict,
        'num_actions': env.action_space.n,
    })

    model.make_persisted(f'.models/{model.name}_{hparams.env_name}.h5')

    return model


def get_data_generator(
    env_name,
    bs,
    min_seq_len,
    max_seq_len,
    frame_size,
    moving_window_slices,
    num_processes,
    stochasticity=None,
):
    def pong_agent_ctor(env):
        nonlocal stochasticity
        if stochasticity is None:
            stochasticity = random.uniform(0.5, 1)

        return PONGAgent(env, stochasticity)

    agent_ctor = None  # random agenet
    if 'TwoPlayerPong' in env_name:
        agent_ctor = pong_agent_ctor

    return preprocessed_mp_generator(
        env_name=env_name,
        bs=bs,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        agent_ctor=agent_ctor,
        frame_size=frame_size,
        num_processes=num_processes,
        buffer_size=512,
        moving_window_slices=moving_window_slices,
    )
