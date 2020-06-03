import random
from collections import deque

import numpy as np

from src.utils.mp_buffer import MiltiprocessBuffer


def get_episode(env, agent, min_seq_len, max_seq_len):
    while True:
        obs = env.reset()
        done = False

        actions = np.zeros((max_seq_len, *env.action_space.shape))
        terminals = np.ones((max_seq_len, ), dtype=np.bool)
        observations = np.zeros((max_seq_len, *obs.shape))
        rewards = np.zeros((max_seq_len, ))

        for i in range(max_seq_len):
            observations[i] = obs
            action = agent(obs)
            actions[i] = action
            terminals[i] = done  # done should be recorded before step

            obs, reward, done, _info = env.step(action)
            rewards[i] = reward

            alive = env.alive if hasattr(env, 'alive') else True
            if done or not alive: break

        if i >= min_seq_len - 1:
            return actions, observations, rewards, terminals


def mp_generator(
    bs,
    env_ctor,
    agent_ctor,
    min_seq_len,
    max_seq_len,
    buffer_size=1000,
    num_processes=8,
):
    def init():
        env = env_ctor()
        agent = agent_ctor()
        while True:
            yield get_episode(
                env=env,
                agent=agent,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
            )

    mpb = MiltiprocessBuffer(
        buffer_size=buffer_size,
        generator_func=init,
        num_processes=num_processes,
    )

    mpb.start()
    local_process_buffer = deque(maxlen=buffer_size)

    def get_batch():
        batch = random.sample(local_process_buffer, bs)
        return [np.array(t) for t in zip(*batch)]

    def generator():
        while True:
            episode = mpb.try_pop()
            if episode is not None:
                local_process_buffer.append(episode)

            if len(local_process_buffer) >= bs:
                yield get_batch()

    generator_instance = generator()

    for _ in range(buffer_size):
        next(generator_instance)

    class StatefulGenerator:
        def __init__(self):
            self.buffer = local_process_buffer

        def __len__(self):
            return len(local_process_buffer)

        def __iter__(self):
            return self

        def __next__(self):
            return next(generator_instance)

    return StatefulGenerator()


def random_mp_generator(
    env_name,
    bs,
    min_seq_len,
    max_seq_len,
    num_processes=16,
):
    from src.utils import make_preprocessed_env

    def env_ctor():
        env = make_preprocessed_env(env_name)
        return env

    def agent_ctor():
        env = env_ctor()
        return lambda obs: env.action_space.sample()

    generator = mp_generator(
        bs=bs,
        env_ctor=env_ctor,
        agent_ctor=agent_ctor,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        buffer_size=1000,
        num_processes=num_processes,
    )

    return generator


if __name__ == '__main__':
    import sneks

    generator = random_mp_generator(
        env_name='snek-rgb-16-v1',
        bs=15,
        min_seq_len=32,
        max_seq_len=32,
    )

    while True:
        batch = next(generator)
        actions, observations, rewards, terminals = batch
        print('Generator len', len(generator))
