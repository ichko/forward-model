import random
from collections import deque, defaultdict

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

        episode_len = i + 1
        if i >= min_seq_len - 1:
            meta = env.meta if hasattr(env, 'meta') else dict()
            return meta, (
                episode_len,
                actions,
                observations,
                rewards,
                terminals,
            )


def slice_episode(episode, moving_window_slices):
    meta, (episode_len, actions, observations, rewards, terminals) = episode
    result = []
    for i in range(episode_len - moving_window_slices):
        episode_slice = meta, (
            moving_window_slices,
            actions[i:i + moving_window_slices],
            observations[i:i + moving_window_slices],
            rewards[i:i + moving_window_slices],
            terminals[i:i + moving_window_slices],
        )

        result.append(episode_slice)

    return result


def mp_generator(
    bs,
    env_ctor,
    agent_ctor,
    min_seq_len,
    max_seq_len,
    moving_window_slices,
    buffer_size=1000,
    num_processes=8,
):
    def init():
        while True:
            env = env_ctor()
            agent = agent_ctor(env)

            episode = get_episode(
                env=env,
                agent=agent,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
            )

            if moving_window_slices is not None:
                episode_slices = slice_episode(
                    episode,
                    moving_window_slices,
                )
                yield episode_slices
            else:
                yield [episode]

    mpb = MiltiprocessBuffer(
        buffer_size=buffer_size,
        generator_func=init,
        num_processes=num_processes,
    )

    mpb.start()
    local_process_buffer = deque(maxlen=buffer_size)

    def get_batch():
        batch = random.sample(local_process_buffer, bs)
        meta, tensors = zip(*batch)
        episode_len, actions, observations, rewards, terminals = [
            np.array(t) for t in zip(*tensors)
        ]

        batched_meta = defaultdict(lambda: [])
        for m in meta:
            for k, v in m.items():
                batched_meta[k].append(v)
        for k, v in batched_meta.items():
            batched_meta[k] = np.array(v)

        return {
            'meta': batched_meta,
            'episode_len': episode_len,
            'actions': actions,
            'observations': observations,
            'rewards': rewards,
            'terminals': terminals,
        }

    def local_process_generator():
        while True:
            episode_slices = mpb.try_pop()
            if episode_slices is not None:
                for s in episode_slices:
                    if len(local_process_buffer) >= bs:
                        yield get_batch()

                    local_process_buffer.append(s)

            if len(local_process_buffer) >= bs:
                yield get_batch()

    generator_instance = local_process_generator()

    class StatefulGenerator:
        def __init__(self):
            self.buffer = local_process_buffer

        def __len__(self):
            return len(local_process_buffer)

        def __iter__(self):
            return self

        def __next__(self):
            return next(generator_instance)

        def __del__(self):
            return mpb.terminate()

    return StatefulGenerator()


def preprocessed_mp_generator(
    env_name,
    bs,
    min_seq_len,
    max_seq_len,
    agent_ctor=None,
    frame_size=None,
    num_processes=16,
    buffer_size=256,
    moving_window_slices: int = None,
):
    from src.utils import make_preprocessed_env

    def env_ctor():
        env = make_preprocessed_env(env_name, frame_size=frame_size)
        return env

    def random_agent_ctor(env):
        return lambda _obs: env.action_space.sample()

    if agent_ctor is None:
        agent_ctor = random_agent_ctor

    return mp_generator(
        bs=bs,
        env_ctor=env_ctor,
        agent_ctor=agent_ctor,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        buffer_size=buffer_size,
        num_processes=num_processes,
        moving_window_slices=moving_window_slices,
    )


if __name__ == '__main__':
    import pong
    import random

    def agent_ctor(env):
        return pong.PONGAgent(env, stochasticity=random.uniform(0.8, 1))

    generator = preprocessed_mp_generator(
        env_name='TwoPlayerPong-32-v0',
        bs=32,
        min_seq_len=10,
        max_seq_len=260,
        agent_ctor=agent_ctor,
        frame_size=(32, 32),
        num_processes=8,
        buffer_size=1024,
        moving_window_slices=32,
    )

    for i in range(100000):
        batch = next(generator)
        print(f'{i:04} Generator size:', len(generator))
