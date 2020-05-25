import random
from collections import deque

import numpy as np
import cv2


class RolloutGenerator:
    def __init__(self, *args, **kwargs):
        self.generator = self.make_generator(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def make_generator(
        self,
        env,
        agent,
        bs,
        min_seq_len,
        max_seq_len,
        buffer_size,
        frame_size,
    ):
        ''' Yields batches of episodes - (ep_id, actions, frames) '''

        self.buffer = deque(maxlen=buffer_size)
        self.episodes_len = []

        def get_batch():
            batch = random.sample(self.buffer, bs)
            # ep_id, actions, frames, dones
            return [np.array(t) for t in zip(*batch)]

        ep_id = -1
        while True:
            ep_id += 1

            obs = env.reset()
            done = False

            actions = np.zeros((max_seq_len, *env.action_space.shape))
            dones = np.ones((max_seq_len, ), dtype=np.bool)
            self.episodes_len.append(0)

            frame = env.render('rgb_array')

            # Assume RGB (3 channel) rendering
            frames = np.zeros((max_seq_len, 3, *frame_size[::-1]))

            for i in range(max_seq_len):
                frame = env.render('rgb_array')
                frame = cv2.resize(frame, frame_size)
                frame = np.transpose(frame, (2, 0, 1))
                frames[i] = frame

                action = agent(obs)

                actions[i] = action
                dones[i] = done

                obs, _reward, done, _info = env.step(action)
                self.episodes_len[-1] += 1

                if len(self.buffer) >= bs:
                    yield get_batch()

                alive = env.alive if hasattr(env, 'alive') else True
                if done or not alive:
                    break

            if i >= min_seq_len:
                self.buffer.append([ep_id, actions, frames, dones])
