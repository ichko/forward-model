import random

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
        max_seq_len,
        buffer_size,
        frame_size,
    ):
        ''' Yields batches of episodes - (ep_id, actions, frames) '''

        self.buffer = []
        self.episodes_len = []

        def get_batch():
            batch = random.sample(self.buffer, bs)
            ep_id, actions, frames = [np.array(t) for t in zip(*batch)]
            return ep_id, actions, frames

        for ep_id in range(buffer_size):
            obs = env.reset()
            done = False

            actions = np.zeros((max_seq_len, *env.action_space.shape))
            self.episodes_len.append(0)

            frame = env.render('rgb_array')

            # Assume RGB (3 channel) rendering
            frames = np.zeros((max_seq_len, 3, *frame_size[::-1]))

            for i in range(max_seq_len):
                action = agent(obs)

                actions[i] = action

                frame = env.render('rgb_array')
                frame = cv2.resize(frame, frame_size)
                frame = np.transpose(frame, (2, 0, 1))
                frames[i] = frame

                obs, _reward, done, _info = env.step(action)
                self.episodes_len[-1] += 1

                if len(self.buffer) >= bs:
                    yield get_batch()
                if done:
                    break

            self.buffer.append([ep_id, actions, frames])

        while True:
            yield get_batch()
