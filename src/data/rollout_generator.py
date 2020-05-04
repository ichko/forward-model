import random

import numpy as np


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
        frame_size=None,
    ):
        ''' Yields batches of episodes - (ep_id, actions, observations, frames?) '''

        self.buffer = []
        self.episodes_len = []

        def get_batch():
            batch = random.sample(self.buffer, bs)
            ep_id, actions, obs, frames = [np.array(t) for t in zip(*batch)]
            return dict(ep_id=ep_id, actions=actions, obs=obs, frames=frames)

        for ep_id in range(buffer_size):
            obs = env.reset()
            done = False

            actions = np.zeros((max_seq_len, *env.action_space.shape))
            observations = np.zeros((max_seq_len, *obs.shape))
            self.episodes_len.append(0)

            frame = env.render('rgb_array')
            # Reverse to preserve in (W, H) form
            frame_size = frame.shape[:2][::-1] \
                if frame_size is None else frame_size

            # Assume RGB (3 channel) rendering
            frames = np.zeros((max_seq_len, *frame_size[::-1], 3))

            for i in range(max_seq_len):
                action = agent(obs)
                actions[i] = action
                observations[i] = obs
                frame = env.render('rgb_array')
                frames[i] = cv2.resize(frame, frame_size)

                obs, _reward, done, _info = env.step(action)
                self.episodes_len[-1] += 1

                if len(self.buffer) >= bs:
                    yield get_batch()
                if done:
                    break

            self.buffer.append([ep_id, actions, observations, frames])

        while True:
            yield get_batch()
