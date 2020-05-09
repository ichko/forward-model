from src.utils import persist

import numpy as np
import cv2

import torch
import torch.utils.data as torch_data


class Dataset(torch_data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p, a, f = self.data[idx]
        return (p, a), f


def get_tuple_data(
    env,
    agent,
    dataset_size,
    bs,
    frame_size,
    precondition_size,
    val_frac,
):
    def get_numpy_data():
        total_step = 0

        actions = np.zeros((dataset_size, ), dtype=np.uint8)
        preconditions = np.zeros(
            (dataset_size, precondition_size, 3, *frame_size[::-1]),
            dtype=np.uint8,
        )
        futures = np.zeros((dataset_size, 3, *frame_size[::-1]),
                           dtype=np.uint8)

        while True:
            env.reset()
            done = False

            frames_queue = np.zeros(
                (precondition_size + 1, 3, *frame_size[::-1]),
                dtype=np.uint8,
            )

            episode_step = 0

            while not done:
                action = agent(env)
                _, _, done, _ = env.step(action)
                frame = env.render('rgb_array')
                frame = cv2.resize(frame, frame_size)
                frame = np.transpose(frame, (2, 0, 1))
                frame = frame.astype(np.uint8)

                frames_queue = np.roll(frames_queue, shift=-1, axis=0)
                frames_queue[-1] = frame

                episode_step += 1
                if episode_step >= precondition_size + 1:
                    precondition = frames_queue[:precondition_size]
                    future = frames_queue[-1]
                    last_action = action

                    actions[total_step] = last_action
                    preconditions[total_step] = precondition
                    futures[total_step] = future

                    total_step += 1
                    if total_step >= dataset_size:
                        return actions, preconditions, futures

    actions, preconditions, futures = persist(
        get_numpy_data,
        f'./.data/{env.unwrapped.spec.id}_{dataset_size}_{bs}_{frame_size}_{precondition_size}.pkl',
        override=False,
    )

    train_val_split = int(dataset_size * val_frac)
    rnd_idx = np.random.permutation(dataset_size)
    train_idx = rnd_idx[train_val_split:]
    val_idx = rnd_idx[:train_val_split]

    train_data = torch_data.TensorDataset(
        torch.FloatTensor(preconditions[train_idx]),
        torch.LongTensor(actions[train_idx]),
        torch.FloatTensor(futures[train_idx]),
    )

    val_data = torch_data.TensorDataset(
        torch.FloatTensor(preconditions[val_idx]),
        torch.LongTensor(actions[val_idx]),
        torch.FloatTensor(futures[val_idx]),
    )

    return (
        torch_data.DataLoader(
            Dataset(train_data),
            batch_size=bs,
        ),
        torch_data.DataLoader(
            Dataset(val_data),
            batch_size=bs,
        ),
    )
