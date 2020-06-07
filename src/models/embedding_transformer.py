import src.utils.torch as tu

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class FrameTransform(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.compute_precondition = nn.Sequential(
            nn.Dropout(0.5),
            tu.conv_transform([3, 32, 32]),
        )

        self.action_transformations = tu.KernelEmbedding(
            num_actions, [
                (32, 32, 5),
                (32, 64, 5),
                (64, 32, 5),
            ])

        self.expand_transformed = nn.Sequential(
            nn.Dropout(0.1),
            tu.conv_transform([32, 32, 3]),
            nn.Sigmoid(),
        )

    def forward(self, action, frame):
        frame_map = self.compute_precondition(frame)
        transformed_frame = self.action_transformations(frame_map, action)
        pref_frame = self.expand_transformed(transformed_frame)

        return pref_frame


class EmbeddingTransformer(tu.BaseModule):
    def __init__(self, num_actions):
        super().__init__()
        self.name = 'Embedding Transformer'
        self.frame_transform = tu.time_distribute(FrameTransform(num_actions))

    def forward(self, actions, frames):
        return self.frame_transform(
            T.LongTensor(actions).to(self.device),
            T.FloatTensor(frames).to(self.device),
        )

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        terminals = T.BoolTensor(terminals).to(self.device)
        y_true = T.FloatTensor(observations).to(self.device)[:, 1:]

        y_pred = self(actions[:, :-1], observations[:, :-1])
        y_pred = tu.mask_sequence(y_pred, ~terminals)

        loss = F.binary_cross_entropy(y_pred, y_true)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optim.step()

        return loss, {'y': y_true, 'y_pred': y_pred}

    def configure_optim(self, lr):
        self.optim = T.optim.Adam(self.parameters(), lr=lr)


def make_model(num_actions):
    return EmbeddingTransformer(num_actions=num_actions)


def sanity_check():
    frame_size = (32, 32)
    num_actions = 3
    max_seq_len = 32
    bs = 32

    model = make_model(num_actions).to('cuda')

    print(model.summary())

    frames = T.randint(
        0,
        255,
        size=(bs, max_seq_len, 3, *frame_size),
    ) / 255.0
    actions = T.randint(0, num_actions, size=(bs, max_seq_len))
    out_frames = model(actions, frames).detach().cpu()
    terminals = T.rand(bs, max_seq_len) > 0.5

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    model.configure_optim(lr=0.0001)
    batch = {
        'actions': actions,
        'observations': frames,
        'terminals': terminals,
    }
    for _ in range(3):
        loss, _info = model.optim_step(batch)

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
