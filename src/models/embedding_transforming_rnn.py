import src.utils.torch as tu

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class RNNCell(nn.Module):
    def __init__(self, hidden, num_actions):
        super().__init__()

        self.update = nn.Sequential(
            tu.KernelEmbedding(
                num_actions,
                ks=5,
                channels=[hidden, 64, hidden],
            ),
            nn.Sigmoid(),
        )

        self.new_state = nn.Sequential(
            tu.KernelEmbedding(
                num_actions,
                ks=5,
                channels=[hidden, 32, hidden],
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        action, state = x

        update = self.update([state, action])
        new_state = self.new_state([state, action])
        transformed_state = state * update + new_state * (1 - update)

        return transformed_state


class Model(tu.BaseModule):
    def __init__(self, frame_size, num_actions, num_preconditions, state_size):
        super().__init__()
        self.name = 'Embedding Transforming RNN'

        self.frame_size = frame_size
        self.precondition_channels = 3 * num_preconditions
        self.num_preconditions = num_preconditions
        self.state_size = state_size

        self.precondition_encoder = nn.Sequential(
            tu.conv_transform(sizes=(
                self.precondition_channels,
                32,
                state_size,
            )),
            nn.Tanh(),
        )

        self.state_to_frame = tu.time_distribute(
            nn.Sequential(
                tu.conv_transform(sizes=(state_size, 3)),
                nn.Sigmoid(),
            ))

        self.cell = RNNCell(state_size, num_actions)

    def _forward(self, x):
        actions, frames = x
        frames = T.FloatTensor(frames).to(self.device)
        actions = T.LongTensor(actions).to(self.device)

        precondistions = frames[:, :self.num_preconditions]
        precondistions = precondistions.view(
            -1,
            self.precondition_channels,
            *self.frame_size[::-1],
        )

        actions = actions[:, self.num_preconditions - 1:]

        state = self.precondition_encoder(precondistions)

        bs = actions.size(0)
        seq_len = actions.size(1)  # seq dim

        out_states = T.zeros(
            bs,
            seq_len,
            self.state_size,
            *self.frame_size[::-1],
        ).to(self.device)

        for i in range(seq_len):
            # Check for NaNs
            if T.sum(T.isnan(state)) > 0:
                print('AAA')
            action = actions[:, i]
            state = self.cell([action, state])
            out_states[:, i] = state

        out_frames = T.sigmoid(out_states[:, :, -3:])

        return out_frames

    def forward(self, x):
        if not self.training:
            return self.rollout(x)

        return self._forward(x)

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        terminals = T.BoolTensor(terminals) \
            .to(self.device)[:, self.num_preconditions:]

        y_pred = self([actions[:, :-1], observations[:, :-1]])
        y_pred = tu.mask_sequence(y_pred, ~terminals)

        y_true = T.FloatTensor(observations).to(
            self.device, )[:, self.num_preconditions:]

        loss = F.binary_cross_entropy(y_pred, y_true)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
            self.optim.step()

        return loss, {'y': y_true, 'y_pred': y_pred}

    def configure_optim(self, lr):
        self.optim = T.optim.Adam(self.parameters(), lr=lr)


def make_model(precondition_size, frame_size, num_actions):
    return Model(
        frame_size=frame_size,
        num_actions=num_actions,
        num_preconditions=precondition_size,
        state_size=8,
    )


def sanity_check():
    num_precondition_frames = 2
    frame_size = (32, 32)
    num_actions = 3
    max_seq_len = 64
    bs = 32

    model = make_model(
        num_precondition_frames,
        frame_size,
        num_actions,
    ).to('cuda')

    print(model.summary())

    frames = T.randint(
        0,
        255,
        size=(bs, max_seq_len, 3, *frame_size),
    ) / 255.0
    actions = T.randint(0, num_actions, size=(bs, max_seq_len))
    out_frames = model([actions, frames]).detach().cpu()
    terminals = T.rand(bs, max_seq_len) > 0.5

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    model.configure_optim(lr=0.0001)
    batch = {
        'actions': actions,
        'observations': frames,
        'terminals': terminals,
    }
    loss, _info = model.optim_step(batch)
    loss, _info = model.optim_step(batch)

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
