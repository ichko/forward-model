import src.utils.nn as tu
import src.utils.drnn as drnn

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Model(tu.BaseModule):
    def __init__(
        self,
        num_precondition_frames,
        frame_size,
        num_actions,
        action_embedding_size,
        hidden_size,
    ):
        super().__init__()
        self.name = 'Frame Transformer Dense'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames

        self.compute_precondition = nn.Sequential(
            nn.Dropout(0.65),
            tu.cat_channels(),
            tu.reshape(-1, self.precondition_channels * 32 * 32),
            tu.dense(i=self.precondition_channels * 32 * 32, o=64),
            nn.BatchNorm1d(64),
            tu.dense(i=64, o=64),
            nn.BatchNorm1d(64),
            tu.dense(i=64, o=hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.to_frame = nn.Sequential(
            tu.dense(i=hidden_size + action_embedding_size, o=128),
            nn.BatchNorm1d(128),
            tu.dense(i=128, o=128),
            nn.BatchNorm1d(128),
            tu.dense(i=128, o=32 * 32 * 3, a=nn.Sigmoid()),
            tu.reshape(-1, 3, 32, 32),
        )

    def forward(self, x):
        actions, precondition = x

        precondition = T.FloatTensor(precondition).to(self.device)
        actions = T.LongTensor(actions).to(self.device)

        action_vectors = self.action_embedding(actions)
        seq_len = action_vectors.size(1)
        bs = action_vectors.size(0)
        frames = T.zeros(bs, seq_len, 3, *self.frame_size[::-1])
        frames = frames.to(self.device)

        for i in range(seq_len):
            precondition_map = self.compute_precondition(precondition)
            action_vector = action_vectors[:, i]
            action_precondition_vec = T.cat(
                [precondition_map, action_vector],
                dim=1,
            )
            frame = self.to_frame(action_precondition_vec)

            frames[:, i] = frame
            precondition = precondition.roll(-1, dims=1)
            precondition[:, -1] = frame

        return frames

    def render(self, _mode='rgb_array'):
        pred_frame = self.forward([[self.actions], [self.precondition]])
        pred_frame = pred_frame[0, -1]
        pred_frame = pred_frame.detach().cpu().numpy()
        return pred_frame

    def reset(self, precondition, precondition_actions, _):
        self.precondition = precondition
        self.actions = precondition_actions

        return self.render()

    def step(self, action):
        self.actions.append(action)
        pred_frame = self.render()

        return pred_frame

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        actions = actions[:, self.num_precondition_frames - 1:-1]

        precondition = observations[:, :self.num_precondition_frames]
        observations = T.FloatTensor(observations).to(self.device)

        terminals = T.BoolTensor(terminals).to(self.device)
        terminals = terminals[:, self.num_precondition_frames:]

        y_true = observations[:, self.num_precondition_frames:]

        y_pred = self([actions, precondition])
        y_pred = tu.mask_sequence(y_pred, ~terminals)

        loss = F.binary_cross_entropy(y_pred, y_true)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return loss, {'y': y_true, 'y_pred': y_pred}

    def configure_optim(self, lr):
        # LR should be 1. The actual LR comes from the scheduler.
        self.optim = T.optim.Adam(self.parameters(), lr=1)

        def lr_lambda(it):
            return lr / (it // 20000 + 1)

        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )


def make_model(precondition_size, frame_size, num_actions):
    return Model(
        num_precondition_frames=precondition_size,
        frame_size=frame_size,
        num_actions=num_actions,
        action_embedding_size=32,
        hidden_size=64,
    )


def sanity_check():
    num_precondition_frames = 1
    frame_size = (32, 32)
    num_actions = 3
    max_seq_len = 32
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
    terminals = T.rand(bs, max_seq_len) > 0.5

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
