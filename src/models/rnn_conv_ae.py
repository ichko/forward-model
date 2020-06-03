import src.utils.torch as tu

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Model(tu.BaseModule):
    def __init__(
        self,
        frame_size,
        num_rnn_layers,
        num_actions,
        action_embedding_size,
        rnn_hidden_size,
        precondition_size,
        frame_encoding_size,
    ):
        super().__init__()
        self.name = 'RNN Conv Auto Encoder'

        self.precondition_size = precondition_size
        self.precondition_channels = self.precondition_size * 3
        self.frame_size = frame_size
        self.num_rnn_layers = num_rnn_layers

        self.precondition_encoder = nn.Sequential(
            tu.conv_encoder(sizes=(6, 32, 64, 32)),
            tu.reshape(-1, 32 * 4 * 4),
            tu.dense(i=32 * 4 * 4, o=rnn_hidden_size * num_rnn_layers, a=None),
        )

        self.frames_encoder = tu.time_distribute(
            nn.Sequential(
                nn.Dropout2d(p=0.5),
                tu.conv_encoder(sizes=(3, 32, 64, 32)),
                tu.reshape(-1, 32 * 4 * 4),
                tu.dense(i=32 * 4 * 4, o=frame_encoding_size, a=None),
            ))

        self.frames_decoder = tu.time_distribute(
            nn.Sequential(
                nn.Dropout(p=0.1),
                tu.dense(i=rnn_hidden_size, o=32 * 4 * 4),
                tu.reshape(-1, 32, 4, 4),
                tu.conv_decoder(sizes=(32, 64, 32, 3)),
                nn.Sigmoid(),
            ))

        self.rnn = nn.GRU(
            action_embedding_size + frame_encoding_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        self.action_embeddings = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

    def _forward(self, x):
        actions, frames = x

        actions = T.LongTensor(actions).to(self.device)
        frames = T.FloatTensor(frames).to(self.device)

        precondition = frames[:, :self.precondition_size]
        precondition = precondition.view(
            -1,
            self.precondition_channels,
            *self.frame_size,
        )
        frames = frames[:, self.precondition_size - 1:]
        actions = actions[:, self.precondition_size - 1:]

        precondition_map = self.precondition_encoder(precondition)
        precondition_map = tu.prepare_rnn_state(
            precondition_map,
            T.tensor(self.num_rnn_layers),
        )

        actions_map = self.action_embeddings(actions)
        frames_map = self.frames_encoder(frames)
        action_frame_pair = T.cat([actions_map, frames_map], dim=-1)

        rnn_out_vectors, _ = self.rnn(action_frame_pair, precondition_map)
        pred_frames = self.frames_decoder(rnn_out_vectors)

        return pred_frames

    def forward(self, x):
        if not self.training:
            return self.rollout(x)

        return self._forward(x)

    def rollout(self, x):
        """Recursive feeding of self generated frames"""
        actions, frames = x
        frames = T.FloatTensor(frames).to(self.device)
        seq_len = frames.size(1)

        for i in range(seq_len - self.precondition_size):
            pred_frames = self._forward([actions, frames])
            frames[:, i + self.precondition_size] = pred_frames[:, i]

        return frames[:, self.precondition_size:]

    def reset(self, precondition):
        self.frames = precondition.tolist()
        self.actions = [0] * self.precondition_size

    def step(self, action):
        self.actions.append(action)
        pred_frame = self._forward([[self.actions], [self.frames]])[0, -1]
        pred_frame = pred_frame.cpu().numpy()
        self.frames.append(pred_frame)

        return pred_frame

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        terminals = T.BoolTensor(terminals) \
            .to(self.device)[:, self.precondition_size:]

        y_pred = self([actions, observations])
        y_pred = y_pred[:, :-1]  # we don't have label for the last frame
        y_pred = tu.mask_sequence(y_pred, ~terminals)

        y_true = T.FloatTensor(observations).to(
            self.device, )[:, self.precondition_size:]

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
        precondition_size=precondition_size,
        frame_size=frame_size,
        num_rnn_layers=1,
        num_actions=num_actions,
        action_embedding_size=32,
        rnn_hidden_size=200,
        frame_encoding_size=128,
    )


def sanity_check():
    num_precondition_frames = 2
    frame_size = (32, 32)
    num_actions = 3
    max_seq_len = 15
    bs = 10

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

    model.configure_optim(lr=0.001)
    loss, _info = model.optim_step({
        'actions': actions,
        'observations': frames,
        'terminals': terminals,
    })

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
