import src.utils.torch as tu

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAutoEncoder(tu.BaseModule):
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
        self.name = 'RNN Auto Encoder'

        self.precondition_size = precondition_size
        self.frame_size = frame_size
        self.num_rnn_layers = num_rnn_layers

        self.precondition_encoder = nn.Sequential(
            nn.Flatten(),
            tu.dense(i=16 * 16 * 3 * precondition_size, o=512),
            tu.dense(i=512, o=rnn_hidden_size * num_rnn_layers),
        )

        self.unit_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            tu.dense(i=16 * 16 * 3, o=512),
            tu.dense(i=512, o=frame_encoding_size, a=nn.Tanh()),
        )
        self.frames_encoder = tu.time_distribute_31D(self.unit_encoder)

        self.unit_decoder = nn.Sequential(
            tu.dense(i=rnn_hidden_size, o=512),
            tu.dense(i=512, o=16 * 16 * 3, a=nn.Sigmoid()),
            tu.reshape(-1, 3, 16, 16),
        )
        self.frames_decoder = tu.time_distribute_13D(self.unit_decoder)

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

        actions = torch.LongTensor(actions).to(self.device)
        frames = torch.FloatTensor(frames / 255.0).to(self.device)

        precondition = frames[:, :self.precondition_size]
        frames = frames[:, self.precondition_size:]
        actions = actions[:, self.precondition_size:]

        precondition_map = self.precondition_encoder(precondition)
        precondition_map = tu.prepare_rnn_state(
            precondition_map,
            torch.tensor(self.num_rnn_layers),
        )

        actions_map = self.action_embeddings(actions)
        frames_map = self.frames_encoder(frames)
        action_frame_pair = torch.cat([actions_map, frames_map], dim=-1)

        rnn_out_vectors, _ = self.rnn(action_frame_pair, precondition_map)
        frames = self.frames_decoder(rnn_out_vectors)

        return frames

    def forward(self, x):
        """
        x -> (frames, actions)
            frames  -> [bs, sequence, 3, H, W]
            actions -> [bs, sequence]
        """
        if not self.training:
            return self.rollout(x)

        return self._forward(x)

    def rollout(self, x):
        """Recursive feeding of self generated frames"""
        actions, frames = x
        frames = torch.FloatTensor(frames).to(self.device)
        seq_len = frames.size(1)

        for i in range(seq_len - self.precondition_size):
            pred_frames = self._forward([actions, frames])
            frames[:, i + self.precondition_size] = pred_frames[:, i] * 255

        return frames[:, self.precondition_size:] / 255.0

    def optim_step(self, batch):
        actions, frames, dones = batch

        dones = torch.BoolTensor(dones).to(
            self.device, )[:, self.precondition_size:]

        self.optim.zero_grad()
        y_pred = self([actions, frames])
        y_pred = tu.mask_sequence(y_pred, ~dones)
        y_true = torch.FloatTensor(frames / 255.0).to(
            self.device, )[:, self.precondition_size:]

        loss = F.binary_cross_entropy(y_pred, y_true)

        if loss.requires_grad:
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return loss, {'y': y_true, 'y_pred': y_pred}

    def configure_optim(self, lr):
        # LR should be 1. The actual LR comes from the scheduler.
        self.optim = torch.optim.Adam(self.parameters(), lr=1)

        def lr_lambda(it):
            return lr / (it // 20000 + 1)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )


def make_model(precondition_size, frame_size, num_actions):
    return RNNAutoEncoder(
        precondition_size=precondition_size,
        frame_size=frame_size,
        num_rnn_layers=2,
        num_actions=num_actions,
        action_embedding_size=64,
        rnn_hidden_size=256,
        frame_encoding_size=256,
    )


def sanity_check():
    num_precondition_frames = 2
    frame_size = (16, 16)
    num_actions = 3
    max_seq_len = 15
    bs = 10

    rnn = make_model(
        num_precondition_frames,
        frame_size,
        num_actions,
    ).to('cuda')

    print(f'RNN NUM PARAMS {rnn.count_parameters():08,}')

    frames = torch.randint(
        0,
        255,
        size=(bs, max_seq_len, 3, *frame_size),
    )
    actions = torch.randint(0, num_actions, size=(bs, max_seq_len))
    out_frames = rnn([actions, frames]).detach().cpu()
    dones = torch.rand(bs, max_seq_len) > 0.5

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    rnn.configure_optim(lr=0.001)
    loss, _info = rnn.optim_step([actions, frames, dones])

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
