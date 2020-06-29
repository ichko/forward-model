import src.utils.nn as tu
import src.utils.drnn as drnn

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.models.rnn_base import RNNBase, base_sanity_check


class Model(RNNBase):
    def __init__(
        self,
        num_precondition_frames,
        frame_size,
        num_rnn_layers,
        num_actions,
        action_embedding_size,
        rnn_hidden_size,
        recurrent_skip,
    ):
        super().__init__(
            action_embedding_size,
            rnn_hidden_size,
            num_rnn_layers,
            recurrent_skip,
        )

        self.name = 'RNN Dense'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames
        self.num_rnn_layers = num_rnn_layers

        # self.direction_precondition = nn.Sequential(
        #     tu.dense(i=1, o=128),
        #     nn.BatchNorm1d(128),
        #     tu.dense(i=128, o=rnn_hidden_size),
        #     nn.BatchNorm1d(rnn_hidden_size),
        # )

        self.frame_precondition = nn.Sequential(
            tu.reshape(-1, self.precondition_channels * 32 * 32),
            tu.dense(i=self.precondition_channels * 32 * 32, o=128),
            nn.BatchNorm1d(128),
            tu.dense(i=128, o=rnn_hidden_size),
            nn.BatchNorm1d(rnn_hidden_size),
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.deconvolve_to_frame = tu.time_distribute(
            nn.Sequential(
                tu.dense(i=rnn_hidden_size, o=128),
                nn.BatchNorm1d(128),
                tu.dense(i=128, o=32 * 32 * 3, a=nn.Sigmoid()),
                tu.reshape(-1, 3, 32, 32),
            ))

    def forward(self, x):
        """
        x -> (actions, precondition)
            actions -> [bs, sequence]
            precondition  -> [bs, preconf_size, 3, H, W]
        return -> future frame
        """
        actions, precondition, _ = x

        precondition = T.FloatTensor(precondition).to(self.device)
        actions = T.LongTensor(actions).to(self.device)

        # If precondition with frames
        if len(precondition.shape) == 5:  # (bs, num_frames, 3, H, W)
            precondition_map = self.frame_precondition(precondition)
        # else preconditioned with direction (from PONG)
        else:
            precondition.unsqueeze_(1)
            precondition_map = self.direction_precondition(precondition)

        action_vectors = self.action_embedding(actions)

        rnn_out_vectors, _ = self.rnn(action_vectors, precondition_map)
        frames = self.deconvolve_to_frame(rnn_out_vectors)

        return frames

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        if 'meta' in batch and 'direction' in batch['meta']:
            precondition = batch['meta']['direction']
        else:
            precondition = observations[:, :self.num_precondition_frames]

        actions = actions[:, self.num_precondition_frames - 1:-1]
        observations = T.FloatTensor(observations).to(self.device)

        terminals = T.BoolTensor(terminals).to(self.device)
        terminals = terminals[:, self.num_precondition_frames:]

        y_true = observations[:, self.num_precondition_frames:]

        y_pred = self([actions, precondition, None])
        y_pred = tu.mask_sequence(y_pred, ~terminals)

        loss = F.binary_cross_entropy(y_pred, y_true)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return loss, {'y': y_true, 'y_pred': y_pred}


def make_model(precondition_size, frame_size, num_actions):
    return Model(
        num_precondition_frames=precondition_size,
        frame_size=frame_size,
        num_rnn_layers=2,
        num_actions=num_actions,
        action_embedding_size=32,
        rnn_hidden_size=32,
        recurrent_skip=8,
    )


def sanity_check():
    base_sanity_check(lambda: make_model(
        precondition_size=2,
        frame_size=(32, 32),
        num_actions=3,
    ))


if __name__ == '__main__':
    sanity_check()
