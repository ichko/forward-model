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
        self.name = 'RNN Spatial Transformer'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames
        self.num_rnn_layers = num_rnn_layers

        self.direction_precondition = nn.Sequential(
            tu.dense(i=1, o=128),
            nn.BatchNorm1d(128),
            tu.dense(i=128, o=rnn_hidden_size),
            nn.BatchNorm1d(rnn_hidden_size),
        )

        # self.frame_precondition = nn.Sequential(
        #     tu.reshape(-1, self.precondition_channels * 32 * 32),
        #     tu.dense(i=self.precondition_channels * 32 * 32, o=128),
        #     nn.BatchNorm1d(128),
        #     tu.dense(i=128, o=rnn_hidden_size),
        #     nn.BatchNorm1d(rnn_hidden_size),
        # )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.frame_feature_extract = tu.time_distribute(
            nn.Sequential(
                tu.conv_block(i=3, o=8, ks=7, s=1, p=3),
                tu.conv_block(i=8, o=8, ks=7, s=1, p=3),
            ))

        self.transform_frame = tu.time_distribute(
            nn.Sequential(
                tu.spatial_transformer(i=rnn_hidden_size, num_channels=8),
                tu.conv_block(i=8, o=8, ks=7, s=1, p=3),
                tu.conv_block(i=8, o=3, ks=7, s=1, p=3, a=nn.Sigmoid()),
            ))

    def forward(self, x):
        actions, precondition, frames = x

        frames = T.FloatTensor(frames).to(self.device)
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

        frames = self.frame_feature_extract(frames)
        pred_frames = self.transform_frame([rnn_out_vectors, frames])

        return pred_frames

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        if 'meta' in batch and 'direction' in batch['meta']:
            precondition = batch['meta']['direction']
        else:
            precondition = observations[:, :self.num_precondition_frames]

        actions = actions[:, self.num_precondition_frames - 1:-1]
        input_frames = observations[:, self.num_precondition_frames - 1:-1]

        terminals = T.BoolTensor(terminals).to(self.device)
        terminals = terminals[:, self.num_precondition_frames:]

        observations = T.FloatTensor(observations).to(self.device)
        y_true = observations[:, self.num_precondition_frames:]

        y_pred = self([actions, precondition, input_frames])
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
