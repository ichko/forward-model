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
        rnn_num_layers,
        num_actions,
        action_embedding_size,
        rnn_hidden_size,
        recurrent_skip,
        precondition_type,
    ):
        super().__init__(
            action_embedding_size,
            rnn_hidden_size,
            rnn_num_layers,
            recurrent_skip,
            precondition_type,
        )

        self.name = 'RNN Dense'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames

        # self.precondition_encoder = nn.Sequential(
        #     tu.dense(i=2, o=128),
        #     nn.BatchNorm1d(128),
        #     tu.dense(i=128, o=rnn_hidden_size),
        #     nn.BatchNorm1d(rnn_hidden_size),
        # )

        self.precondition_encoder = nn.Sequential(
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

        precondition_map = self.precondition_encoder(precondition)
        action_vectors = self.action_embedding(actions)
        rnn_out_vectors, _ = self.rnn(action_vectors, precondition_map)
        frames = self.deconvolve_to_frame(rnn_out_vectors)

        return frames


def make_model(config):
    return Model(
        num_precondition_frames=config['precondition_size'],
        frame_size=config['frame_size'],
        rnn_num_layers=config['rnn_num_layers'],
        num_actions=config['num_actions'],
        action_embedding_size=config['action_embedding_size'],
        rnn_hidden_size=config['hidden_size'],
        recurrent_skip=config['recurrent_skip'],
        precondition_type=config['precondition_type'],
    )


def sanity_check():
    base_sanity_check(lambda: make_model(dict(
        precondition_size=2,
        frame_size=(32, 32),
        num_actions=3,
        rnn_num_layers=2,
        action_embedding_size=32,
        hidden_size=32,
        recurrent_skip=8,
    )))


if __name__ == '__main__':
    sanity_check()
