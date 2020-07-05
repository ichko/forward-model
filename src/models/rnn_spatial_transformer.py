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
    ):
        super().__init__(
            action_embedding_size,
            rnn_hidden_size,
            rnn_num_layers,
            recurrent_skip,
        )
        self.name = 'RNN Spatial Transformer'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames

        self.precondition_encoder = nn.Sequential(
            tu.dense(i=2, o=64),
            nn.BatchNorm1d(64),
            tu.dense(i=64, o=rnn_hidden_size),
            nn.BatchNorm1d(rnn_hidden_size),
        )

        # self.precondition_encoder = nn.Sequential(
        #     tu.reshape(-1, self.precondition_channels * 32 * 32),
        #     tu.dense(i=self.precondition_channels * 32 * 32, o=64),
        #     nn.BatchNorm1d(64),
        #     tu.dense(i=64, o=rnn_hidden_size),
        #     nn.BatchNorm1d(rnn_hidden_size),
        # )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        num_affine_channels = 3
        self.frame_feature_extract = tu.time_distribute(
            nn.Sequential(
                nn.Dropout(0.6),
                tu.conv_block(i=3, o=32, ks=3, s=1, p=0),
                tu.conv_block(i=32, o=32, ks=5, s=1, p=0),
                tu.conv_block(i=32, o=32, ks=5, s=1, p=0),
                tu.conv_block(i=32, o=num_affine_channels, ks=7, s=1, p=0),
            ))

        self.transform_frame = tu.time_distribute(
            nn.Sequential(
                tu.spatial_transformer(
                    i=rnn_hidden_size,
                    num_channels=num_affine_channels,
                ),
                tu.deconv_block(i=num_affine_channels, o=16, ks=7, s=1, p=0),
                tu.deconv_block(i=16, o=16, ks=5, s=1, p=0),
                tu.deconv_block(i=16, o=16, ks=5, s=1, p=0),
                tu.deconv_block(i=16, o=3, ks=3, s=1, p=0, a=nn.Sigmoid()),
            ))

    def forward(self, x):
        actions, precondition, frames = x

        frames = T.FloatTensor(frames).to(self.device)
        precondition = T.FloatTensor(precondition).to(self.device)
        actions = T.LongTensor(actions).to(self.device)

        precondition_map = self.precondition_encoder(precondition)
        action_vectors = self.action_embedding(actions)
        rnn_out_vectors, _ = self.rnn(action_vectors, precondition_map)
        frames = self.frame_feature_extract(frames)
        pred_frames = self.transform_frame([rnn_out_vectors, frames])

        return pred_frames

def make_model(config):
    return Model(
        num_precondition_frames=config['precondition_size'],
        frame_size=config['frame_size'],
        rnn_num_layers=config['rnn_num_layers'],
        num_actions=config['num_actions'],
        action_embedding_size=config['action_embedding_size'],
        rnn_hidden_size=config['hidden_size'],
        recurrent_skip=config['recurrent_skip'],
    )


def sanity_check():
    base_sanity_check(lambda: make_model(dict(
        precondition_size=2,
        frame_size=(32, 32),
        num_actions=3,
        rnn_num_layers=2,
        action_embedding_size=16,
        hidden_size=32,
        recurrent_skip=4,
    )))


if __name__ == '__main__':
    sanity_check()
