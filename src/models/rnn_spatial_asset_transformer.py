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
        self.name = 'RNN Spatial Asset Transformer'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames

        # self.precondition_encoder = nn.Sequential(
        #     tu.cat_channels(),
        #     tu.conv_to_flat(
        #         input_size=frame_size,
        #         channel_sizes=[self.precondition_channels, 16, 32, 64, 8],
        #         ks=4,
        #         s=1,
        #         out_size=rnn_hidden_size,
        #     ),
        # )

        self.precondition_encoder = nn.Sequential(
            tu.dense(i=2, o=128),
            nn.BatchNorm1d(128),
            tu.dense(i=128, o=rnn_hidden_size),
            nn.BatchNorm1d(rnn_hidden_size),
        )

        self.num_assets = 3
        self.initialized = False
        self.assets = nn.Parameter(
            T.randn(1, 1, self.num_assets, *self.frame_size[::-1]))
        self.assets.requires_grad = True

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.transform_frame = tu.time_distribute(
            nn.Sequential(
                tu.spatial_transformer(
                    i=rnn_hidden_size,
                    num_channels=self.num_assets,
                ),
                # tu.conv_block(i=self.num_assets, o=16, ks=7, s=1, p=3),
                # tu.conv_block(i=16, o=3, ks=5, s=1, p=2, a=nn.Sigmoid()),
            ))

    def forward(self, x):
        actions, precondition, frames = x

        frames = T.FloatTensor(frames).to(self.device)
        precondition = T.FloatTensor(precondition).to(self.device)
        actions = T.LongTensor(actions).to(self.device)
        bs = actions.size(0)
        seq_len = actions.size(1)

        # if not self.initialized:
        #     self.initialized = True
        #     f = frames[0, 0, 0].clone()
        #     delim = 6
        #     self.assets.data[0, 0, 0][:, :delim] = f[:, :delim]
        #     self.assets.data[0, 0, 1][:, -delim:] = f[:, -delim:]
        #     self.assets.data[0, 0, 2][:, delim:-delim] = f[:, delim:-delim]

        precondition_map = self.precondition_encoder(precondition)

        action_vectors = self.action_embedding(actions)
        rnn_out_vectors, _ = self.rnn(action_vectors, precondition_map)

        assets = self.assets.repeat(bs, seq_len, 1, 1, 1)
        pred_frames = self.transform_frame([rnn_out_vectors, assets])
        self.transformed_assets = pred_frames

        pred_frames = pred_frames.sum(dim=2, keepdim=True)
        pred_frames = T.sigmoid(pred_frames)
        pred_frames = pred_frames.repeat(1, 1, 3, 1, 1)

        # mi, ma = pred_frames.min(), pred_frames.max()
        # pred_frames = (pred_frames - mi) / (ma - mi)

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
        precondition_type=config['precondition_type'],
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
