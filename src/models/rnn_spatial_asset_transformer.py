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
        self.name = 'RNN Spatial Asset Transformer'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_precondition_frames = num_precondition_frames
        self.num_rnn_layers = num_rnn_layers

        self.direction_precondition = nn.Sequential(
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

        # self.frame_precondition = nn.Sequential(
        #     tu.cat_channels(),
        #     tu.conv_to_flat(
        #         input_size=frame_size,
        #         channel_sizes=[self.precondition_channels, 16, 32, 64, 8],
        #         ks=4,
        #         s=1,
        #         out_size=rnn_hidden_size,
        #     ),
        # )

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

        # If precondition with frames
        if len(precondition.shape) == 5:  # (bs, num_frames, 3, H, W)
            precondition_map = self.frame_precondition(precondition)
        # else preconditioned with direction (from PONG)
        else:
            precondition_map = self.direction_precondition(precondition)

        action_vectors = self.action_embedding(actions)
        rnn_out_vectors, _ = self.rnn(action_vectors, precondition_map)

        assets = self.assets.repeat(bs, seq_len, 1, 1, 1)
        pred_frames = self.transform_frame([rnn_out_vectors, assets])
        self.transformed_assets = pred_frames.clone()

        pred_frames = pred_frames.mean(dim=2, keepdim=True)
        pred_frames = T.sigmoid(pred_frames)
        pred_frames = pred_frames.repeat(1, 1, 3, 1, 1)

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
        action_embedding_size=16,
        rnn_hidden_size=32,
        recurrent_skip=4,
    )


def sanity_check():
    base_sanity_check(lambda: make_model(
        precondition_size=2,
        frame_size=(32, 32),
        num_actions=3,
    ))


if __name__ == '__main__':
    sanity_check()
