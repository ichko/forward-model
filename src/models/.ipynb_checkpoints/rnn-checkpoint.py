import src.utils.torch as tu

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(tu.BaseModule):
    """
    Frame preconditioned RNN. Structure:
        - input: (frame[s], actions)
        - frame -> representation vector E
        - E -> place in state of rnn
        - actions + rnn -> output
        - output -> de_convolve -> frame sequence
    Model assumes that the first frame[s] contains all the information for the
    initialization of the environment. Example - snake, pong, breakout
    """
    def __init__(
        self,
        num_precondition_frames,
        frame_size,
        num_rnn_layers,
        num_actions,
        action_embedding_size,
        rnn_hidden_size,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3

        self.compute_precondition = nn.Sequential(
            tu.conv_block(i=self.precondition_channels, o=64, ks=5, s=2, p=2, d=1),
            tu.conv_block(i=64, o=64, ks=5, s=1, p=2, d=1),
            tu.conv_block(i=64, o=64, ks=7, s=1, p=2, d=1),
            tu.conv_block(i=64, o=8, ks=9, s=1, p=2, d=2),
        )

        self.precondition_out = tu.compute_conv_output(
            self.compute_precondition,
            (3, *frame_size),
        )
        self.flat_precondition_size = np.prod(self.precondition_out[-3:])

        self.precondition_frame_to_rnn = nn.Sequential(
            nn.Flatten(),
            tu.dense(
                self.flat_precondition_size,
                rnn_hidden_size * num_rnn_layers,
            ),
        )

        self.rnn = nn.GRU(
            action_embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )
        
    def forward(self, x):
        """
        x -> (precondition_frames, actions)
            precondition_frames -> [bs, num_preconditions, 3, H, W]
            actions             -> [bs, sequence, action_id]
        """
        precondition_frames, actions = x
        precondition_frames = precondition_frames.reshape(
            -1,
            self.precondition_channels,
            *self.frame_size[::-1],  # W, H -> H, W
        )
        
        rnn_preconditions = self.compute_precondition(precondition_frames)
        rnn_preconditions = self.precondition_frame_to_rnn(rnn_preconditions)
        
        return rnn_preconditions


def sanity_check():
    num_precondition_frames = 1
    frame_size = (32, 32)
    num_actions = 3
    max_seq_len = 32
    bs = 10
    
    rnn = RNN(
        num_precondition_frames=num_precondition_frames,
        frame_size=frame_size,
        num_rnn_layers=3,
        num_actions=num_actions,
        action_embedding_size=32,
        rnn_hidden_size=32,
    )

    print(f'RNN NUM PARAMS {rnn.count_parameters():08,}')
    print(f'PRECONDITION FEATURE MAP {rnn.precondition_out} [{rnn.flat_precondition_size}]')
    
    precondition_frames = torch.rand(bs, num_precondition_frames, 3, *frame_size)
    actions = torch.randint(0, num_actions, size=(bs, max_seq_len, 1))
    out_frames = rnn([precondition_frames, actions])
    
    print(f'OUT FRAMES SHAPE {out_frames.shape}')


if __name__ == '__main__':
    sanity_check()
