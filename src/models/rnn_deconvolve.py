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
        self.name = 'RNN Frame Deconvolve'

        self.frame_size = frame_size
        self.precondition_channels = num_precondition_frames * 3
        self.num_rnn_layers = num_rnn_layers

        self.compute_precondition = nn.Sequential(
            nn.Dropout2d(p=0.1),
            tu.conv_block(
                i=self.precondition_channels,
                o=16,
                ks=3,
                s=1,
                p=1,
                d=1,
            ),
            tu.conv_block(i=16, o=32, ks=5, s=1, p=2, d=1),
            nn.Dropout2d(p=0.05),
            tu.conv_block(
                i=32,
                o=4,
                ks=7,
                s=1,
                p=3,
                d=1,  # TODO: try with bigger dilation
            ),
        )

        self.precondition_out = tu.compute_conv_output(
            self.compute_precondition,
            (self.precondition_channels, *frame_size),
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

        self.deconvolve_to_frame = nn.Sequential(
            tu.dense(i=rnn_hidden_size, o=400),
            tu.reshape(-1, 4, 10, 10),
            tu.deconv_block(i=4, o=32, ks=5, s=1, p=1, d=1),
            nn.Dropout2d(p=0.1),
            tu.deconv_block(i=32, o=32, ks=7, s=1, p=3, d=1),
            tu.deconv_block(i=32, o=3, ks=7, s=1, p=1, d=1, a=nn.Sigmoid()),
        )

    def forward(self, x):
        """
        x -> (precondition_frames, actions)
            precondition_frames -> [bs, num_preconditions, 3, H, W]
            actions             -> [bs, sequence]
        return -> future frame
        """
        precondition_frames, actions = x
        precondition_frames = torch.FloatTensor(precondition_frames) \
            .to(self.device) / 255

        actions = torch.LongTensor(actions).to(self.device)

        precondition_frames = precondition_frames.reshape(
            -1,
            self.precondition_channels,
            *self.frame_size[::-1],  # W, H -> H, W
        )
        seq_len = actions.size(1)

        rnn_preconditions = self.compute_precondition(precondition_frames)
        rnn_preconditions = self.precondition_frame_to_rnn(rnn_preconditions)
        rnn_preconditions = torch.stack(
            rnn_preconditions.chunk(self.num_rnn_layers, dim=1),
            dim=0,
        )

        action_vectors = self.action_embedding(actions)
        rnn_out_vectors, _ = self.rnn(action_vectors, rnn_preconditions)
        # put sequence in batch dim to facilitate fast (time distributed) convolution
        rnn_out_vectors = rnn_out_vectors.reshape(
            -1,
            rnn_out_vectors.size(-1),
        )

        frames = self.deconvolve_to_frame(rnn_out_vectors)
        frames = frames.reshape(-1, seq_len, *frames.shape[-3:])

        return frames

    def configure_optim(self, lr):
        self.optim = torch.optim.Adam(self.parameters(), lr=1)

        def lr_lambda(it):
            return lr / (it // 1000 + 1)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )

    def optim_step(self, batch):
        x, y = batch

        self.optim.zero_grad()
        y_pred = self(x)
        y = torch.FloatTensor(y).to(self.device) / 255.0
        loss = F.mse_loss(y_pred, y)

        if loss.requires_grad:
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return loss, {'y': y, 'y_pred': y_pred}


def make_model(precondition_size, frame_size, num_actions):
    return RNN(
        num_precondition_frames=precondition_size,
        frame_size=frame_size,
        num_rnn_layers=3,
        num_actions=num_actions,
        action_embedding_size=32,
        rnn_hidden_size=128,
    )


def sanity_check():
    num_precondition_frames = 1
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
    print(
        f'PRECONDITION FEATURE MAP {rnn.precondition_out} [{rnn.flat_precondition_size}]'
    )

    precondition_frames = torch.rand(
        bs,
        num_precondition_frames,
        3,
        *frame_size,
    )
    actions = torch.randint(0, num_actions, size=(bs, max_seq_len))
    out_frames = rnn([precondition_frames, actions]).detach().cpu()

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    rnn.configure_optim(lr=0.001)
    loss, _info = rnn.optim_step([[precondition_frames, actions], out_frames])

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
