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
    ):
        super().__init__()
        self.name = 'RNN Auto Encoder'

        self.frame_size = frame_size
        self.num_rnn_layers = num_rnn_layers

        self.encoder = nn.Sequential(
            nn.Flatten(),
            tu.dense(i=16 * 16 * 3, o=256),
            tu.dense(i=256, o=128, a=nn.Tanh()),
        )

        self.decoder = nn.Sequential(
            tu.dense(i=128, o=256),
            tu.dense(i=256, o=16 * 16 * 3, a=nn.Sigmoid()),
            tu.reshape(-1, 16, 16, 30),
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
        x -> (frames, actions)
            precondition_frames -> [bs, sequence, 3, H, W]
            actions             -> [bs, sequence]
        """
        frames, actions = x

        frames = torch.FloatTensor(frames / 255.0).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        first_frame = frames[:, 0]
        first_frame_enc = self.encoder(first_frame_enc)
        action_enc = self.action_embedding(actions)

        self.rnn(action_enc, )

        return frames

    def configure_optim(self, lr):
        # LR should be 1. The actual LR comes from the scheduler.
        self.optim = torch.optim.Adam(self.parameters(), lr=1)

        def lr_lambda(it):
            return lr / (it // 5000 + 1)

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
        num_rnn_layers=2,
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

    precondition_frames = torch.randint(
        0,
        255,
        size=(bs, num_precondition_frames, 3, *frame_size),
    )
    actions = torch.randint(0, num_actions, size=(bs, max_seq_len))
    out_frames = rnn([precondition_frames, actions]).detach().cpu()

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    rnn.configure_optim(lr=0.001)
    loss, _info = rnn.optim_step([[precondition_frames, actions], out_frames])

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
