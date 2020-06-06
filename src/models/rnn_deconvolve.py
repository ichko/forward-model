import src.utils.torch as tu

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Model(tu.BaseModule):
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
        self.num_precondition_frames = num_precondition_frames
        self.num_rnn_layers = num_rnn_layers

        inp = self.precondition_channels
        self.compute_precondition = nn.Sequential(
            nn.Dropout(p=0.8),
            tu.conv_block(i=inp, o=32, ks=4, s=2, p=1, d=1),
            tu.conv_block(i=32, o=64, ks=4, s=2, p=1, d=1),
            nn.Dropout(p=0.5),
            tu.conv_block(i=64, o=64, ks=4, s=2, p=1, d=1),
            tu.conv_block(i=64, o=128, ks=4, s=2, p=1, d=1),
            nn.Dropout(p=0.5),
            tu.conv_block(i=128, o=256, ks=4, s=2, p=1, d=1, bn=False),
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
            dropout=0.6,
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.deconvolve_to_frame = tu.time_distribute(
            nn.Sequential(
                tu.dense(i=rnn_hidden_size, o=256),
                tu.reshape(-1, 256, 1, 1),
                nn.Dropout(p=0.5),
                tu.deconv_block(i=256, o=128, ks=5, s=2, p=1, d=1),
                tu.deconv_block(i=128, o=64, ks=5, s=2, p=1, d=1),
                nn.Dropout(p=0.3),
                tu.deconv_block(i=64, o=32, ks=5, s=2, p=2, d=2),
                tu.deconv_block(
                    i=32,
                    o=3,
                    ks=4,
                    s=2,
                    p=2,
                    d=1,
                    a=nn.Sigmoid(),
                    bn=False,
                ),
            ))

    def forward(self, x):
        """
        x -> (actions, precondition_frames)
            precondition_frames -> [bs, num_preconditions, 3, H, W]
            actions             -> [bs, sequence]
        return -> future frame
        """
        actions, precondition_frames = x

        actions = T.LongTensor(actions).to(self.device)
        precondition_frames = T.FloatTensor(precondition_frames) \
            .to(self.device)

        precondition_frames = precondition_frames.reshape(
            -1,
            self.precondition_channels,
            *self.frame_size[::-1],  # W, H -> H, W
        )

        rnn_preconditions = self.compute_precondition(precondition_frames)
        rnn_preconditions = self.precondition_frame_to_rnn(rnn_preconditions)
        rnn_preconditions = T.stack(
            rnn_preconditions.chunk(self.num_rnn_layers, dim=1),
            dim=0,
        )

        action_vectors = self.action_embedding(actions)
        rnn_out_vectors, _ = self.rnn(action_vectors, rnn_preconditions)
        frames = self.deconvolve_to_frame(rnn_out_vectors)

        return frames

    def render(self, _mode='rgb_array'):
        pred_frame = self.forward([[self.actions], [self.precondition]])
        pred_frame = pred_frame[0, -1]
        pred_frame = pred_frame.detach().cpu().numpy()
        return pred_frame

    def reset(self, precondition, precondition_actions):
        self.precondition = precondition
        self.actions = precondition_actions

        return self.render()

    def step(self, action):
        self.actions.append(action)
        pred_frame = self.render()

        return pred_frame

    def optim_step(self, batch):
        actions = batch['actions']
        observations = batch['observations']
        terminals = batch['terminals']

        precondition = observations[:, :self.num_precondition_frames]
        actions = actions[:, self.num_precondition_frames - 1:-1]

        y_true = T.FloatTensor(observations).to(
            self.device, )[:, self.num_precondition_frames:]
        terminals = T.BoolTensor(terminals).to(
            self.device, )[:, self.num_precondition_frames:]

        y_pred = self([actions, precondition])
        y_pred = tu.mask_sequence(y_pred, ~terminals)

        loss = F.binary_cross_entropy(y_pred, y_true)

        if loss.requires_grad:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return loss, {'y': y_true, 'y_pred': y_pred}

    def configure_optim(self, lr):
        # LR should be 1. The actual LR comes from the scheduler.
        self.optim = T.optim.Adam(self.parameters(), lr=1)

        def lr_lambda(it):
            return lr / (it // 20000 + 1)

        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )


def make_model(precondition_size, frame_size, num_actions):
    return Model(
        num_precondition_frames=precondition_size,
        frame_size=frame_size,
        num_rnn_layers=2,
        num_actions=num_actions,
        action_embedding_size=64,
        rnn_hidden_size=256,
    )


def sanity_check():
    num_precondition_frames = 2
    frame_size = (32, 32)
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

    precondition_frames = T.randint(
        0,
        255,
        size=(bs, num_precondition_frames, 3, *frame_size),
    ) / 255.0
    actions = T.randint(0, num_actions, size=(bs, max_seq_len))
    dones = T.rand(bs, max_seq_len) > 0.5
    out_frames = rnn([actions, precondition_frames]).detach().cpu()

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    rnn.configure_optim(lr=0.001)
    loss, _info = rnn.optim_step({
        'actions': actions,
        'observations': out_frames,
        'terminals': dones
    })

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
