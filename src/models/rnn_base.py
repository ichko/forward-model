import src.utils.nn as tu
import src.utils.drnn as drnn

import numpy as np

import torch as T
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_rnn_layers,
        recurrent_skip,
    ):
        super().__init__()

        self.initial_rnn = nn.GRU(
            input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dilated_rnn = drnn.dilate_rnn(
            cell=nn.GRU(
                hidden_size,
                hidden_size=hidden_size,
                num_layers=num_rnn_layers,
                batch_first=True,
            ),
            skip=recurrent_skip,
        )

    def forward(self, x, state):
        # This is done because the first dimension reflects the number of rnn layer
        state = state.unsqueeze(0)
        x, _ = self.initial_rnn(x, state)
        return self.dilated_rnn(x)


class RNNBase(tu.BaseModule):
    def __init__(
        self,
        action_embedding_size,
        rnn_hidden_size,
        num_rnn_layers,
        recurrent_skip=1,
    ):
        super().__init__()

        self.rnn = RNNCell(
            action_embedding_size,
            rnn_hidden_size,
            num_rnn_layers,
            recurrent_skip,
        )

    def render(self, _mode='rgb_array'):
        pred_frame = self.forward(
            [[self.actions], [self.precondition], [self.input_frames]], )
        pred_frame = pred_frame[0, -1]
        pred_frame = pred_frame.detach().cpu().numpy()
        self.input_frames.append(pred_frame)

        return pred_frame

    def reset(self, precondition, precondition_actions, input_frames):
        self.precondition = precondition
        self.actions = precondition_actions
        self.input_frames = input_frames

        return self.render()

    def step(self, action):
        self.actions.append(action)
        pred_frame = self.render()

        return pred_frame

    def configure_optim(self, lr):
        # LR should be 1. The actual LR comes from the scheduler.
        self.optim = T.optim.Adam(self.parameters(), lr=1)

        def lr_lambda(it):
            return lr / (it // 20000 + 1)

        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )


def base_sanity_check(make_model):
    frame_size = (32, 32)
    num_actions = 3
    max_seq_len = 32
    bs = 32

    model = make_model().to('cuda')

    print(model.summary())

    frames = T.randint(
        0,
        255,
        size=(bs, max_seq_len, 3, *frame_size),
    ) / 255.0
    actions = T.randint(0, num_actions, size=(bs, max_seq_len))
    terminals = T.rand(bs, max_seq_len) > 0.5
    direction = T.rand(bs)

    model.configure_optim(lr=0.0001)
    batch = {
        'meta': {
            # 'direction': direction
        },
        'actions': actions,
        'observations': frames,
        'terminals': terminals,
    }
    loss, _info = model.optim_step(batch)
    loss, _info = model.optim_step(batch)

    print(f'OPTIM STEP LOSS {loss.item()}')
