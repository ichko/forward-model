import src.utils.torch as tu

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelRegressor(tu.BaseModule):
    def __init__(
        self,
        num_preconditions,
        frame_size,
        num_actions,
        action_embedding_size,
        precondition_flat_size,
    ):
        super().__init__()
        self.name = 'Kernel Regressor'

        self.precondition_channels = 3 * num_preconditions
        self.frame_size = frame_size

        self.compute_precondition = nn.Sequential(
            tu.conv_block(
                i=self.precondition_channels,
                o=16,
                ks=5,
                s=2,
                p=2,
                d=1,
            ),
            tu.conv_block(i=16, o=32, ks=5, s=2, p=2, d=1),
            tu.conv_block(i=32, o=4, ks=7, s=1, p=2, d=2),
        )

        self.precondition_out = tu.compute_conv_output(
            self.compute_precondition,
            (3, *frame_size),
        )
        self.flat_precondition_size = np.prod(self.precondition_out[-3:])

        self.precondition_frame_to_vec = nn.Sequential(
            nn.Flatten(),
            tu.dense(
                self.flat_precondition_size,
                precondition_flat_size,
            ),
        )

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.action_precondition_to_kernels = nn.Sequential(
            tu.dense(i=precondition_flat_size + action_embedding_size, o=256),
            tu.dense(i=256, o=2048, a=nn.Tanh()),
            tu.reshape(-1, 2, 8, 8, 4, 4),
        )

        self.current_frame_map = nn.Sequential(
            tu.conv_block(i=3, o=16, ks=3, s=1, p=1, d=1),
            tu.conv_block(i=16, o=32, ks=5, s=1, p=2, d=2),
            tu.conv_block(i=32, o=32, ks=5, s=1, p=2, d=2),
            tu.conv_block(i=32, o=8, ks=7, s=1, p=2, d=2),
        )

        self.expand_transformed = nn.Sequential(
            tu.deconv_block(i=8, o=16, ks=3, s=1, p=1, d=2),
            tu.deconv_block(i=16, o=16, ks=5, s=1, p=2, d=2),
            tu.deconv_block(i=16, o=16, ks=7, s=1, p=2, d=2),
            tu.deconv_block(i=16, o=3, ks=7, s=1, p=2, d=2, a=nn.Sigmoid()),
        )

    def forward(self, x):
        """
        x -> (precondition_frames, actions)
            precondition_frames -> [bs, num_preconditions, 3, H, W]
            actions             -> [bs]
        """
        precondition_frames, actions = x
        precondition_frames = torch.FloatTensor(precondition_frames) \
            .to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        current_frame = precondition_frames[:, 0]
        precondition_frames = precondition_frames.reshape(
            -1,
            self.precondition_channels,
            *self.frame_size[::-1],  # W, H -> H, W
        )

        current_frame_feature_map = self.current_frame_map(current_frame)
        precondition_map = self.compute_precondition(precondition_frames)
        precondition_vec = self.precondition_frame_to_vec(precondition_map)
        action_vec = self.action_embedding(actions)
        precondition_action = torch.cat([precondition_vec, action_vec], dim=1)

        transformation_kernels = self.action_precondition_to_kernels(
            precondition_action, )
        kernel_1, kernel_2 = torch.chunk(
            transformation_kernels,
            chunks=2,
            dim=1,
        )
        kernel_1, kernel_2 = [t[:, 0] for t in [kernel_1, kernel_2]]

        transformed_frame = tu.batch_conv(current_frame_feature_map, kernel_1)
        transformed_frame = F.relu(transformed_frame)
        transformed_frame = tu.batch_conv(transformed_frame, kernel_2)
        transformed_frame = torch.tanh(transformed_frame)

        pred_future_frame = self.expand_transformed(transformed_frame)

        return pred_future_frame

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

    def rollout(self, preconditions, actions):
        """
        precondition -> [num_preconditions, 3, H, W]
        actions -> [sequence_of_ids]
        """
        frame_sequence = [f for f in preconditions]
        for action in actions:
            future_frame = self(
                [[preconditions], [action]], )[0].detach().cpu().numpy() * 255
            preconditions = np.roll(preconditions, shift=-1, axis=0)
            preconditions[-1] = future_frame
            frame_sequence.append(future_frame)

        # the last frame is not rendered
        return np.array(frame_sequence[:-1]) / 255.0


def make_model(num_precondition_frames, frame_size, num_actions):
    return KernelRegressor(
        num_preconditions=num_precondition_frames,
        frame_size=frame_size,
        num_actions=num_actions,
        action_embedding_size=32,
        precondition_flat_size=512,
    )


def sanity_check():
    num_precondition_frames = 1
    frame_size = (64, 64)
    num_actions = 3
    bs = 10

    model = make_model(
        num_precondition_frames=num_precondition_frames,
        frame_size=frame_size,
        num_actions=num_actions,
    ).to('cuda')

    print(f'RNN NUM PARAMS {model.count_parameters():08,}')
    print(
        f'PRECONDITION FEATURE MAP {model.precondition_out} [{model.flat_precondition_size}]'
    )

    precondition_frames = torch.rand(
        bs,
        num_precondition_frames,
        3,
        *frame_size,
    )
    actions = torch.randint(0, num_actions, size=(bs, ))
    out_frames = model([precondition_frames, actions]).detach().cpu()

    print(f'OUT FRAMES SHAPE {out_frames.shape}')

    model.configure_optim(lr=0.001)
    loss, _info = model.optim_step(
        [[precondition_frames, actions], out_frames], )

    print(f'OPTIM STEP LOSS {loss.item()}')


if __name__ == '__main__':
    sanity_check()
