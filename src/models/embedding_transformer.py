import src.utils.torch as tu

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTransformer(tu.BaseModule):
    def __init__(self, frame_size, num_actions, num_preconditions):
        super().__init__()
        self.name = 'Embedding Transformer'

        self.frame_size = frame_size
        self.precondition_channels = 3 * num_preconditions

        inp = self.precondition_channels
        self.compute_precondition = nn.Sequential(
            nn.Dropout(0.5),
            tu.conv_block(i=inp, o=64, ks=5, s=1, p=2, d=1),
            tu.conv_block(i=64, o=64, ks=5, s=1, p=2, d=1),
            tu.conv_block(
                i=64,
                o=64,
                ks=7,
                s=1,
                p=3,
                d=1,
                bn=False,
                a=None,
            ),
        )

        self.kernel_shapes = [
            (64, 64, 5, 5),
            (128, 64, 5, 5),
            (64, 128, 5, 5),
            (32, 64, 5, 5),
        ]

        self.batch_norms = nn.Sequential(
            *[nn.BatchNorm2d(k[0]) for k in self.kernel_shapes])

        self.kernels_flat = [np.prod(k) for k in self.kernel_shapes]

        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=np.sum(self.kernels_flat),
        )

        self.expand_transformed = nn.Sequential(
            nn.Dropout(0.1),
            tu.deconv_block(i=32, o=64, ks=3, s=1, p=1, d=1),
            tu.deconv_block(i=64, o=32, ks=5, s=1, p=2, d=1),
            tu.deconv_block(
                i=32,
                o=3,
                ks=7,
                s=1,
                p=3,
                d=1,
                a=nn.Sigmoid(),
                bn=False,
            ),
        )

    def forward(self, x):
        """
        x -> (precondition_frames, actions)
            precondition_frames -> [bs, 1, 3, H, W] - should be only one
            actions             -> [bs]
        """
        precondition_frames, actions = x
        precondition_frames = torch.FloatTensor(precondition_frames) \
            .to(self.device) / 255.0
        actions = torch.LongTensor(actions).to(self.device)

        precondition_frames = precondition_frames.reshape(
            -1,
            self.precondition_channels,
            *self.frame_size[::-1],  # W, H -> H, W
        )

        precondition_feature_map = self.compute_precondition(
            precondition_frames, )

        action_vec = self.action_embedding(actions)

        kernels = tu.extract_tensors(action_vec, self.kernel_shapes)

        transformed_frame = precondition_feature_map
        for i, k in enumerate(kernels):
            ks = self.kernel_shapes[i]
            transformed_frame = tu.batch_conv(
                transformed_frame,
                k,
                p=ks[-1] // 2,
            )
            # TODO Should we batch-norm?
            # transformed_frame = self.batch_norms[0](transformed_frame)
            transformed_frame = F.leaky_relu(
                transformed_frame,
                negative_slope=0.3,
            )

        pred_future_frame = self.expand_transformed(transformed_frame)

        return pred_future_frame

    def configure_optim(self, lr):
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

    def rollout(self, preconditions, actions):
        """
        precondition -> [num_preconditions, 3, H, W]
        actions      -> [sequence_of_ids]
        """
        frame_sequence = [f for f in preconditions]
        for action in actions:
            future_frame = self(
                [[preconditions], [action]], )[0].detach().cpu().numpy()
            future_frame = future_frame * 255
            preconditions = np.roll(preconditions, shift=-1, axis=0)
            preconditions[-1] = future_frame
            frame_sequence.append(future_frame)

        # the last frame is not rendered
        return np.array(frame_sequence[:-2]) / 255.0


def make_model(num_precondition_frames, frame_size, num_actions):
    return EmbeddingTransformer(
        frame_size=frame_size,
        num_actions=num_actions,
        num_preconditions=num_precondition_frames,
    )


def sanity_check():
    num_precondition_frames = 2
    frame_size = (32, 32)
    num_actions = 4
    bs = 64

    model = make_model(
        num_precondition_frames=num_precondition_frames,
        frame_size=frame_size,
        num_actions=num_actions,
    ).to('cuda')

    print(f'RNN NUM PARAMS {model.count_parameters():08,}')
    print(f'TRANSFORM FLAT SIZE {np.sum(model.kernels_flat)}')

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
