import os

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


def get_activation():
    LEAKY_SLOPE = 0.2
    return nn.LeakyReLU(LEAKY_SLOPE, inplace=True)


def one_hot(t, one_hot_size=None):
    one_hot_size = t.max() + 1 if one_hot_size is None else one_hot_size

    hot = T.zeros((t.size(0), one_hot_size))
    hot[T.arange(t.size(0)), t] = 1
    return hot


def cat_channels(t):
    """
        Concatenate number of channels in a single tensor
        Converts tensor with shape:
            (bs, num_channels, channel_size, h, w)
        to tensor with shape:
            (bs, num_channels * channel_size, h, w)
    """
    shape = t.size()
    cat_dim_size = shape[1] * shape[2]
    return t.view(-1, cat_dim_size, *shape[3:])


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'Custom Module'

    def count_parameters(self):
        return count_parameters(self)

    def make_persisted(self, path):
        self.path = path

    def persist(self):
        T.save(self.state_dict(), self.path)

    def preload_weights(self):
        self.load_state_dict(T.load(self.path))

    def save(self, path=None):
        path = path if self.path is None else path
        T.save(self, f'{self.path}_whole.h5')

    def can_be_preloaded(self):
        return os.path.isfile(self.path)

    def summary(self):
        result = f' > {self.name[:38]:<38} | {count_parameters(self):09,}\n'
        for name, module in self.named_children():
            type = module._get_name()
            num_prams = count_parameters(module)
            result += f' >  {name[:20]:>20}: {type[:15]:<15} | {num_prams:9,}\n'

        return result

    @property
    def device(self):
        return next(self.parameters()).device


def batch_conv(x, w, p=0):
    # SRC - https://discuss.pyT.org/t/apply-different-convolutions-to-a-batch-of-tensors/56901/2

    batch_size = x.size(0)
    output_size = w.size(1)

    o = F.conv2d(
        x.reshape(1, batch_size * x.size(1), x.size(2), x.size(3)),
        w.reshape(batch_size * w.size(1), w.size(2), w.size(3), w.size(4)),
        groups=batch_size,
        padding=p,
    )
    o = o.reshape(batch_size, output_size, o.size(2), o.size(3))

    return o


def dense(i, o, a=get_activation()):
    l = nn.Linear(i, o)
    return l if a is None else nn.Sequential(l, a)


def reshape(*shape):
    class Reshaper(nn.Module):
        def forward(self, x):
            return x.reshape(shape)

    return Reshaper()


def lam(forward):
    class Lambda(nn.Module):
        def forward(self, *args):
            return forward(*args)

    return Lambda()


def resize(t, size):
    return F.interpolate(t, size, mode='bicubic', align_corners=True)


def conv_block(i, o, ks, s, p, a=get_activation(), d=1, bn=True):
    block = [nn.Conv2d(i, o, kernel_size=ks, stride=s, padding=p, dilation=d)]
    if bn: block.append(nn.BatchNorm2d(o))
    if a is not None: block.append(a)

    return nn.Sequential(*block)


def deconv_block(i, o, ks, s, p, a=get_activation(), d=1, bn=True):
    block = [
        nn.ConvTranspose2d(
            i,
            o,
            kernel_size=ks,
            stride=s,
            padding=p,
            dilation=d,
        )
    ]

    if bn: block.append(nn.BatchNorm2d(o))
    if a is not None: block.append(a)

    return nn.Sequential(*block)


def conv_blocks(conv_cell, sizes, ks, a):
    layers = [
        conv_cell(
            i=sizes[l],
            o=sizes[l + 1],
            ks=ks,
            s=2,
            p=ks // 2 - 1,
            a=(None if l == len(sizes) - 2 else a),
            d=1,
            # batch norm everywhere except the last layer
            bn=(l != len(sizes) - 2),
        ) for l in range(len(sizes) - 1)
    ]

    return nn.Sequential(*layers)


def conv_encoder(sizes, ks=4, a=get_activation()):
    return conv_blocks(conv_block, sizes, ks, a)


def conv_decoder(sizes, ks=4, a=get_activation()):
    return conv_blocks(deconv_block, sizes, ks, a)


def compute_conv_output(net, frame_shape):
    with T.no_grad():
        t = T.rand(1, *frame_shape)
        out = net(t)

        return out.shape


def extract_tensors(vec, tensor_shapes):
    slice_indices = [0] + [np.prod(t) for t in tensor_shapes]
    slice_indices = np.cumsum(slice_indices)

    tensors = []
    for i in range(len(slice_indices) - 1):
        t = vec[:, slice_indices[i]:slice_indices[i + 1]]
        t = t.reshape(-1, *tensor_shapes[i])
        tensors.append(t)

    return tensors


@T.jit.script
def mask_sequence(tensor, mask):
    initial_shape = tensor.shape
    bs, seq = mask.shape
    masked = T.where(
        mask.reshape(bs * seq, -1),
        tensor.reshape(bs * seq, -1),
        T.tensor(0, dtype=T.float32).to(tensor.device),
    )

    return masked.reshape(initial_shape)


@T.jit.script
def prepare_rnn_state(state, num_rnn_layers):
    """
    RNN cells expect the initial state
    in the shape -> [rnn_num_layers, bs, rnn_state_size]

    In this case rnn_state_size = state // rnn_num_layers.
    The state is distributed among the layers

    state          -> [bs, state_size]
    rnn_num_layers -> int
    """
    return T.stack(state.chunk(num_rnn_layers, dim=1), dim=0)


def time_distribute(module, input=None):
    """
    Distribute execution of module over batched sequential input tensor.
    This is done in the batch dimension to facilitate parallel execution.

    input  -> [bs, seq, *x*]
    module -> something that takes *x*
    return -> [bs, seq, module(x)]
    """
    if input is None: return time_distribute_decorator(module)

    bs = input.size(0)
    seq_len = input.size(1)
    input = input.reshape(-1, *input.shape[2:])

    out = module(input)
    out = out.reshape(bs, seq_len, *out.shape[1:])

    return out


def time_distribute_decorator(module):
    class TimeDistributed(nn.Module):
        def __init__(self):
            # IMPORTANT:
            #   This init has to be here so that the
            #   passed module parameters can be part of the
            #   state (parameters) of the overall model.

            super().__init__()
            self.module = module

        def forward(self, input):
            bs = input.size(0)
            seq_len = input.size(1)
            input = input.reshape(-1, *input.shape[2:])

            out = module(input)
            out = out.view(bs, seq_len, *out.shape[1:])

            return out

    return TimeDistributed()


def time_distribute_13D(module):
    class TimeDistributed(nn.Module):
        def forward(self, input):
            bs, seq_len, s = [input.size(i) for i in range(3)]
            input = input.reshape(-1, s)
            out = module(input)
            return out.reshape(
                bs,
                seq_len,
                out.size(1),
                out.size(2),
                out.size(3),
            )

    return T.jit.script(TimeDistributed())


def time_distribute_31D(module):
    class TimeDistributed(nn.Module):
        def forward(self, input):
            bs, seq_len, c, h, w = [input.size(i) for i in range(5)]
            input = input.reshape(-1, c, h, w)
            out = module(input)
            return out.reshape(bs, seq_len, out.size(1))

    return T.jit.script(TimeDistributed())


if __name__ == '__main__':
    # Sanity check mask_sequence
    tensor = T.rand(2, 3, 4)
    mask = T.rand(2, 3) > 0.5
    masked = mask_sequence(tensor, mask)
    print(mask)
    print(masked)
