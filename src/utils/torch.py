import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation():
    LEAKY_SLOPE = 0.5
    return nn.LeakyReLU(LEAKY_SLOPE, inplace=True)


def one_hot(t, one_hot_size=None):
    one_hot_size = t.max() + 1 if one_hot_size is None else one_hot_size

    hot = torch.zeros((t.size(0), one_hot_size))
    hot[torch.arange(t.size(0)), t] = 1
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


class BaseModule(nn.Module):
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_persisted(self, path):
        self.path = path

    def persist(self):
        torch.save(self.state_dict(), self.path)

    def save(self):
        torch.save(self, f'{self.path}_whole.h5')

    def preload_weights(self):
        self.load_state_dict(torch.load(self.path))

    def can_be_preloaded(self):
        return os.path.isfile(self.path)

    @property
    def device(self):
        return next(self.parameters()).device


def load_whole_model(path):
    return torch.load(path)


def batch_conv(x, w, p=0):
    # SRC - https://discuss.pytorch.org/t/apply-different-convolutions-to-a-batch-of-tensors/56901/2

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
            return x.reshape(*shape)

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


def compute_conv_output(net, frame_shape):
    with torch.no_grad():
        t = torch.rand(1, *frame_shape)
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
