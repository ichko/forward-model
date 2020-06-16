import torch as T
import torch.nn as nn
import torch.functional as F

SEQ_DIM = 1


def pad_seq(t, skip):
    pass


def dilate_rnn(cell, skip, autopad=True):
    class DRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = cell
            self.skip = skip

        def forward(self, x, state=None):
            x_input = x
            should_pad = x.size(SEQ_DIM) % self.skip != 0

            if autopad and should_pad:
                pad_shape = list(x.size())
                seq_pad_size = T.ceil(T.tensor(
                    x.size(SEQ_DIM) / self.skip)) * self.skip
                pad_shape[SEQ_DIM] = int(seq_pad_size.item())
                x_pad = T.zeros(*pad_shape).to(x.device)
                x_pad[:, :x.size(SEQ_DIM)] = x
                x_input = x_pad

            skipped = seq_skip(x_input, self.skip)

            skipped_rollout, last_state = self.cell(skipped)
            rollout = reverse_seq_skip(skipped_rollout, self.skip)
            # last_state = reverse_seq_skip(last_state, self.skip)
            last_state = None

            if autopad and should_pad:
                initial_seq_len = x.size(SEQ_DIM)
                rollout = rollout[:, :initial_seq_len]

            return rollout, last_state

    return DRNN()


def seq_skip(t, skip):
    """
    Make skips in the seq dimension by moving them in the batch dimension.
    Example:
        t = torch.arange(16).reshape(4, -1)
        t
        >>  tensor([[0, 1, 2, 3],
                    [4, 5, 6, 7]])

        seq_skip(t, skip=2)
        >>  tensor([[0, 2],
                    [1, 3],
                    [4, 6],
                    [5, 7]])
    """
    batch_dim = 0
    seq_dim = SEQ_DIM

    assert t.shape[seq_dim] % skip == 0, \
        f'seq_dim ({seq_dim}) should be divisible by the skip size ({skip})'

    new_shape = list(t.shape)
    new_shape[batch_dim] = new_shape[batch_dim] * skip
    new_shape[seq_dim] = -1

    return T.cat(
        [t[:, i::skip] for i in range(skip)],
        dim=seq_dim,
    ).reshape(*new_shape)


def reverse_seq_skip(skipped, skip):
    """Reverses the transformation of seq_skip"""
    batch_dim = 0
    seq_dim = SEQ_DIM

    assert skipped.shape[batch_dim] % skip == 0, \
        f'batch_dim ({batch_dim}) should be divisible by the skip size ({skip})'

    old_seq_size = skipped.shape[seq_dim]
    new_shape = list(skipped.shape)
    new_batch_size = new_shape[batch_dim] // skip
    new_shape[batch_dim] = new_batch_size
    new_shape[seq_dim] = -1
    s = skipped.reshape(*new_shape)

    return T.cat(
        [s[:, i::old_seq_size] for i in range(old_seq_size)],
        dim=seq_dim,
    )


def verify_skip(shape, skip):
    expected = T.rand(shape)
    s = seq_skip(expected, skip)
    actual = reverse_seq_skip(s, skip)

    assert T.all(actual == expected)


if __name__ == '__main__':
    verify_skip((10, 10), 2)
    verify_skip((16, 16), 4)
    verify_skip((32, 16), 8)
    verify_skip((16, 16), 16)
    verify_skip((16, 16, 10), 4)
    verify_skip((16, 16, 10), 2)
    verify_skip((3, 16, 10), 16)
    verify_skip((18, 16, 10, 10, 3), 16)
    verify_skip((32, 16, 5, 10, 17), 16)
    verify_skip((31, 64, 10, 26, 2), 16)
    print('Verified!')
