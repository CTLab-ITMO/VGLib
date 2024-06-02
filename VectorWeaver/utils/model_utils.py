import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def make_batch(batch_size, data):
    return torch.stack([data for x in range(batch_size)])


def make_seq(seq_size, data):
    data = data.reshape(data.shape[0], 1, data.shape[-1])
    return torch.cat([data for x in range(seq_size)], dim=1)
