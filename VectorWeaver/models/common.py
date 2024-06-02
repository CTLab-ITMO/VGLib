import torch
from torch import nn
import math
from utils.model_utils import make_batch


class MyLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.do = nn.Sequential(
            nn.Identity(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.do(x)


class DenseNet(nn.Module):
    def __init__(self, layers, dim_in, dim_out):
        super().__init__()
        self.list = nn.ModuleList([MyLinear(dim_in + i * (dim_in // (layers - 1)), dim_in // (layers - 1)) for i in range(layers - 1)])
        self.final = MyLinear(dim_in + (layers - 1) * (dim_in // (layers - 1)), dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer in self.list:
            x = torch.cat([x, layer(x)], dim=-1)
        x = self.dropout(x)
        return self.final(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(1000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, dim, config):
        super().__init__()

        self.seq_len = seq_len
        self.config = config

        self.pos_embed_learned_table = nn.Embedding(self.seq_len, dim)
        self.pos_embed_table = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            MyLinear(dim, dim)
        )
        self.unite_pos = DenseNet(3, dim * 2, dim)

    def forward(self, batch_size):
        pos_embed = torch.Tensor([i for i in range(self.seq_len)]).long().to(self.config.device)
        pos_embed = torch.cat([self.pos_embed_table(pos_embed), self.pos_embed_learned_table(pos_embed)], dim=-1)
        pos_embed = self.unite_pos(pos_embed)
        return make_batch(batch_size, pos_embed)
