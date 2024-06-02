import torch
from torch import nn

from models.common import SinusoidalPositionEmbeddings


class ResidualBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.proj = nn.Linear(inp, out)
        self.norm = nn.GroupNorm(8, out)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TimeEncoder(nn.Module):
    def __init__(self, seq_len, config):
        super().__init__()

        self.seq_len = seq_len

        self.pos_embed_learned_table = nn.Embedding(self.seq_len, config.path_hidden_size)
        self.pos_embed_table = nn.Sequential(
            SinusoidalPositionEmbeddings(config.path_hidden_size),
            nn.Linear(config.path_hidden_size, config.path_hidden_size),
            nn.ReLU()
        )
        self.unite_pos = nn.Sequential(
            nn.Linear(config.path_hidden_size * 2, config.path_hidden_size),
            nn.ReLU(),
            nn.Linear(config.path_hidden_size, config.path_hidden_size),
            nn.ReLU(),
            nn.Linear(config.path_hidden_size, config.path_hidden_size),
        )

    def forward(self, t):
        pos_embed = torch.cat([self.pos_embed_table(t), self.pos_embed_learned_table(t)], dim=-1)
        pos_embed = self.unite_pos(pos_embed)
        return pos_embed


class StableDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        DIFFUSION_HIDDEN = 2048
        self.DIFFUSION_STEPS = 15

        self.time_encoder = TimeEncoder(config.T, config)

        self.make_input = nn.Sequential(
            nn.Linear(config.path_hidden_size * 2, DIFFUSION_HIDDEN),
            nn.Tanh()
        )

        self.up = nn.ModuleList([
            ResidualBlock(DIFFUSION_HIDDEN, DIFFUSION_HIDDEN) for x in range(self.DIFFUSION_STEPS)
        ])

        self.make_embed = nn.Sequential(
            nn.Linear(DIFFUSION_HIDDEN, DIFFUSION_HIDDEN),
            nn.LeakyReLU(0.2),
            nn.Linear(DIFFUSION_HIDDEN, DIFFUSION_HIDDEN)
        )

        self.down = nn.ModuleList([
            ResidualBlock(DIFFUSION_HIDDEN * 3, DIFFUSION_HIDDEN) for x in range(self.DIFFUSION_STEPS)
        ])

        self.result = nn.Sequential(
            nn.Linear(DIFFUSION_HIDDEN, config.path_hidden_size),
            nn.Tanh(),
            nn.Linear(config.path_hidden_size, config.path_hidden_size)
        )

    def forward(self, x, t):
        embed_up = []

        t = self.time_encoder(t)
        x = self.make_input(torch.cat([t, x], dim=-1))

        for i in range(self.DIFFUSION_STEPS):
            x = self.up[i](x)
            embed_up.append(x)

        x = self.make_embed(x)

        for i in range(self.DIFFUSION_STEPS):
            x = self.down[i](torch.cat([x, embed_up[-i - 1], x * embed_up[-i - 1]], dim=-1))
        x = self.result(x)

        return x
