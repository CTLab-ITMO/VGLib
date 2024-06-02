from torch import nn
from models.common import PositionalEncoder, MyLinear, DenseNet
import torch
from utils.model_utils import make_batch, sigmoid
from models.autoencoder.path_encoder import PathEncoder


class VariationalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.path_encoder = PathEncoder(config)
        self.pos_embed = PositionalEncoder(config.paths_number, config.path_hidden_size, config)

        self.mean_token = nn.Embedding(1, config.path_hidden_size)
        self.var_token = nn.Embedding(1, config.path_hidden_size)

        self.make_input_embed = DenseNet(2, config.path_hidden_size * 2, config.path_hidden_size)

        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=config.path_hidden_size, nhead=8), num_layers=3)

        self.make_mean = DenseNet(2, config.path_hidden_size, config.path_hidden_size)
        self.make_var = DenseNet(2, config.path_hidden_size, config.path_hidden_size)

        self.N = torch.distributions.Normal(0, 1)
        if self.config.is_cuda:
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, svg, color, mask):
        batch_size = svg.shape[0]

        svg = svg.reshape(batch_size * self.config.paths_number, self.config.coordinates_number)
        color = color.reshape(batch_size * self.config.paths_number, 3)
        mask = mask[:, :, 0].reshape(batch_size * self.config.paths_number, 1)

        paths_numbers = torch.arange(0, self.config.paths_number).long().to(self.config.device)
        paths_numbers = torch.cat([paths_numbers] * batch_size)
        paths = self.path_encoder(svg, color, mask, paths_numbers).reshape(batch_size, self.config.paths_number, -1)

        pos_embed = self.pos_embed(batch_size)

        input_embed = self.make_input_embed(torch.cat([paths, pos_embed], dim=-1))

        mean_token = make_batch(batch_size, self.mean_token(torch.LongTensor([0]).to(self.config.device)))
        var_token = make_batch(batch_size, self.var_token(torch.LongTensor([0]).to(self.config.device)))
        output_embed = torch.cat([mean_token, var_token], dim=1)

        hidden_state = self.transformer(output_embed.permute(1, 0, 2), input_embed.permute(1, 0, 2)).permute(1, 0, 2)

        z_mean = self.make_mean(hidden_state[:, 0, :])
        z_var = torch.exp(self.make_var(hidden_state[:, 1, :]))

        z = z_mean + z_var * self.N.sample(z_mean.shape).to(self.config.device)
        kl = (z_var ** 2 + z_mean ** 2 - torch.log(z_var) - 1 / 2).mean()
        return z, kl
