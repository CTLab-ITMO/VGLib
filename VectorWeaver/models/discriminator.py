from torch import nn
from models.autoencoder.path_encoder import PathEncoder
from models.common import PositionalEncoder, DenseNet
import torch
from utils.model_utils import make_batch


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.path_encoder = PathEncoder(config, quantize=False)
        self.pos_embed = PositionalEncoder(self.config.paths_number, config.path_hidden_size, config)

        self.mean_token = nn.Embedding(1, config.path_hidden_size)

        self.make_input_embed = DenseNet(2, config.path_hidden_size * 2, config.path_hidden_size)

        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=config.path_hidden_size, nhead=8), num_layers=3)

        self.make_mean = DenseNet(2, config.path_hidden_size, 1)

    def forward(self, svg, color, mask):
        batch_size = svg.shape[0]

        svg = svg.reshape(batch_size * self.config.paths_number, self.config.coordinates_number)
        color = color.reshape(batch_size * self.config.paths_number, 3)
        mask = mask[:, :, 0].reshape(batch_size * self.config.paths_number, 1)

        path_number = torch.Tensor([i for i in range(self.config.paths_number)] * batch_size).long().to(self.config.device)
        pathes = self.path_encoder(svg, color, mask, path_number).reshape(batch_size, self.config.paths_number, self.config.path_hidden_size)

        pos_embed = self.pos_embed(batch_size)
        input_embed = self.make_input_embed(torch.cat([pathes, pos_embed], dim=-1))

        mean_token = make_batch(batch_size, self.mean_token(torch.LongTensor([0]).to(self.config.device)))
        output_embed = torch.cat([mean_token], dim=1)

        hidden_state = self.transformer(output_embed.permute(1, 0, 2), input_embed.permute(1, 0, 2)).permute(1, 0, 2)

        return 1 / (1 + torch.exp(-self.make_mean(hidden_state[:, 0, :])))

