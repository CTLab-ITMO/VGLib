from torch import nn
from models.common import PositionalEncoder, DenseNet
from models.autoencoder.path_decoder import PathDecoder


class VariationalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pos_embed = PositionalEncoder(config.paths_number, config.path_hidden_size, config)

        self.make_input = DenseNet(2, config.path_hidden_size, config.path_hidden_size)

        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=config.path_hidden_size, nhead=8), num_layers=3)

        self.path_decoder = PathDecoder(config)

    def forward(self, hidden):
        batch_size = hidden.shape[0]

        input_embed = self.make_input(hidden).reshape(batch_size, 1, self.config.path_hidden_size)
        output_embed = self.pos_embed(batch_size)

        hidden_state = self.transformer(output_embed.permute(1, 0, 2), input_embed.permute(1, 0, 2)).permute(1, 0, 2)
        hidden_state = hidden_state.reshape(batch_size * self.config.paths_number, self.config.path_hidden_size)

        result, color, mask = self.path_decoder(hidden_state)

        result = result.reshape(batch_size, self.config.paths_number, -1)
        color = color.reshape(batch_size, self.config.paths_number, -1)
        mask = mask.reshape(batch_size, self.config.paths_number, -1)

        return result, color, mask
