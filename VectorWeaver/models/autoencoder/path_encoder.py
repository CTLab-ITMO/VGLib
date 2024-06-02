from torch import nn
from models.common import PositionalEncoder, MyLinear, DenseNet
from transformers import BertConfig, BertModel
import torch


class PathEncoder(nn.Module):
    def __init__(self, config, quantize=True):
        super().__init__()
        self.config = config
        self.quantize = quantize

        self.pos_embed = PositionalEncoder((1 + config.segments_number), config.segment_hidden_size, config)

        if self.quantize:
            self.float_embed = nn.Embedding(config.discretization_grid_size, config.segment_hidden_size)

        self.r_embed = nn.Embedding(256, config.segment_hidden_size)
        self.g_embed = nn.Embedding(256, config.segment_hidden_size)
        self.b_embed = nn.Embedding(256, config.segment_hidden_size)
        self.mask_embed = nn.Embedding(2, config.segment_hidden_size)
        self.path_number_embed = nn.Embedding(config.paths_number, config.segment_hidden_size)

        self.segment_embed = MyLinear(config.segment_hidden_size * 6, config.segment_hidden_size)

        self.non_quantize_embed = nn.Sequential(
            nn.Linear(1, config.segment_hidden_size),
            DenseNet(3, config.segment_hidden_size, config.segment_hidden_size)
        )

        self.make_input_embed = DenseNet(2, config.segment_hidden_size * 2, config.segment_hidden_size)

        transformer_length = (1 + config.segments_number) + 3 + 2
        self.encoder = BertModel(BertConfig(
            1,
            config.segment_hidden_size,
            num_attention_heads=8,
            num_hidden_layers=2,
            max_position_embeddings=transformer_length
        ))

        self.expand_embeds = MyLinear(config.segment_hidden_size, config.small_hidden_size)

        self.make_output = nn.Sequential(
            nn.Linear(config.small_hidden_size * transformer_length, config.path_hidden_size),
            DenseNet(2, config.path_hidden_size, config.path_hidden_size)
        )

    def forward(self, svg, color, mask, path_number):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, self.config.coordinates_number)
        svg = torch.cat([svg[:, 0:2], svg[:, 0:2], svg], dim=-1)

        if self.quantize:
            svg_long = torch.clamp(
                (svg + self.config.discretization_range) / (2 * self.config.discretization_range) * self.config.discretization_grid_size,
                0, self.config.discretization_grid_size - 1
            ).long()
            encoded_coords = self.float_embed(svg_long)
        else:
            svg_long = svg.reshape(batch_size, self.config.coordinates_number + 4, 1)
            encoded_coords = self.non_quantize_embed(svg_long)

        def get_color_value(x):
            return torch.clamp(((x + 1) * self.config.color_range / 2).long(), 0, self.config.color_range - 1)

        r_embed = self.r_embed(get_color_value(color[:, 0])).reshape(batch_size, 1, -1)
        g_embed = self.g_embed(get_color_value(color[:, 1])).reshape(batch_size, 1, -1)
        b_embed = self.b_embed(get_color_value(color[:, 2])).reshape(batch_size, 1, -1)

        path_number_embed = self.path_number_embed(path_number).reshape(batch_size, 1, -1)
        mask_embed = self.mask_embed(mask.long()).reshape(batch_size, 1, -1)

        segment_coords = encoded_coords.reshape(batch_size, (1 + self.config.segments_number), -1)
        segment_embeds = self.segment_embed(segment_coords)

        pos_embed = self.pos_embed(batch_size)

        input_embeds = self.make_input_embed(torch.cat([segment_embeds, pos_embed], dim=-1))

        hidden_state = self.encoder(
            inputs_embeds=torch.cat([input_embeds, r_embed, g_embed, b_embed, mask_embed, path_number_embed], dim=1)
        ).last_hidden_state

        hidden_state = self.expand_embeds(hidden_state)

        return self.make_output(hidden_state.reshape(batch_size, -1))
