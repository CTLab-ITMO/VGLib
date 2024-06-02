from torch import nn
from models.common import PositionalEncoder, MyLinear, DenseNet
from transformers import BertConfig, BertModel
import torch
from utils.model_utils import make_batch, sigmoid


class PathDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pos_embed = PositionalEncoder((1 + config.segments_number), config.segment_hidden_size, config)
        self.r_token = nn.Embedding(1, config.segment_hidden_size)
        self.g_token = nn.Embedding(1, config.segment_hidden_size)
        self.b_token = nn.Embedding(1, config.segment_hidden_size)
        self.mask_token = nn.Embedding(1, config.segment_hidden_size)

        self.make_input_embed = DenseNet(3, config.path_hidden_size, config.small_hidden_size * (1 + config.segments_number))

        self.expand_embeds = MyLinear(config.small_hidden_size, config.segment_hidden_size)

        self.unite_with_time = DenseNet(2, config.segment_hidden_size * 2, config.segment_hidden_size)

        transformer_length = (1 + config.segments_number) + 3 + 1
        self.encoder = BertModel(BertConfig(
            1,
            config.segment_hidden_size,
            num_attention_heads=8,
            num_hidden_layers=2,
            max_position_embeddings=transformer_length
        ))

        self.make_output = DenseNet(3, config.segment_hidden_size, 6)

        self.make_color = DenseNet(3, config.segment_hidden_size, 1)

        self.make_mask = DenseNet(3, config.segment_hidden_size, 1)

    def forward(self, svg):
        batch_size = svg.shape[0]
        svg = svg.reshape(batch_size, self.config.path_hidden_size)
        segment_embeds = self.make_input_embed(svg)

        pos_embed = self.pos_embed(batch_size)
        segment_embeds = segment_embeds.reshape(batch_size, (1 + self.config.segments_number), self.config.small_hidden_size)
        segment_embeds = self.expand_embeds(segment_embeds)

        r_embed = make_batch(batch_size, self.r_token(torch.LongTensor([0]).to(self.config.device)))
        g_embed = make_batch(batch_size, self.g_token(torch.LongTensor([0]).to(self.config.device)))
        b_embed = make_batch(batch_size, self.b_token(torch.LongTensor([0]).to(self.config.device)))
        mask_embed = make_batch(batch_size, self.mask_token(torch.LongTensor([0]).to(self.config.device)))

        embeds = self.unite_with_time(torch.cat([segment_embeds, pos_embed], dim=-1))
        hidden_state = self.encoder(
            inputs_embeds=torch.cat([embeds, r_embed, g_embed, b_embed, mask_embed], dim=1)
        ).last_hidden_state

        svg_hidden_state = hidden_state[:, :-4, :]
        color_state = hidden_state[:, -4:-1, :]
        mask_state = hidden_state[:, -1:, :]

        svg = self.make_output(svg_hidden_state).reshape(batch_size, self.config.coordinates_number + 4)[:, 4:]
        color = self.make_color(color_state).reshape(batch_size, 3)
        mask = self.make_mask(mask_state).reshape(batch_size, 1) * torch.ones(1, self.config.coordinates_number).to(self.config.device)

        return svg, color, sigmoid(mask)
