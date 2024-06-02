from torch import nn
from models.autoencoder.variational_encoder import VariationalEncoder
from models.autoencoder.variational_decoder import VariationalDecoder


class VariationalAutoencoder(nn.Module):
    def __init__(self, config):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(config)
        self.decoder = VariationalDecoder(config)

    def forward(self, x, clr, mask):
        z, kl = self.encoder(x, clr, mask)
        fake_x, fake_clr, fake_mask = self.decoder(z)
        return fake_x, fake_clr, fake_mask, kl
