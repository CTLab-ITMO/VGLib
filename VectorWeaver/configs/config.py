import torch

class Config:
    def __init__(self):
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.is_cuda else torch.device("cpu")
        self.num_gpus = torch.cuda.device_count() if self.is_cuda else 0

        self.paths_number = 10
        self.segments_number = 20
        # points_number = 3 points for each bezier curve, 1 start point
        # coordinates_number = 2 * points_number (x, y)
        self.coordinates_number = self.segments_number * 6 + 2

        self.discretization_grid_size = 2 ** 12
        self.discretization_range = 1

        self.color_range = 256

        self.small_hidden_size = 128
        self.segment_hidden_size = 512
        self.path_hidden_size = 1024

        self.autoencoder_lr = 1e-6
        self.discriminator_lr = 1e-5
        self.autoencoder_batch_size = 256 * self.num_gpus if self.is_cuda else 16

        self.diffusion_lr = 1e-5
        self.diffusion_batch_size = 1024 * self.num_gpus if self.is_cuda else 16

        self.T = 1000
