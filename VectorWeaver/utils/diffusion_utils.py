import torch
import torch.nn.functional as F
from tqdm import tqdm


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionUtils:

    def __init__(self, config):
        self.config = config

        self.T = 1000
        self.betas = linear_beta_schedule(timesteps=self.T).to(config.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(config.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(config.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.fast_config = [i for i in range(self.T - 1, -1, -5)] + [i for i in range(100, -1, -1)]

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.config.device) * x_0.to(self.config.device) \
               + sqrt_one_minus_alphas_cumprod_t.to(self.config.device) * noise.to(self.config.device), noise.to(self.config.device)

    def sample_timestep(self, x, t, predict):
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predict / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
        if torch.equal(t, t * 0):
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def predict_original(self, x, t, predict):
        sqrt_recip_alphas_cumprod_t = get_index_from_list(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        return sqrt_recip_alphas_cumprod_t * (x - sqrt_one_minus_alphas_cumprod_t * predict)

    def generate_latent_fast(self, diffusion, n=1):
        diffusion.eval()
        with torch.no_grad():
            img = torch.randn((n, self.config.path_hidden_size), device=self.config.device)
            for i in tqdm(range(len(self.fast_config) - 1)):
                t = torch.full((n,), self.fast_config[i], device=self.config.device, dtype=torch.long)
                t_next = torch.full((n,), self.fast_config[i + 1], device=self.config.device, dtype=torch.long)

                x_0 = self.predict_original(img, t, diffusion(img, t))
                img, _ = self.forward_diffusion_sample(x_0, t_next)

            return img

    def generate_latent(self, diffusion, n=1):
        diffusion.eval()
        with torch.no_grad():
            img = torch.randn((n, self.config.path_hidden_size), device=self.config.device)
            for i in tqdm(range(self.T - 1, -1, -1)):
                t = torch.full((n,), i, device=self.config.device, dtype=torch.long)
                img = self.sample_timestep(img, t, diffusion(img, t))

            return img
