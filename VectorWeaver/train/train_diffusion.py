from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from utils.diffusion_utils import DiffusionUtils
from utils.load_utils import load_dataset, load_vae, load_discriminator, save_vae, save_discriminator, load_diffusion, save_diffusion
from configs.config import Config


def train_epoch(vae, diffusion, diff_utils, optim_dif, dataloader, config):
    lr_dif_scheduler = CosineAnnealingWarmRestarts(optim_dif, T_0=20, T_mult=1, eta_min=config.diffusion_lr / 10)

    vae.eval()

    steps = 0
    sum_mse = 0
    pbar = tqdm(dataloader)
    for x in pbar:
        x = x.to(config.device)
        steps += 1

        t = torch.randint(0, config.T, (x.shape[0],), device=config.device).long()
        noised, noise = diff_utils.forward_diffusion_sample(x, t)
        pred_noise = diffusion(noised, t)

        loss_mse = ((noise - pred_noise) ** 2).mean()
        optim_dif.zero_grad()
        loss_mse.backward()
        optim_dif.step()

        sum_mse += loss_mse

        lr_dif_scheduler.step()

        x.detach().cpu()

        pbar.set_description("var_mse: %0.3f" % (sum_mse / steps))


def main(args):
    config = Config()
    diff_utils = DiffusionUtils(config)

    vae = load_vae(config, args.checkpoint_input)
    diffusion = load_diffusion(config, args.checkpoint_input)

    train_dataset = load_dataset(args.input)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.autoencoder_batch_size)

    latent_data = []
    for x, mask, fill in tqdm(train_loader):
        x = x.to(config.device)
        fill = fill.to(config.device)
        mask = mask.to(config.device)
        z, _ = vae.encoder(x, fill, mask)
        x.detach().cpu()
        fill.detach().cpu()
        mask.detach().cpu()
        for i in z.detach():
            latent_data.append(i)

    train_loader = torch.utils.data.DataLoader(latent_data, batch_size=config.diffusion_batch_size)

    if config.num_gpus > 1:
        vae = nn.DataParallel(vae)
        diffusion = nn.DataParallel(diffusion)

    optim_dif = torch.optim.AdamW(diffusion.parameters(), lr=config.diffusion_lr, weight_decay=1e-5)

    epoch = 0
    while True:
        print("Epoch #" + str(epoch))
        train_epoch(vae, diffusion, diff_utils, optim_dif, train_loader, config)
        epoch += 1
        save_vae(vae, args.checkpoint_output)
        save_diffusion(diffusion, args.checkpoint_output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input", help='path to prepared dataset', required=True)
    parser.add_argument("--checkpoint_input", help='path to input checkpoint')
    parser.add_argument("--checkpoint_output", help='path to output checkpoint', required=True)

    args = parser.parse_args()

    main(args)
