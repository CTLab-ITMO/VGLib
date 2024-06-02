from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from utils.load_utils import load_dataset, load_vae, load_discriminator, save_vae, save_discriminator
from configs.config import Config
from torch.nn import BCELoss


def train_discriminator_step(data, fill, mask, vae, disc, optim_disc, config):
    vae.eval()
    disc.train()
    bce = BCELoss()

    data = data.to(config.device)
    fill = fill.to(config.device)
    mask = mask.to(config.device)

    data_fake, fill_fake, mask_fake, kl = vae(data, fill, mask)

    real_prob = disc(data, fill, mask)
    fake_prob = disc(data_fake, fill_fake, mask_fake)

    loss_disc = bce(real_prob, torch.ones(real_prob.shape).float().to(config.device)) + \
                bce(fake_prob, torch.zeros(fake_prob.shape).float().to(config.device))

    optim_disc.zero_grad()
    loss_disc.backward()
    optim_disc.step()

    data.cpu().detach()
    mask.cpu().detach()
    fill.cpu().detach()

    return loss_disc.item()


def train_vae_step(data, fill, mask, vae, disc, optim_vae, config):
    data = data.to(config.device)
    fill = fill.to(config.device)
    mask = mask.to(config.device)

    vae.train()
    disc.eval()
    bce = BCELoss()

    data_fake, clr_fake, mask_fake, kl = vae(data, fill, mask)
    fake_prob = disc(data_fake, clr_fake, mask_fake)

    loss_mse = (((data - data_fake) ** 2) * mask).sum() / mask.sum() * 100
    loss_kl = kl.mean() / 10
    loss_clr = (((fill - clr_fake) ** 2) * mask[:, :, 0:3]).sum() / mask[:, :, 0:3].sum() * 10
    loss_mask = bce(mask_fake, mask.float()) * 100
    loss_adv = bce(fake_prob, torch.ones(fake_prob.shape).float().to(config.device)) / 10

    loss_vae = loss_mse + loss_kl + loss_clr + loss_mask + loss_adv

    optim_vae.zero_grad()
    loss_vae.backward()
    optim_vae.step()

    data.cpu().detach()
    mask.cpu().detach()
    fill.cpu().detach()

    return loss_mse.item(), loss_kl.item(), loss_clr.item(), loss_mask.item(), loss_adv.item()


def train_epoch(vae, disc, optim_vae, optim_disc, dataloader, config):
    lr_vae_scheduler = CosineAnnealingWarmRestarts(optim_vae, T_0=20, T_mult=1, eta_min=config.autoencoder_lr / 10)
    lr_disc_scheduler = CosineAnnealingWarmRestarts(optim_disc, T_0=20, T_mult=1, eta_min=config.discriminator_lr / 10)

    vae.train()
    disc.train()

    steps = 0
    sum_disc, sum_mse, sum_kl, sum_clr, sum_mask, sum_adv = 0, 0, 0, 0, 0, 0
    pbar = tqdm(dataloader)
    for data, mask, fill in pbar:
        mask = torch.cat([mask] * config.coordinates_number, dim=-1)

        steps += 1

        # TRAIN DISCRIMINATOR
        loss_disc = train_discriminator_step(data, fill, mask, vae, disc, optim_disc, config)
        sum_disc += loss_disc

        # TRAIN AUTOENCODER
        loss_mse, loss_kl, loss_clr, loss_mask, loss_adv = train_vae_step(data, fill, mask, vae, disc, optim_vae, config)
        sum_mse += loss_mse
        sum_kl += loss_kl
        sum_clr += loss_clr
        sum_mask += loss_mask
        sum_adv += loss_adv

        lr_vae_scheduler.step()
        lr_disc_scheduler.step()

        pbar.set_description("var_mse: %0.3f, var_disc: %0.3f, var_kl: %0.3f, var_clr: %0.3f, sum_mask: %0.3f, var_adv: %0.3f" % (
            sum_mse / steps, sum_disc / steps, sum_kl / steps, sum_clr / steps, sum_mask / steps, sum_adv / steps))


def main(args):
    config = Config()

    train_dataset = load_dataset(args.input)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.autoencoder_batch_size)

    vae = load_vae(config, args.checkpoint_input)
    disc = load_discriminator(config, args.checkpoint_input)

    if config.num_gpus > 1:
        vae = nn.DataParallel(vae)
        disc = nn.DataParallel(disc)

    optim_vae = torch.optim.AdamW(vae.parameters(), lr=config.autoencoder_lr, weight_decay=1e-5)
    optim_disc = torch.optim.AdamW(disc.parameters(), lr=config.discriminator_lr, weight_decay=1e-5)

    epoch = 0
    while True:
        print("Epoch #" + str(epoch))
        train_epoch(vae, disc, optim_vae, optim_disc, train_loader, config)
        epoch += 1
        save_vae(vae, args.checkpoint_output)
        save_discriminator(disc, args.checkpoint_output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input", help='path to prepared dataset', required=True)
    parser.add_argument("--checkpoint_input", help='path to input checkpoint')
    parser.add_argument("--checkpoint_output", help='path to output checkpoint', required=True)

    args = parser.parse_args()

    main(args)
