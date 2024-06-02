import os
import torch
from torch import nn

from models.diffusion import StableDiffusion
from models.vae import VariationalAutoencoder
from models.discriminator import Discriminator
from tqdm import tqdm
import pickle


def load_dataset(dataset_path):
    images = []
    for source_file in tqdm(os.listdir(dataset_path)):
        img_path = os.path.join(dataset_path, source_file)
        data, fill, mask = pickle.load(open(img_path, 'rb'))
        data = torch.Tensor(data)
        fill = torch.Tensor(fill)
        mask = torch.Tensor(mask)
        images.append((data * mask, mask, fill))

    return images


def load_vae(config, checkpoint_path):
    vae = VariationalAutoencoder(config)
    if os.path.isfile(os.path.join(checkpoint_path, 'vae')):
        print("VAE found")
        vae.load_state_dict(torch.load(os.path.join(checkpoint_path, 'vae'), map_location=torch.device(config.device)), strict=True)
    else:
        print("VAE not found")
    return vae


def load_discriminator(config, checkpoint_path):
    disc = Discriminator(config)
    if os.path.isfile(os.path.join(checkpoint_path, 'disc')):
        print("Discriminator found")
        disc.load_state_dict(torch.load(os.path.join(checkpoint_path, 'disc'), map_location=torch.device(config.device)), strict=True)
    else:
        print("Discriminator not found")
    return disc


def load_diffusion(config, checkpoint_path):
    disc = StableDiffusion(config)
    if os.path.isfile(os.path.join(checkpoint_path, 'diffusion')):
        print("Diffusion found")
        disc.load_state_dict(torch.load(os.path.join(checkpoint_path, 'diffusion'), map_location=torch.device(config.device)), strict=True)
    else:
        print("Diffusion not found")
    return disc


def save_vae(vae, checkpoint_path):
    model = vae
    if isinstance(vae, nn.DataParallel):
        model = vae.module

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'vae'))


def save_discriminator(disc, checkpoint_path):
    model = disc
    if isinstance(disc, nn.DataParallel):
        model = disc.module

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'disc'))


def save_diffusion(disc, checkpoint_path):
    model = disc
    if isinstance(disc, nn.DataParallel):
        model = disc.module

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'diffusion'))

