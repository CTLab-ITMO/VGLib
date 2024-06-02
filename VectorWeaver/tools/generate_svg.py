import pickle
from argparse import ArgumentParser
import torch
from configs.config import Config
from utils.svg_utils import print_svg
from utils.diffusion_utils import DiffusionUtils
from utils.load_utils import load_vae, load_diffusion


def main(args):
    config = Config()
    diff_utils = DiffusionUtils(config)

    vae = load_vae(config, args.checkpoint)
    diffusion = load_diffusion(config, args.checkpoint)

    latent = diff_utils.generate_latent_fast(diffusion)

    vae.eval()
    data, clr, mask = vae.decoder(latent)
    data = data * mask
    clr = torch.clamp(clr, -1, 1)
    print_svg(data[0], clr[0], args.output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", help='path to checkpoint', required=True)
    parser.add_argument("--output", help='output path for SVG', required=True)

    args = parser.parse_args()

    main(args)
