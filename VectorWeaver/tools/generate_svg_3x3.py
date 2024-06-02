from argparse import ArgumentParser

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
from cairosvg import svg2png

from configs.config import Config
from utils.diffusion_utils import DiffusionUtils
from utils.load_utils import load_vae, load_diffusion
from utils.svg_utils import print_svg


def main(args):
    config = Config()
    diff_utils = DiffusionUtils(config)

    vae = load_vae(config, args.checkpoint)
    diffusion = load_diffusion(config, args.checkpoint)

    latent = diff_utils.generate_latent_fast(diffusion, n=9)

    vae.eval()
    data, clr, mask = vae.decoder(latent)
    data = data * mask
    clr = torch.clamp(clr, -1, 1)

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            svg_str = print_svg(data[i * 3 + j], clr[i * 3 + j])
            png = svg2png(bytestring=svg_str)
            ax[i, j].imshow(imageio.imread(png))
            ax[i, j].axis('off')
    plt.savefig(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--checkpoint", help='path to checkpoint', required=True)
    parser.add_argument("--output", help='output path for SVG', required=True)

    args = parser.parse_args()

    main(args)
