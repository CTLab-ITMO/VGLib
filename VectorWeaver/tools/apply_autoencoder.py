import pickle
from argparse import ArgumentParser
import torch
from utils.svg_utils import print_svg
from utils.load_utils import load_vae
from configs.config import Config


def main(args):
    config = Config()
    data, fill, mask, = pickle.load(open(args.input, 'rb'))
    data = torch.Tensor(data).unsqueeze(0)
    fill = torch.Tensor(fill).unsqueeze(0)
    mask = torch.Tensor(mask).unsqueeze(0)

    vae = load_vae(config, args.checkpoint)
    vae.eval()

    with torch.no_grad():
        rec_img, rec_clr, rec_mask, _ = vae(data, fill, mask)
        rec_img = rec_img * rec_mask
        rec_clr = torch.clamp(rec_clr, -1, 1)
        print_svg(rec_img[0], rec_clr[0], args.output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input", help='pickle file from dataset', required=True)
    parser.add_argument("--checkpoint", help='path to checkpoint', required=True)
    parser.add_argument("--output", help='output path for SVG', required=True)

    args = parser.parse_args()

    main(args)
