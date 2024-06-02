import pickle
from argparse import ArgumentParser
import torch
from utils.svg_utils import print_svg


def main(args):
    data, fill, mask, = pickle.load(open(args.input, 'rb'))
    print_svg(torch.Tensor(data), torch.Tensor(fill), args.output)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input", help='pickle file from dataset', required=True)
    parser.add_argument("--output", help='output path for SVG', required=True)

    args = parser.parse_args()

    main(args)
