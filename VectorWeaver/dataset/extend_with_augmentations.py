import pickle
import os
import numpy as np
import torch
from argparse import ArgumentParser
from configs.config import Config
from utils.svg_utils import save_data


def augment_id(data, mask, fill, config):
    return data, mask, fill


def augment_shuffle(data, mask, fill, config):
    perm = torch.randperm(data.size()[0])
    data = data[perm]
    mask = mask[perm]
    fill = fill[perm]
    return data, mask, fill


def augment_move_normal(data, mask, fill, config):
    mv = (torch.normal(0, 0.05, size=(config.paths_number, 2, 1)) * torch.ones((1, config.coordinates_number // 2)))\
        .permute(0, 2, 1).reshape(config.paths_number, config.coordinates_number)
    return (data + mv).reshape(config.paths_number, config.coordinates_number), mask, fill


def augment_cyclic_shift(data, mask, fill, config):
    for i in range(len(data)):
        if abs(data[i][0] - data[i][-2]) < 1e-9 and abs(data[i][1] - data[i][-1]) < 1e-9:
            x = np.random.randint(config.segments_number - 2) * 6 + 6
            data[i] = torch.cat([data[i][x:], data[i][2:x], data[i][x:x+2]], dim=-1)
    return data, mask, fill


def augment_absolute_random(data, mask, fill, config):
    return torch.normal(0, 1, (config.paths_number, config.coordinates_number)), \
        mask, \
        torch.normal(0, 1, (config.paths_number, 3))


last_images = []
def augment_mix_smth(data, mask, fill, config):
    for i in range(len(data)):
        x = np.random.randint(len(last_images))
        data[i] = last_images[x][0][i]
        mask[i] = last_images[x][1][i]
        fill[i] = last_images[x][2][i]
    return data, mask, fill


augments = [
    (augment_id, 1),
    (augment_shuffle, 1),
    (augment_cyclic_shift, 7),
    (augment_absolute_random, 1),
    (augment_move_normal, 1),
    (augment_mix_smth, 4),
]


def main(args):
    global last_images
    config = Config()
    new_file_number = 0
    for source_file in os.listdir(args.data_folder):
        data, fill, mask, = pickle.load(open(os.path.join(args.data_folder, source_file), 'rb'))
        data = torch.Tensor(data)
        fill = torch.Tensor(fill)
        mask = torch.Tensor(mask)

        last_images.append((data, mask, fill))
        last_images = last_images[-5:]

        for aug, numb in augments:
            for cp in range(numb):
                data_, mask_, fill_ = aug(data.clone(), mask.clone(), fill.clone(), config)
                save_data(data_, mask_, fill_, os.path.join(args.output_folder, str(new_file_number)))
                new_file_number += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_folder", help='path to the initial dataset', required=True)
    parser.add_argument("--output_folder", help='output path for prepared dataset with augmentations', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args)
