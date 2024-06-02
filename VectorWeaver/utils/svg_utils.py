from svgelements import *
import torch
import pickle
import numpy as np


def save_data(data, mask, fill, file):
    pckl = (data.numpy().astype(dtype=np.float32),
            fill.numpy().astype(dtype=np.float16),
            mask.numpy().astype(dtype=np.dtype(bool)))

    with open(file, 'wb') as fout:
        pickle.dump(pckl, fout)


def print_svg(data, fill, name=None):
    def make_point(x, y):
        return Point((x + 1) * 50, (y + 1) * 50)

    s = SVG()
    s.width = '100px'
    s.height = '100px'
    svg_path = Path()
    empty = True
    for i in range(data.shape[0]):
        path = data[i].clone().detach()
        r = ((fill[i][0] + 1) * 128).long()
        g = ((fill[i][1] + 1) * 128).long()
        b = ((fill[i][2] + 1) * 128).long()
        r = torch.clamp(r, 0, 255)
        g = torch.clamp(g, 0, 255)
        b = torch.clamp(b, 0, 255)
        clr = int(r * 256 * 256 + g * 256 + b)
        if fill[i][0] * fill[i][0] + fill[i][1] * fill[i][1] + fill[i][2] * fill[i][2] > 0.05:
            if not empty:
                s.append(svg_path)
            svg_path = Path()
            svg_path.fill = Color(clr)

        empty = False
        svg_path.append(Move(make_point(path[0], path[1])))
        for j in range(0, len(path) - 2, 6):
            svg_path.append(CubicBezier(
                make_point(path[j], path[j + 1]),
                make_point(path[j + 2], path[j + 3]),
                make_point(path[j + 4], path[j + 5]),
                make_point(path[j + 6], path[j + 7]),
            ))
    if not empty:
        s.append(svg_path)
    if name is None:
        return s.string_xml()
    s.write_xml(name)
