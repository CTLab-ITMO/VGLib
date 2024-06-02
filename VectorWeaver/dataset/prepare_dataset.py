import os
from svgelements import *
import torch
from argparse import ArgumentParser
from dataset.exceptions import SvgPrepareException, SvgParseException, \
    SvgUnknownElement, SvgToManyPaths, SvgToManySegments
from configs.config import Config
from utils.svg_utils import save_data


def extend_path(path, config):
    if len(path) == 0:
        return [Point(0.0, 0.0) for i in range(config.segments_number * 3 + 1)], False
    while len(path) < config.segments_number * 3 + 1:
        nw = [path[0]]
        cur_len = len(path)
        for i in range(0, len(path) - 1, 3):
            if cur_len < config.segments_number * 3 + 1:
                Ax, Ay = path[i].x, path[i].y
                Bx, By = path[i+1].x, path[i+1].y
                Cx, Cy = path[i+2].x, path[i+2].y
                Dx, Dy = path[i+3].x, path[i+3].y
                Ex, Ey = (Ax + Bx) / 2, (Ay + By) / 2
                Fx, Fy = (Bx + Cx) / 2, (By + Cy) / 2
                Gx, Gy = (Cx + Dx) / 2, (Cy + Dy) / 2
                Hx, Hy = (Ex + Fx) / 2, (Ey + Fy) / 2
                Jx, Jy = (Fx + Gx) / 2, (Fy + Gy) / 2
                Kx, Ky = (Hx + Jx) / 2, (Hy + Jy) / 2

                nw.append(Point(Ex, Ey))
                nw.append(Point(Hx, Hy))
                nw.append(Point(Kx, Ky))
                nw.append(Point(Jx, Jy))
                nw.append(Point(Gx, Gy))
                nw.append(Point(Dx, Dy))

                cur_len += 3
            else:
                nw.append(path[i+1])
                nw.append(path[i+2])
                nw.append(path[i+3])
        path = nw
    return path, True


def parse_svg(source_svg_file, config):
    try:
        svg = SVG.parse(source_svg_file)
    except Exception as e:
        raise SvgParseException(e)

    prepared_paths = []
    for element in svg.elements():
        current_path = []

        def add_prepared_line_to(end):
            start = current_path[-1]
            control1 = Point(start.x + (end.x - start.x) / 3, start.y + (end.y - start.y) / 3)
            control2 = Point(start.x + 2 * (end.x - start.x) / 3, start.y + 2 * (end.y - start.y) / 3)
            current_path.append(control1)
            current_path.append(control2)
            current_path.append(end)

        if isinstance(element, Rect):
            current_path.append(Point(element.x, element.y))
            add_prepared_line_to(Point(element.x + element.width, element.y))
            add_prepared_line_to(Point(element.x + element.width, element.y + element.height))
            add_prepared_line_to(Point(element.x, element.y + element.height))
            add_prepared_line_to(Point(element.x, element.y))
            prepared_paths.append([element.fill.rgb, current_path.copy()])

        elif isinstance(element, Circle):
            radius = element.rx
            c = 0.55191502449 * radius

            current_path.append(Point(element.cx, element.cy + radius))
            current_path.append(Point(element.cx + c, element.cy + radius))
            current_path.append(Point(element.cx + radius, element.cy + c))
            current_path.append(Point(element.cx + radius, element.cy))

            current_path.append(Point(element.cx + radius, element.cy - c))
            current_path.append(Point(element.cx + c, element.cy - radius))
            current_path.append(Point(element.cx, element.cy - radius))

            current_path.append(Point(element.cx - c, element.cy - radius))
            current_path.append(Point(element.cx - radius, element.cy - c))
            current_path.append(Point(element.cx - radius, element.cy))

            current_path.append(Point(element.cx - radius, element.cy + c))
            current_path.append(Point(element.cx - c, element.cy + radius))
            current_path.append(Point(element.cx, element.cy + radius))

            prepared_paths.append([element.fill.rgb, current_path.copy()])

        elif isinstance(element, Polygon):
            current_path.append(element.points[0])
            for i in range(1, len(element.points)):
                add_prepared_line_to(element.points[i])

            prepared_paths.append([element.fill.rgb, current_path.copy()])

        elif isinstance(element, Path):
            fill = element.fill.rgb
            for subelement in element:
                if isinstance(subelement, Move):
                    if len(current_path) > 0:
                        prepared_paths.append([fill, current_path.copy()])
                        current_path.clear()
                        fill = -1
                    current_path.append(subelement.end)
                elif isinstance(subelement, CubicBezier):
                    current_path.append(subelement.control1)
                    current_path.append(subelement.control2)
                    current_path.append(subelement.end)
                elif isinstance(subelement, Line):
                    add_prepared_line_to(subelement.end)
                elif isinstance(subelement, Close):
                    pass
                else:
                    raise SvgUnknownElement("unknown element: " + type(subelement).__name__)
            prepared_paths.append([fill, current_path.copy()])
        elif isinstance(element, Group) or isinstance(element, SVG):
            pass
        else:
            raise SvgUnknownElement("unknown element: " + type(element).__name__)

    if len(prepared_paths) > config.paths_number:
        raise SvgToManyPaths()
    if max([len(path) for fill, path in prepared_paths]) > config.segments_number * 3 + 1:
        raise SvgToManySegments()

    maxCoordinate = max([max([max(point.x, point.y) for point in path]) for fill, path in prepared_paths])
    minCoordinate = min([min([min(point.x, point.y) for point in path]) for fill, path in prepared_paths])

    # normalize coordinates to [-0.9, 0.9]
    for fill, path in prepared_paths:
        for point in path:
            point.x = ((point.x - minCoordinate) / (maxCoordinate - minCoordinate) - 0.5) * 2 * 0.9
            point.y = ((point.y - minCoordinate) / (maxCoordinate - minCoordinate) - 0.5) * 2 * 0.9

    # fake paths
    while len(prepared_paths) < config.paths_number:
        prepared_paths.append([0, []])

    datas = []
    masks = []
    fills = []
    for fill, path in prepared_paths:
        new_path, fake_path = extend_path(path, config)
        datas.append(torch.cat([torch.Tensor([x, y]) for x, y in new_path]))
        masks.append(torch.Tensor([fake_path]))
        r = fill // 256 // 256 if fill != -1 else 128
        g = fill // 256 % 256 if fill != -1 else 128
        b = fill % 256 if fill != -1 else 128
        r = r / 128 - 1
        g = g / 128 - 1
        b = b / 128 - 1
        fills.append(torch.Tensor([r, g, b]))
    return torch.stack(datas), torch.stack(masks), torch.stack(fills)


def main(args):
    config = Config()
    bad_files = 0
    for file_number, source_svg_file in enumerate(os.listdir(args.data_folder)):
        try:
            data, mask, fill = parse_svg(os.path.join(args.data_folder, source_svg_file), config)
            save_data(data, mask, fill, os.path.join(args.output_folder, str(file_number)))
            print(source_svg_file + " parsed", "bad files:", bad_files, "all files:", file_number)
        except SvgPrepareException as e:
            print(e.__class__.__name__, e, "bad files:", bad_files, "all files:", file_number)
            bad_files += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_folder", help='path to the folder with SVGs', required=True)
    parser.add_argument("--output_folder", help='output path for prepared dataset', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args)
