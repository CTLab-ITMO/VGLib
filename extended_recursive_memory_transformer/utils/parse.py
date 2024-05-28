from transformers import BertModel, BertTokenizer
from collections import defaultdict
import torch
import torch.nn as nn
import importlib
import sys
import os
import sysconfig
import pathlib
import svgutils
import pandas as pd
import ast
import torch
from svgpathtools import svg2paths, wsvg, Path, CubicBezier
import plotly.express as px
import sys
import svgwrite
import svgpathtools
from tokenizers.models import BPE
from transformers import BertTokenizer
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import BertTokenizer, BertModel, BertConfig
from random import randrange
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def save_info_parse(df, path, name, struct, similarity):
    text = [path, name, struct, similarity]
    temp_data = pd.DataFrame([text], columns = ['full_path', 'name', 'struct', 'similarity'])
    df = pd.concat([df, temp_data], axis = 0, ignore_index=True)
    return df


def get_point_str(point, max_width, max_height, shift_x, shift_y):
    num_point_x = float(point.real)
    num_point_x = num_point_x + shift_x
    proc_point_x = num_point_x / max_width * 100
    proc_point_x = int(proc_point_x)
    num_point_y = float(point.imag)
    num_point_y = num_point_y + shift_y
    proc_point_y = num_point_y / max_height * 100
    proc_point_y = int(proc_point_y)
    point_str = ' ' + str(proc_point_x) + ' ' + str(proc_point_y)
    return point_str


def read_data(path):
    file_paths = []
    for root, directories, files in os.walk(path):
        for filename in files:
            file_paths.append(os.path.join(root, filename))
    return file_paths


def parse_svg(path_folder):
    err_read = 0
    read = 0
    k = 0
    err_svg = 0
    df_parse = pd.DataFrame(columns=['full_path', 'name', 'struct'])
    paths_file = read_data(path_folder)
    print("Обнаружено файлов:", len(paths_file))
    for path_file in paths_file:
        name = os.path.basename(path_file)
        print("Начинаю обработку изображения:", name)
        k = k + 1
        try:
            paths, attributes = svg2paths(path_file)
            read = read + 1
        except Exception as e:
            err_read = err_read + 1
            print(e)
            continue
        struct = "<START>"
        paths_new = Path()
        start = True
        for path in paths:
            if len(path) == 0:
                continue
            if start:
                for obj in path:
                    try:
                        len(obj)
                        min_x = paths[0][0][0].real
                        min_y = paths[0][0][0].imag
                        width = min_x
                        height = min_y
                        break
                    except:
                        continue
                start = False

            for obj in path:
                try:
                    len(obj)
                    for point in obj:
                        num_point_x = float(point.real)
                        num_point_y = float(point.imag)
                        if num_point_x < min_x:
                            min_x = num_point_x
                        if num_point_x > width:
                            width = num_point_x
                        if num_point_y < min_y:
                            min_y = num_point_y
                        if num_point_y > height:
                            height = num_point_y
                except:
                    start_x = obj.start.real
                    start_y = obj.start.imag
                    if start_x < min_x:
                        min_x = start_x
                    if start_x > width:
                        width = start_x
                    if start_y < min_y:
                        min_y = start_y
                    if start_y > height:
                        height = start_y
                    radius_x = obj.radius.real
                    radius_y = obj.radius.imag
                    if radius_x < min_x:
                        min_x = radius_x
                    if radius_x > width:
                        width = radius_x
                    if radius_y < min_y:
                        min_y = radius_y
                    if radius_y > height:
                        height = radius_y
                    end_x = obj.end.real
                    end_y = obj.end.imag
                    if end_x < min_x:
                        min_x = end_x
                    if end_x > width:
                        width = end_x
                    if end_y < min_y:
                        min_y = end_y
                    if end_y > height:
                        height = end_y

        max_width = width
        max_height = height
        shift_x = -min_x
        shift_y = -min_y
        max_width = max_width + shift_x
        max_height = max_height + shift_y
        for path in paths:
            if len(path) == 0:
                continue
            struct = struct + ' ' + '<PATH>'
            for obj in path:
                try:
                    len(obj)
                except:
                    name_obj = 'ARC'
                    struct = struct + ' ' + name_obj
                    start_x = obj.start.real
                    start_x = start_x + shift_x
                    proc_start_x = int(start_x / max_width * 100)
                    start_y = obj.start.imag
                    start_y = start_y + shift_y
                    proc_start_y = int(start_y / max_height * 100)
                    point = str(proc_start_x) + ' ' + str(proc_start_y)
                    struct = struct + ' ' + point
                    radius_x = obj.radius.real
                    radius_x = radius_x + shift_x
                    proc_radius_x = int(radius_x / max_width * 100)
                    radius_y = obj.radius.imag
                    radius_y = radius_y + shift_y
                    proc_radius_y = int(radius_y / max_height * 100)
                    point = str(proc_radius_x) + ' ' + str(proc_radius_y)
                    struct = struct + ' ' + point
                    end_x = obj.end.real
                    end_x = end_x + shift_x
                    proc_end_x = int(end_x / max_width * 100)
                    end_y = obj.end.imag
                    end_y = end_y + shift_y
                    proc_end_y = int(end_y / max_height * 100)
                    point = ' ' + str(proc_end_x) + ' ' + str(proc_end_y)
                    struct = struct + ' ' + point
                    paths_new.append(obj)
                    continue
                if len(obj) == 2:
                    name_obj = 'Line'
                    struct = struct + ' ' + name_obj
                    for point in obj:
                        point_str = get_point_str(point, max_width, max_height, shift_x, shift_y)
                        struct = struct + ' ' + point_str
                    paths_new.append(obj)
                elif len(obj) == 3:
                    name_obj = 'QuadraticBezier'
                    struct = struct + ' ' + name_obj
                    for point in obj:
                        point_str = get_point_str(point, max_width, max_height, shift_x, shift_y)
                        struct = struct + ' ' + point_str
                    paths_new.append(obj)
                elif len(obj) == 4:
                    paths_new.append(obj)
                    name_obj = 'CubicBezier'
                    struct = struct + ' ' + name_obj
                    for point in obj:
                        point_str = get_point_str(point, max_width, max_height, shift_x, shift_y)
                        struct = struct + ' ' + point_str
        if len(struct.split()) < 5:
            print('Мало данных в:', path_file)
            err_svg = err_svg + 1
            continue
        struct = struct + ' ' + '<END>'
        df_parse = save_info_parse(df_parse, path_file, name, struct)
    df_name = os.path.join(path_folder, 'data_info.csv')
    df_parse.to_csv(df_name, index=False)
    print("Парсинг закончен! Данные сохранены в:", df_name)
    return df_parse
