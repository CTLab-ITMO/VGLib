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
import configparser
from utils import (
    create_tokenizer,
    init_model,
    parse,
    preparation_data,
    use_model,
)

if __name__ == "__name__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_save = config.get('service', 'path_save')
    path_weights = config.get('service', 'path_weights')
    path_folder = config.get('service', 'path_folder')
    
    save = pd.path.join(path_save, 'result.csv')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Используется:", device)
    tokenizer, len_tokenizer = create_tokenizer.update_tokenizer()
    model = init_model.create_model(path_weights, len_tokenizer, device)
    data = parse.parse_svg(path_folder)
    data = preparation_data.tokenize_data(data, tokenizer)
    data = use_model.analiz_data(data, model, tokenizer, device)
    data.to_csv(save, index = False)



