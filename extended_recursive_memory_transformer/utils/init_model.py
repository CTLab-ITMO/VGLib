
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

def create_model(path_weights, len_tokenize, device, name_model='bert-base-uncased'):
    config = BertConfig(
        vocab_size=len_tokenize
    )
    model = BertModel(config)
    model = model.to(device)
    model.load_state_dict(torch.load(path_weights, map_location=device))
    return model

