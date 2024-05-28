
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


def create_prefix(num_token = 5):
    name_prefix = "READ"
    prefix = ""
    for tok_ind in range(num_token - 1, -1, -1):
        token_read = f"<{name_prefix}_{tok_ind}>"
        prefix = prefix + token_read + ' '
    return prefix


def create_struct(data, num_token = 5, split_size = 200):
    postfix = " <WRITE>"
    prefix = create_prefix(num_token)
    for ind in data.index:
        segm_list = []
        original_string = data['struct'][ind]
        original_string = original_string.split()
        split_strings = [original_string[i:i + split_size] for i in range(0, len(original_string), split_size)]
        split_strings = [' '.join(sublist) for sublist in split_strings]
        for segm in split_strings:
            segm_str = prefix + segm + postfix
            segm_list.append(segm_str)
        data.at[ind, 'struct'] = segm_list


def tokenize(data, tokenizer, split_size = 200, num_token = 5):
    max_length = split_size + num_token + 1 + 2
    tokens = [tokenizer(text, padding='max_length', max_length=max_length, return_tensors='pt') for text in df_test['struct']]
    data['tokens'] = tokens
    return data

def tokenize_data(data, tokenizer):
    data = create_struct(data)
    data = tokenize(data, tokenizer)
    return data




