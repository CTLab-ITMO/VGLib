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


def test_model(data, model, tokenizer, device, num_token=5):
    list_hidden = []
    token_write_index = None
    token_read_index = []
    token_write = '<WRITE>'
    name_read = 'READ'
    token_write = tokenizer.convert_tokens_to_ids(token_write)
    token_read_list = []
    for tok_ind in range(0, num_token):
        token_read = f"<{name_read}_{tok_ind}>"
        token_read = tokenizer.convert_tokens_to_ids(token_read)
        token_read_list.append(token_read)
    token_end = tokenizer.convert_tokens_to_ids("<END>")
    model.eval()
    proc = int(len(data)/100)
    num_proc = 0
    with torch.no_grad():
        for ind in data.index:
            if ind % proc == 0:
                num_proc = num_proc + 1
                print(f"Завершено на: {num_proc}%")
            anchor_struct = data['tokens'][ind]
            anchor_struct = anchor_struct.to(device)
            id_anchor = anchor_struct['input_ids']
            id_list_anchor = [id_anchor[i, :] for i in range(id_anchor.size(0))]
            token_type_ids_anchor = anchor_struct['token_type_ids']#.squeeze(1)
            token_type_list_anchor = [token_type_ids_anchor[i, :] for i in range(token_type_ids_anchor.size(0))]
            attention_mask_anchor = anchor_struct['attention_mask']
            attention_mask_list_anchor = [attention_mask_anchor[i, :] for i in range(attention_mask_anchor.size(0))]
            if len(token_read_index) == 0:
                    for ind in range(0, len(token_read_list)):
                        token_read_index.append(torch.where(id_anchor == token_read_list[ind])[1][0])
                    token_write_index = torch.where(id_anchor == token_write)[1][0]
            hidden_outputs_anchor = segment_analysis(model, id_list_anchor, token_type_list_anchor, attention_mask_list_anchor, token_read_index, token_write_index, token_end, num_token)
            list_hidden.append(hidden_outputs_anchor)
    return list_hidden


def segment_analysis(model, id_list, token_type_ids_list, attention_mask_list, token_read_index, token_write_index, token_end, num_token):
    for num_segm in range(0, len(id_list)):
        torch.cuda.empty_cache()
        id_1 = id_list[num_segm].unsqueeze(0)
        token_type_ids_1 = token_type_ids_list[num_segm].unsqueeze(0)
        attention_mask_1 = attention_mask_list[num_segm].unsqueeze(0)
        if num_segm == 0:
            last_hidden_state_1, outputs_1 = model(input_ids = id_1, attention_mask = attention_mask_1,
                                                   token_type_ids = token_type_ids_1, return_dict = False)
            # забираем скрытые представления токена памяти записи
            last_hidden_mem_1 = last_hidden_state_1[0, token_write_index, :]
            # тут содержатся итоговый глобальный контекст который аккамулируется в мем
            stack_mem = torch.zeros(num_token, last_hidden_mem_1.shape[0])
            # Добавление нового значения в начало стека памяти
            stack_mem[0, :] = last_hidden_mem_1.clone()
            result_output = outputs_1.clone()
            hidden_outputs_1 = last_hidden_mem_1.clone()
        else:
            # делаем тестовую обработку для получения скрытого представления сегмента
            current_hidden_state_1, outputs_1 = model(input_ids = id_1, attention_mask = attention_mask_1,
                                                      token_type_ids = token_type_ids_1, return_dict = False)
            # подставляем контекст прошлого сегмента
            for ind in range(0, len(token_read_index)):
                hidden_mem = stack_mem[ind]
                current_hidden_state_1[:, token_read_index[ind], :] = hidden_mem
            # делаем новую обработку сегмента для получения его состояния с учетом предыдущего контекста
            last_hidden_state_1, outputs_1 = model(inputs_embeds=current_hidden_state_1, attention_mask = attention_mask_1,
                                                   token_type_ids = token_type_ids_1, return_dict = False)
            # забираем контекст
            last_hidden_mem_1 = last_hidden_state_1[0, token_write_index, :]
            stack_mem[1:] = stack_mem[:-1].clone()
            stack_mem[0] = last_hidden_mem_1.clone()
    return outputs_1


def analiz_data(data, model, tokenizer, device):
    result = test_model(data, model, tokenizer, device)
    data['hidden'] = None
    for ind in range(0, len(result)):
        tens = result[ind].to('cpu')
        data.at[ind, 'hidden'] = tens.numpy()
    return data