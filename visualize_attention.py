from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModel
import numpy as np
#import tensorflow as tf
import torch
import torch.nn.functional as F
import pandas as pd
import os
import seaborn as sns


def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png', figure_title=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    if figure_title is not None:
        ax.set_title(figure_title)
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.close()

def visual_score(input_ids_1, input_ids_2, token_scores, split_posi, tokenizer, id=None, figure_path=None, label=None, predict=None):
    # ids_1: condition; ids_2: sentence
    ids_1 = np.array(input_ids_1, dtype=np.int32)
    ids_2 = np.array(input_ids_2, dtype=np.int32)
    ids = np.concatenate([ids_1, ids_2], axis=0)
    texts = tokenizer.convert_ids_to_tokens(ids)
    title = f"label: {label} predict: {predict}" 

    def replacef(text):
        if text == '<s>':
            return '[CLS]'
        elif text == '</s>':
            return '[SEP]'
        return text.replace('Ġ', '')

    texts = [replacef(text) for text in texts]
    length1 = len(ids_1)
    length2 = len(ids_2)

    score = resize_score(torch.Tensor(token_scores), length1, length2, split_posi)
   
    # # 显示Attention
    attention = score.unsqueeze(0)#[0:1, :]
    df = pd.DataFrame(attention, columns=texts, dtype=float)

    attention_plot(df, annot=True, x_texts=texts, y_texts=[''], figsize=(15, 5), 
                   figure_path=figure_path,
                   figure_name=f'token_score_{id}.png', figure_title=title)
    
def resize_score(scores, l1, l2, split_posi):
    return torch.cat([scores[split_posi:split_posi+l1], scores[0:l2]], dim=0)
    

def visual_attn(input_ids_1, input_ids_2, attentions, split_posi, tokenizer, id=None, figure_path=None, label=None, predict=None):
    ids_1 = np.array(input_ids_1, dtype=np.int32)
    ids_2 = np.array(input_ids_2, dtype=np.int32)
    ids = np.concatenate([ids_1, ids_2], axis=0)
    texts = tokenizer.convert_ids_to_tokens(ids)
    title = f"label: {label} predict: {predict}" 

    def replacef(text):
        if text == '<s>':
            return '[CLS]'
        elif text == '</s>':
            return '[SEP]'
        return text.replace('Ġ', '')

    texts = [replacef(text) for text in texts]
    length1 = len(ids_1)
    length2 = len(ids_2)

    attention_mean = torch.mean(torch.Tensor(attentions), dim=0).squeeze()
    attention_mean = resize_attention(attention_mean, length1, length2, split_posi)

    # 显示Attention, 除了SEP
    attention_plot(attention_mean, annot=True, x_texts=texts, y_texts=texts, figsize=(15, 15),
                    figure_path=figure_path,
                    figure_name=f'attention_weight_sum_{id}.png', figure_title=title)

    # Attention 归一化
    attentions_norm = F.normalize(attention_mean, dim=-1)
    attention_plot(attentions_norm, x_texts=texts, y_texts=texts, annot=True, figsize=(15, 15),
                    figure_path=figure_path,
                    figure_name=f'attention_weight_norm_{id}.png', figure_title=title)
    
    # ==============================================================
    # 按每个Token统计重要度
    # ==============================================================
    # 按Token求和
    attention_token_sum = torch.sum(attention_mean, dim=0, keepdim=True)

    # # 显示Attention
    attention = attention_token_sum#[0:1, :]
    df = pd.DataFrame(attention, columns=texts, dtype=float)

    attention_plot(df, annot=True, x_texts=texts, y_texts=[''], figsize=(15, 5), 
                   figure_path=figure_path,
                   figure_name=f'attention_weight_token_score_{id}.png', figure_title=title)
    

    # 显示Attention, 除了SEP, 规范化
    attentions_norm = F.normalize(attention_token_sum, dim=-1)
    attention = attentions_norm#[0:1, :]

    df = pd.DataFrame(attention, columns=texts, dtype=float)
    attention_plot(df, x_texts=texts, y_texts=[''], annot=True, figsize=(15, 5),
                    figure_path=figure_path,
                    figure_name=f'attention_weight_token_norm_{id}.png', figure_title=title)


def resize_attention(attentions, l1, l2, split_posi):
    attentions_ = torch.cat([attentions[0:l1], attentions[split_posi:split_posi+l2]], dim=0)
    attentions = torch.cat([attentions_[:, 0:l1], attentions_[:, split_posi:split_posi+l2]], dim=1)
    return attentions



 