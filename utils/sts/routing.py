import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, List
from transformers.models.bert.modeling_bert import BertSelfAttention
from .utils import *
import math


class MultiHeadLinear(nn.Module):
    def __init__(self, hidden_size, num_heads, num_experts, dropout=0.1):
        super(MultiHeadLinear, self).__init__()
        ffd_head_size = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.nheads = num_heads
        self.num_experts = num_experts
        self.ffd_head_size = ffd_head_size

        self.Whead = nn.Linear(hidden_size, hidden_size)
        self.Wmerge = nn.Linear(hidden_size, hidden_size)
        self.router = nn.Parameter(torch.randn(num_experts, ffd_head_size)) 


        self.NExperts = nn.Linear(ffd_head_size, num_experts * ffd_head_size)
    
    def forward(self, features, condition=None):
        batch_size, hidden_size = features.shape
        shape_1 = (batch_size, self.nheads, self.ffd_head_size)
        shape_2 = (batch_size, self.nheads, self.num_experts, self.ffd_head_size)
        shape_3 = (batch_size, hidden_size)
        
        features = self.Whead(features).view(shape_1)
        features = self.NExperts(features).view(shape_2)

        head_score = torch.einsum("bhed, ed -> bhe", [features, self.router]) / math.sqrt(self.ffd_head_size)
        head_prob = F.softmax(head_score, dim=-1)
        features = torch.einsum("bhed, bhe ->bhd", [features, head_prob])

        features = features.view(shape_3)
        features = self.Wmerge(features)
        return features
    
class SelecterTopk(nn.Module):
    def __init__(self, k=None, p=None) -> None:
        super(SelecterTopk, self).__init__()
        self.k = k
        self.p = p

    def forward(self, score):
        bsz, slen = score.shape

        if self.p == 1:
            return score#, torch.eye(slen, device=score.device).unsqueeze(0).repeat(bsz, 1, 1)
        elif self.p is not None:
            k = torch.sum(score[score!=0]) * self.p / bsz
        else:
            k = self.k
        
        result_tensor = torch.zeros_like(score)
        _, top_indices = torch.topk(score, k, dim=1)
        result_tensor.scatter_(1, top_indices, 1)

        #indices = torch.nonzero(result_tensor, as_tuple=False)
        #indices = indices[:, 1].reshape(bsz, self.k, 1)

        #transposed_matrix = torch.arange(slen, device=indices.device).unsqueeze(0).unsqueeze(1) == indices

        return result_tensor#, transposed_matrix.to(torch.float)
    

class Normalizer(nn.Module):
    def __init__(self, theta=0.3, theta0=4, T=20, beta=0.7, k=None, p=None, temperature=1) -> None:
        super(Normalizer, self).__init__()
        self.temperature = temperature

        self.k = k# select soft k
        self.p = p# select soft percent

        self.theta = theta
        self.theta0 = theta0
        self.beta = beta
        self.T = T

    def forward(self, score, mask=None):
        score *= self.temperature

        if mask is not None:
            score = score.masked_fill(mask==0, float('-inf'))
            
        bsz, seq_length = score.shape
        device = score.device
        if self.k == 1 and self.theta == 0:
            return torch.softmax(score, dim=-1)
        if self.p is not None:
            k = torch.sum(torch.isfinite(score), dim=-1, keepdim=True) * self.p
        else:
            k = self.k2

        a = torch.zeros((bsz, 1), device=device)
        b = torch.zeros((bsz, seq_length), device=device)
        pilot_a = a
        theta = self.theta0

        for _ in range(self.T):
            s = torch.sum(torch.exp((score + b) / theta), dim=-1, keepdim=True)
            pilot_a = theta * torch.log(k / (s + 1e-20))
            #if torch.any(torch.isinf(pilot_a)) or torch.any(torch.isnan(pilot_a)):
            #    break
            a = pilot_a
            b = - torch.relu(score + a)

            theta = max(self.beta * theta, self.theta)
        
        gamma = torch.exp((score + a + b) / self.theta)

        return gamma
    

class ScorerV1(nn.Module):
    def __init__(self, hidden_size, temperature=1, nheads=1, use_condition=False, use_position=False, sent_transform=False) -> None:
        super(ScorerV1, self).__init__()
        self.hidden_size = hidden_size
        self.use_condition = use_condition
        self.use_position = use_position
        self.max_position_embeddings = 128
        self.nheads = nheads
        self.attention_head_size = hidden_size // nheads
        self.temperature = temperature

        self.W = nn.Linear(hidden_size, hidden_size) if use_condition else nn.Linear(hidden_size, 1)
        self.Wsent = nn.Linear(hidden_size, hidden_size) if sent_transform else nn.Identity()
        self.distance_embedding = nn.Embedding(self.max_position_embeddings * 2 - 1,
                                               self.nheads
                                                #self.attention_head_size
                                                )
    
    def transpose_for_scores(self, input_tensor):
        # 将输入张量进行维度变换，以适应注意力权重计算
        batch_size, seq_length, hidden_size = input_tensor.shape
        new_shape = (batch_size, seq_length, self.nheads, hidden_size // self.nheads)
        
        # 对张量进行维度变换
        output_tensor = input_tensor.view(*new_shape)
        
        # 将最后两个维度交换，以便后续计算
        output_tensor = output_tensor.permute(0, 2, 1, 3)
        
        return output_tensor

    def forward(self, X_norm, mask=None, condition=None):
        if self.use_condition is None or (self.use_condition and condition is None):
            count = torch.sum(mask, dim=-1, keepdim=True)
            score = (torch.ones_like(mask) / count).unsqueeze(1).unsqueeze(2)
        elif self.use_condition:
            condition = self.transpose_for_scores(self.W(condition))
            X_norm = self.transpose_for_scores(self.Wsent(X_norm))

            score = torch.matmul(condition, X_norm.transpose(-1, -2))
            
            if self.use_position:
                slength = X_norm.shape[2]
                position_ids_l = torch.arange(slength, dtype=torch.long, device=X_norm.device).view(-1, 1)
                position_ids_r = torch.arange(slength, dtype=torch.long, device=X_norm.device).view(1, -1)
                distance = position_ids_l - position_ids_r

                positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=X_norm.dtype)  # fp16 compatibility

                relative_position_scores = positional_embedding.permute(2, 0, 1).unsqueeze(0)
                #relative_position_scores = torch.tanh(relative_position_scores) + 1

                #relative_position_scores = torch.einsum("bhld,lrd->bhlr", X_norm, positional_embedding)
                score = score + relative_position_scores
            #score = score / math.sqrt(self.attention_head_size)
            #s = torch.einsum("bhcd, bhsd -> bhcs", [condition, X_norm]) / math.sqrt(self.attention_head_size)
            score = score / math.sqrt(self.attention_head_size)
        else:
            score = self.transpose_for_scores(self.W(X_norm)).permute(0, 1, 3, 2) # 0.0046269893646240234
        score = torch.mean(torch.mean(score, dim=1), dim=1)
        score = Normalizer(k=1, theta=0)(score, mask)

        return score
    

class ScorerV2(nn.Module):
    def __init__(self, hidden_size):
        super(ScorerV2, self).__init__()
        self.temperature = 1 / math.sqrt(hidden_size)
        self.nomalize = Normalizer(k=1, theta=0)
        self.bilinear = nn.Linear(hidden_size, 1)

    def forward(self, sentence, mask, condition):
        score = self.bilinear(sentence * condition * self.temperature).squeeze()
        score = self.nomalize(score, mask)
        return score


class RouterBase(nn.Module):
    def __init__(self, hidden_size, temperature=1, k=10, p=None,
                 nheads=1, use_condition=True, use_position=False, sent_transform=False) -> None:
        super(RouterBase, self).__init__()
        self.scorer = ScorerV1(hidden_size, temperature, nheads, use_condition, use_position, sent_transform)
        self.select = SelecterTopk(k, p)

    def forward(self, X_norm, mask=None, condition=None):
        score = self.scorer(X_norm, mask, condition)
        m = self.select(score)
        return m
    
    

class HyperNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W1 = nn.Linear(input_size, hidden_size * input_size)
        self.W2 = nn.Linear(input_size, hidden_size * input_size)


    def forward(self, sentence, condition):
        query1 = self.W1(condition).reshape(-1, self.hidden_size, self.input_size)
        query2 = self.W2(condition).reshape(-1, self.hidden_size, self.input_size)
        Wc = torch.einsum("bkh, bkH -> bhH", [query1, query2])

        output = torch.einsum("bhH, bh -> bH", [Wc, sentence])
        return output
