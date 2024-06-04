import torch
from torch import nn
from torch.nn.functional import cosine_similarity, sigmoid
from transformers import TrainerCallback
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity, sigmoid
from typing import Union, Tuple, Optional
from dataclasses import dataclass
from transformers.utils import ModelOutput
import math

# Pooler class. Copied and adapted from SimCSE code
class Pooler(nn.Module):
    '''
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    '''
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last', 'routing'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs=None, last_hidden=None, pooler_output=None, hidden_states=None, pooler_type=None):
        if outputs is not None:
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        pooler_type = self.pooler_type if pooler_type is None else pooler_type

        if pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif pooler_type == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1) + 1e-10)
        elif pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_map: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ConditionEncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BiConditionEncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_scores_2: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class MyEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class MyModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_scores: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MyExtractOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class QuadrupletLoss:
    def __init__(self, distance_function, margin=1.0):
        'A cosine distance margin quadruplet loss'
        self.margin = margin
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg = self.distance_function(neg1, neg2)
        loss = torch.clamp_min(self.margin + dist_pos - dist_neg, 0)
        return loss.mean()

  
class InfoNCELoss:
    def __init__(self, temperature=0.7, distance_function=None):
        'A cosine distance margin quadruplet loss'
        self.temperature = temperature
        self.distance_function = distance_function

    def __call__(self, pos1, pos2, neg1, neg2):
        dist_pos = self.distance_function(pos1, pos2)
        dist_neg1 = self.distance_function(pos1, neg2)
        dist_neg2 = self.distance_function(pos2, neg1)

        dist_pos = torch.exp(dist_pos / self.temperature)
        dist_neg1 = torch.exp(dist_neg1 / self.temperature)
        dist_neg2 = torch.exp(dist_neg2 / self.temperature)
        loss = -torch.log(dist_pos / (dist_neg1 + dist_neg2))
        return loss.mean()

    
class RankingLoss:
    def __init__(self, margin=0.1):
        self.margin = margin

    def __call__(self, token_scores, labels, masks=None):
        total_loss = 0.0
        valid_batch = 0
        for token_score, label, mask in zip(token_scores, labels, masks):
            key_num = torch.sum(label)
            if key_num == 0:
                continue
            else:
                valid_batch += 1

            token_score = token_score[mask==1]
            label = label[mask==1]
            
            positive_indices = label.nonzero(as_tuple=True)
            negative_indices = (label == 0).nonzero(as_tuple=True)

            # 获取监督信号为1的token的注意力分数
            positive_score = token_score[positive_indices]
            # 获取监督信号为0的token的注意力分数
            negative_score = token_score[negative_indices]

            # 计算差异
            differences = negative_score.view(-1, 1) - positive_score.view(1, -1)
            loss = torch.clamp(differences + self.margin, min=0)
            total_loss += torch.mean(loss)
        if valid_batch == 0:
            return 0 #torch.tensor(0.0, dtype=token_scores.dtype, device=token_scores.device)
        return total_loss / valid_batch
    

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, nheads, dropout, position_embedding_type=None, max_position_embeddings=128):
        super().__init__()
        if hidden_size % nheads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({nheads})"
            )

        self.num_attention_heads = nheads
        self.attention_head_size = int(hidden_size / nheads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.position_embedding_type = "position_embedding_type"

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * max_position_embeddings - 1, self.attention_head_size)



    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor, # 输入的隐藏状态，即模型的输入特征。
        attention_mask: Optional[torch.FloatTensor] = None, # 用于屏蔽（mask）某些位置的注意力，防止模型关注到填充（padding）的部分。
        # 这是一个可选的参数，可以是一个二维张量，形状为 [batch_size, sequence_length]，其中非零值表示需要注意的位置。
        head_mask: Optional[torch.FloatTensor] = None, # 用于掩盖（mask）某些注意力头，使模型在计算注意力时忽略这些头。形状为 [num_attention_heads, num_attention_heads]。
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 如果是用于跨注意力模块，这是来自编码器的注意力掩码。
        encoder_attention_mask: Optional[torch.FloatTensor] = None, 
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            key_layer = self.key(encoder_hidden_states)
            value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

