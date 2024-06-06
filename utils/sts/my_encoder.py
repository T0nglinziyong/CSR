import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers import PreTrainedModel, AutoModel
import math
#from .attn_dropout import *
from .routing import *
from .utils import *
import time
import functools

class CustomizedEncoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.attn_type = config.mask_type
        self.attn_type_2 = config.mask_type_2 \
            if config.mask_type_2 is not None else config.mask_type
        self.rout_start = config.routing_start
        self.rout_end = config.routing_end
        self.router_type = config.router_type
        self.use_attn = config.use_attn

        hidden_size = config.hidden_size

        backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            ignore_mismatched_sizes=True,
        ).base_model
        
        #for param in backbone.parameters():
        #    param.requires_grad = False
        
        # the original Roberta only has one token_type_embedding
        self.embeddings = backbone.embeddings
        self.embeddings.token_type_embedding = nn.Embedding(2, config.hidden_size)
        self.embeddings.token_type_embedding.weight.data = backbone.embeddings.token_type_embeddings.weight.data.repeat(2, 1)

        self.encoder = backbone.encoder.layer
        self.pooler = None
        for i in range(config.routing_start, config.routing_end):
            self.encoder[i].add_module("Router", ScorerV1(hidden_size=config.hidden_size, temperature=config.temperature, nheads=16, 
                                                        use_condition=True, use_position=False, sent_transform=True))
            self.encoder[i].add_module("RouterV2", ScorerV2(hidden_size=config.hidden_size))
            
            self.encoder[i].add_module("LightAttn", BertSelfAttention(hidden_size, nheads=16, dropout=0.1))
            self.encoder[i].add_module("Adapter", nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(config.hidden_dropout_prob)))

            self.encoder[i].add_module("FFD", nn.Sequential(nn.Linear(hidden_size, hidden_size*2), nn.ReLU(), nn.Dropout(config.hidden_dropout_prob), nn.Linear(hidden_size*2, hidden_size)))
            #self.encoder[i].add_module("FFD", nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()))
            
            self.encoder[i].add_module("dropout", nn.Dropout(0.1))
            self.encoder[i].add_module("LayerNorm", nn.LayerNorm(hidden_size))
            self.encoder[i].add_module("ReLU", nn.ReLU())
            #self.encoder[i].LightAttn.load_state_dict(self.encoder[i].attention.self.state_dict())
            #self.encoder[i].Adapter.load_state_dict(self.encoder[i].attention.output.state_dict())

            self.encoder[i].Router.W.load_state_dict(self.encoder[i].attention.self.query.state_dict())
            self.encoder[i].Router.Wsent.load_state_dict(self.encoder[i].attention.self.key.state_dict())

        for layer in self.encoder:
            layer.use_router = hasattr(layer, "Router")

    def get_embedding(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)

        '''if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.embeddings.padding_idx)'''

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        embedding_output = self.embeddings(
            input_ids=input_ids, #word_embeddings维度是(50265, 1024)
            position_ids=position_ids, # position_embeddings维度是(512, 1024) 输入为None
            token_type_ids=token_type_ids, # token_type_embeddings的维度是(1, 1024) 输入为全0
            inputs_embeds=inputs_embeds,
        )

        extended_attention_mask = self.manip_attention_mask(attention_mask, manip_type=self.attn_type)

        return embedding_output, extended_attention_mask, attention_mask
        
    def self_attention(
        self, attention,
        hidden_states: torch.Tensor, # 输入的隐藏状态，即模型的输入特征。
        attention_mask: Optional[torch.FloatTensor] = None, # 形状为 [batch_size, sequence_length]，其中非零值表示需要注意的位置。
        head_mask: Optional[torch.FloatTensor] = None, # 用于掩盖（mask）某些注意力头，使模型在计算注意力时忽略这些头。形状为 [num_attention_heads, num_attention_heads]。
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 如果是用于跨注意力模块，这是来自编码器的注意力掩码。
        encoder_attention_mask: Optional[torch.FloatTensor] = None, 
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = attention.query(hidden_states)
    
        if encoder_hidden_states is not None:
            key_layer = attention.key(encoder_hidden_states)
            value_layer = attention.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            key_layer = attention.key(hidden_states)
            value_layer = attention.value(hidden_states)
        
        # Change the tensor to[batch_size, num_head, seq_length, atten_head_size]
        query_layer = attention.transpose_for_scores(mixed_query_layer) 
        key_layer = attention.transpose_for_scores(key_layer)
        value_layer = attention.transpose_for_scores(value_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # shape = [batch, num_head, query_length, key_length]

        #'RobertaSelfAttention' object has no attribute 'distance_embedding'
        if attention.position_embedding_type == "relative_key" or attention.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = attention.distance_embedding(distance + attention.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(attention.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores_ = attention_scores + attention_mask 
        else:
            attention_scores_ = attention_scores

        attention_probs = nn.functional.softmax(attention_scores_, dim=-1)
        
        attention_probs = attention.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # PyTorch中的某些操作，如转置、视图操作（例如view()）、某些索引操作等，可能会导致张量不再是连续的，这时可以使用.contiguous()方法将其转换为连续的张量
        new_context_layer_shape = context_layer.size()[:-2] + (attention.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_scores, attention_probs) if output_attentions else (context_layer, attention_scores)
        return outputs

    def attention_forward(
        self, layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        key_ids=None,
        org_mask=None,
    ) -> Tuple[torch.Tensor]:
        attention = layer.attention

        self_outputs = self.self_attention(
            attention.self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=hidden_states, 
            encoder_attention_mask=attention_mask, 
            output_attentions=output_attentions or True,
        )
        self_output = self_outputs[0]

        m, token_score = self.get_router_score(
            layer, 
            features=hidden_states, 
            attention_mask=attention_mask, 
            attention_score=self_outputs[1], 
            attention_prob=self_outputs[2],
            org_mask=org_mask, 
            router_type=self.router_type
        )
        conditional_output = self.router_forward(layer, self_output, org_mask) * m #* key_ids.unsqueeze(-1)
        attention_output = attention.output(self_output, hidden_states + conditional_output)# linear + dropout + layernorm
        outputs = (attention_output,) + self_outputs[2:]  + (token_score,) # add attentions if we output them
        return outputs
    
    def layer_forward(
        self, layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        org_mask=None,
        )-> Tuple[torch.Tensor]:
        self_attention_outputs = self.attention_forward(
            layer,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            key_ids=key_ids,
            org_mask=org_mask,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        intermediate_output = layer.intermediate(attention_output) # linear + act
        layer_output = layer.output(intermediate_output, attention_output)# linear + dropout + layernorm

        outputs = (layer_output,) + outputs

        return outputs

    def encoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_token_scores: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        org_mask=None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_token_scores = () if output_token_scores else None

        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = self.layer_forward(
                layer_module,
                hidden_states,
                attention_mask,
                layer_head_mask,
                key_ids=key_ids,
                output_attentions=output_attentions,
                org_mask=org_mask,
            )

            hidden_states = layer_outputs[0]
           
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_token_scores:
                all_token_scores = all_token_scores + (layer_outputs[-1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_token_scores,
                ] if v is not None )
        return MyEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            token_scores=all_token_scores,
        )
    
    def manip_attention_mask(self, mask, split_posi=None, manip_type=0, qlen=None):
        bsz, slen = mask.shape
        qlen = slen if qlen is None else qlen
        split_posi = slen // 2

        mask_expanded = mask.unsqueeze(1)
        mask_3d = mask_expanded.repeat(1, qlen, 1)

        if manip_type == 0:
            pass
        
        elif manip_type == 1:
            # cls只能在自己域内交互
            mask_3d[:, 0, split_posi :] = 0
            mask_3d[:, split_posi, : split_posi] = 0
        
        elif manip_type == 2:
            # condition只能与自身交互
            mask_3d[:, split_posi : ,  : split_posi] = 0

        elif manip_type == 3:
            # sentence只能与自身交互
            mask_3d[:, : split_posi, split_posi :] = 0
        
        elif manip_type == 4:
            # 每个模块之间不直接进行交互
            mask_3d[:, : split_posi, split_posi :] = 0
            mask_3d[:, split_posi :, : split_posi] = 0

        elif manip_type == 5:
            # condition cls在域内交互
            mask_3d[:, split_posi ,  : split_posi] = 0

        elif manip_type == 6:
            # sentence cls在域内交互
            mask_3d[:, 0, split_posi :] = 0
        elif manip_type == 7:
            # condition cls在域内交互
            mask_3d[:, split_posi ,  : split_posi] = 0
            mask_3d[:, :, split_posi] = 0
            mask_3d[:, split_posi, split_posi] = 1
            

        return self.get_extended_attention_mask(mask_3d, (bsz, qlen))
        
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_token_scores: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else True
        output_token_scores = output_token_scores if output_token_scores is not None else True

        embedding_output, default_attention_mask, attention_mask \
            = self.get_embedding(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, )
        
        encoder_outputs = self.encoder_forward(
            embedding_output,
            attention_mask=default_attention_mask,
            head_mask=head_mask,
            key_ids=key_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_token_scores=output_token_scores,
            return_dict=return_dict,
            org_mask=attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
 
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return MyModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            token_scores=encoder_outputs.token_scores,
        )
    
    def get_router_score(
        self, layer,
        features: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        attention_score: Optional[torch.FloatTensor] = None,
        attention_prob: Optional[torch.FloatTensor] = None,
        org_mask = None,
        router_type = 2,
    ):
        input_shape = features.size()[:-1]
        split_pos = input_shape[1] // 2
        word_count = torch.sum(org_mask, dim=-1, keepdim=True)
        if not layer.use_router:
            return 0, torch.mean(attention_prob, dim=1)[:, 0] * word_count #torch.ones_like(org_mask)#
        
        
        if router_type == 0:
            mask = org_mask
            mask[:, split_pos:] = 0
            word_count = torch.sum(mask, dim=-1, keepdim=True)
            m = layer.Router(features, mask, condition=features[:, split_pos:split_pos+1])
            s = m * word_count

        elif router_type == 1:
            mask = org_mask
            mask[:, split_pos:] = 0
            word_count = torch.sum(mask, dim=-1, keepdim=True)
            m = layer.RouterV2(features, mask, condition=features[:, split_pos:split_pos+1])
            s = m * word_count

        elif router_type == 2:
            m = torch.mean(torch.mean(attention_prob, dim=1), dim=1)
            s = m * word_count

        elif router_type == 3:

            m = torch.mean(attention_prob, dim=1)[:, split_pos]
            m[:, split_pos:] = 0
            m /= torch.sum(m, dim=-1, keepdim=True)
            s = m  * word_count
            #s[:, split_pos:] = 1

        elif router_type == 4:
            attention_prob = torch.mean(attention_prob, dim=1)
            m = attention_prob[:, split_pos, :split_pos]
            m2 = attention_prob[:, 0, split_pos:]
            m = torch.cat([m, m2], dim=-1)
            s = m * word_count
        
        return m.unsqueeze(-1), s
    
    def router_forward(self, layer, features, org_mask):
        if not layer.use_router:
            return 0
        # Type1: Only FFD
        if not self.use_attn:
            Z_cond = layer.FFD(features)
            return Z_cond

        # Type2: LightAttn + FFD
        X_routed = features
        #X_routed = torch.einsum("bkn, bnd ->bkd", [P, feature]) 
        mask_routed = self.manip_attention_mask(org_mask, manip_type=self.attn_type_2)
        
        Z_cond_ = layer.LightAttn(
            hidden_states=X_routed,
            encoder_hidden_states=features,
            encoder_attention_mask=mask_routed,
        )[0]
        Z_cond = layer.Adapter(Z_cond_)
        
        # Type3: With LayerNorm
        #Z_cond = layer.LayerNorm(Z_cond + X_routed)
        #Z_cond = torch.einsum("bkn, bkd -> bnd", [P, Z_cond])

        #Z_cond = layer.LayerNorm(Z_cond * m.unsqueeze(-1) + feature)
        return Z_cond
