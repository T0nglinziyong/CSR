import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
from .utils import *
from .my_encoder import CustomizedEncoder
from .routing import MultiHeadLinear

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers import PreTrainedModel, AutoModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


class BiEncoderForClassification_(PreTrainedModel):
    '''Encoder model with backbone and classification head.'''
    def __init__(self, config):
        super().__init__(config)
        self.backbone = CustomizedEncoder(config)
        '''self.backbone = AutoModel.from_pretrained(
            config.model_name_or_path,
            from_tf=bool('.ckpt' in config.model_name_or_path),
            config=config,
            cache_dir=config.cache_dir,
            revision=config.model_revision,
            use_auth_token=True if config.use_auth_token else None,
            add_pooling_layer=False,
        ).base_model'''

        self.margin = config.margin
        self.layer_score = config.routing_end - 1 if config.layer_score is None else config.layer_score

        classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
                )
        else:
            self.transform = None

        self.pooler = Pooler(config.pooler_type)

        self.multihead_ffd = None # MultiHeadLinear(hidden_size=config.hidden_size, num_heads=8, num_experts=8)
        if config.pooler_type in {'avg_first_last', 'avg_top2'}:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False
        if config.objective == 'mse':
            self.loss_fct_cls = nn.MSELoss
            self.loss_fct_kwargs = {}
        elif config.objective in {'triplet', 'triplet_mse'}:
            self.loss_fct_cls = QuadrupletLoss
            self.loss_fct_kwargs = {'distance_function': lambda x, y: 1.0 - cosine_similarity(x, y)}
        else:
            raise ValueError('Only regression and triplet objectives are supported for BiEncoderForClassification')
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        key_ids=None,

        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        key_ids_2=None,

        input_ids_3=None,
        attention_mask_3=None,
        token_type_ids_3=None,
        position_ids_3=None,
        head_mask_3=None,
        inputs_embeds_3=None,
        key_ids_3=None,
        labels=None,
        **kwargs,
        ):
        bsz = input_ids.shape[0]
        input_ids = self.concat_features(input_ids, input_ids_2, input_ids_3)
        attention_mask = self.concat_features(attention_mask, attention_mask_2, attention_mask_3)
        token_type_ids = self.concat_features(token_type_ids, token_type_ids_2, token_type_ids_3)
        position_ids = self.concat_features(position_ids, position_ids_2, position_ids_3)
        head_mask = self.concat_features(head_mask, head_mask_2, head_mask_3)
        inputs_embeds = self.concat_features(inputs_embeds, inputs_embeds_2, inputs_embeds_3)
        key_ids = self.concat_features(key_ids, key_ids_2, key_ids_3)
       
        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            key_ids=key_ids,
            output_hidden_states=self.output_hidden_states,
            output_attentions=False,
            output_token_scores=True,
            )
        
        features = self.pooler(attention_mask, outputs)

        if self.transform is not None:
            features = self.transform(features)
        if self.multihead_ffd is not None:
            features = self.multihead_ffd(features)

        features_1, features_2 = torch.split(features, bsz, dim=0)  # [sentence1, condtion], [sentence2, condition]

        loss = None
        if self.config.objective in {'triplet', 'triplet_mse'}:
            positives1, negatives1 = torch.split(features_1, bsz // 2, dim=0)
            positives2, negatives2 = torch.split(features_2, bsz // 2, dim=0)
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(positives1, positives2, negatives1, negatives2)
            logits = cosine_similarity(features_1, features_2, dim=1)
            if self.config.objective in {'triplet_mse'} and labels is not None:
                loss += nn.MSELoss()(logits, labels)
            else:
                logits = logits.detach()
        else:
            logits = cosine_similarity(features_1, features_2, dim=1)
            if labels is not None:
                loss = self.loss_fct_cls(**self.loss_fct_kwargs)(logits, labels)
            if key_ids is not None and labels is not None:
                #loss += RankingLoss(margin=self.margin)(outputs.token_scores[self.layer_score][:, :split_posi],
                #                       key_ids[:, :split_posi], attention_mask[:, :split_posi])
                loss += RankingLoss(margin=self.margin)(outputs.token_scores[self.layer_score], key_ids, attention_mask)

        return BiConditionEncoderOutput(
            loss=loss,
            logits=logits,
            token_scores=outputs.token_scores[self.layer_score][:bsz],
            token_scores_2=outputs.token_scores[self.layer_score][bsz:],
        )
    
    def concat_features(self, feature_1=None, feature_2=None, feature_c=None):
        if feature_1 is None or feature_2 is None:
            return None
        if feature_c is not None:
            feature_1 = torch.cat([feature_1, feature_c], dim=1)
            feature_2 = torch.cat([feature_2, feature_c], dim=1)
        return torch.cat([feature_1, feature_2], dim=0)
    
    