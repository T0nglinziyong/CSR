from .modeling_roberta_ import RobertaSelfAttention, RobertaLayer, RobertaModel
from transformers.models.roberta import modeling_roberta

modeling_roberta.RobertaSelfAttention = RobertaSelfAttention
modeling_roberta.RobertaLayer = RobertaLayer
modeling_roberta.RobertaModel = RobertaModel