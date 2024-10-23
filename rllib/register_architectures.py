from ray.rllib.models import ModelCatalog

from rllib.architectures.attention import Attention
from rllib.architectures.dense import Dense
from rllib.architectures.transformer import Transformer


def register_architectures():
    ModelCatalog.register_custom_model('dense', Dense)
    ModelCatalog.register_custom_model('transformer', Transformer)
    ModelCatalog.register_custom_model('attention', Attention)
