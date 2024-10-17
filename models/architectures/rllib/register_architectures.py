from ray.rllib.models import ModelCatalog

from models.architectures.rllib.attention import Attention
from models.architectures.rllib.dense import Dense
from models.architectures.rllib.transformer import Transformer


def register_architectures():
    ModelCatalog.register_custom_model('dense', Dense)
    ModelCatalog.register_custom_model('transformer', Transformer)
    ModelCatalog.register_custom_model('attention', Attention)
