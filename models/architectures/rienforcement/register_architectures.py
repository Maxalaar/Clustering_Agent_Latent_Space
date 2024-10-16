from ray.rllib.models import ModelCatalog

from models.architectures.rienforcement.dense import Dense
from models.architectures.rienforcement.transformer import Transformer


def register_architectures():
    ModelCatalog.register_custom_model('dense', Dense)
    ModelCatalog.register_custom_model('transformer', Transformer)
