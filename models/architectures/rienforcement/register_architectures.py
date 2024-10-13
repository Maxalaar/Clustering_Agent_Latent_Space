from ray.rllib.models import ModelCatalog

from models.architectures.rienforcement.dense import Dense


def register_architectures():
    ModelCatalog.register_custom_model('dense', Dense)
