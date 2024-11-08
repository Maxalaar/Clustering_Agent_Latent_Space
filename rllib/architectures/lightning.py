import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from lightning.surrogate_policy import SurrogatePolicy


class LightningModel(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)

        self.flatten_observation = None
        if kwargs.get('use_gpu'):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.lightning_module: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(kwargs.get('checkpoint_path'), map_location=torch.device('cpu'))

    def forward(self, input_dict, state, seq_lens):
        self.flatten_observation = input_dict['obs_flat']
        action = self.lightning_module(self.flatten_observation)
        return action, []

    def value_function(self):
        return torch.full((self.flatten_observation.size(0),), float('nan')).to(self.flatten_observation.device)
