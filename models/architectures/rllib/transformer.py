import gymnasium
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.architectures.create_dense_architecture import create_dense_architecture


class Transformer(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)

        self.activation_function = kwargs.get('activation_function', nn.LeakyReLU())

        dimension_token = kwargs.get('dimension_token', 16)
        number_heads = kwargs.get('number_heads', 1)
        dimension_feedforward = kwargs.get('dimension_feedforward', 64)
        number_transformer_layers = kwargs.get('number_transformer_layers', 1)
        dropout = kwargs.get('dropout', 0.1)

        number_value_observation = gymnasium.spaces.utils.flatten(observation_space, observation_space.sample()).shape[0]

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dimension_token,
            nhead=number_heads,
            dim_feedforward=dimension_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.action_token_projector = nn.Linear(2, dimension_token)
        self.action_layer_transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=number_transformer_layers)
        self.action_dense_layer = create_dense_architecture(
            input_dimension=dimension_token * number_value_observation,
            shape_layers=[64],
            output_dimension=num_outputs,
            activation_function=self.activation_function,
        )

        self.critic_token_projector = nn.Linear(2, dimension_token)
        self.critic_layer_transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=number_transformer_layers)
        self.critic_dense_layer = create_dense_architecture(
            input_dimension=dimension_token * number_value_observation,
            shape_layers=[64],
            output_dimension=1,
            activation_function=self.activation_function,
        )

        self.observation_positional_encoding = None

    def forward(self, input_dict, state, seq_lens):
        observation = input_dict['obs_flat'].unsqueeze(-1)
        positional_encoding = torch.linspace(0, 1, observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(observation.shape[0], -1, -1).to(observation.device)
        self.observation_positional_encoding = torch.cat((observation, positional_encoding), dim=-1)

        tokens = self.action_token_projector(self.observation_positional_encoding)
        transformer_output = self.action_layer_transformer_encoder(tokens)
        embedding = transformer_output.view(transformer_output.shape[0], -1)
        action = self.action_dense_layer(embedding)
        return action, []

    def value_function(self):
        tokens = self.critic_token_projector(self.observation_positional_encoding)
        transformer_output = self.critic_layer_transformer_encoder(tokens)
        embedding = transformer_output.view(transformer_output.shape[0], -1)
        value = self.critic_dense_layer(embedding)
        return torch.reshape(value, [-1])
