import gymnasium
import torch
from ray.experimental.array.remote import shape
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.architectures.create_dense_architecture import create_dense_architecture


class SelfAttention(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(SelfAttention, self).__init__()
        self.input_dimension = input_dimension
        self.query = nn.Linear(self.input_dimension, output_dimension)
        self.key = nn.Linear(self.input_dimension, output_dimension)
        self.value = nn.Linear(self.input_dimension, output_dimension)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dimension ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class Attention(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)

        self.activation_function = kwargs.get('activation_function', nn.LeakyReLU())
        dimension_token = kwargs.get('dimension_token', 16)
        number_heads = kwargs.get('number_heads', 1)

        self.multihead_attention = nn.MultiheadAttention(dimension_token, number_heads, batch_first=True)
        self.self_attention = SelfAttention(2, dimension_token)

        number_value_observation = gymnasium.spaces.utils.flatten(observation_space, observation_space.sample()).shape[0]

        # Action
        self.action_query_projector = nn.Sequential(nn.Linear(2, dimension_token, bias=False), self.activation_function)
        self.action_key_projector = nn.Sequential(nn.Linear(2, dimension_token, bias=False), self.activation_function)
        self.action_value_projector = nn.Sequential(nn.Linear(2, dimension_token, bias=False), self.activation_function)

        self.action_dense_layer = create_dense_architecture(
            input_dimension=dimension_token * number_value_observation,
            shape_layers=[64],
            output_dimension=num_outputs,
            activation_function=self.activation_function,
        )

        # Critic
        self.critic_query_projector = nn.Sequential(nn.Linear(2, dimension_token, bias=False), self.activation_function)
        self.critic_key_projector = nn.Sequential(nn.Linear(2, dimension_token, bias=False), self.activation_function)
        self.critic_value_projector = nn.Sequential(nn.Linear(2, dimension_token, bias=False), self.activation_function)

        self.critic_dense_layer = create_dense_architecture(
            input_dimension=dimension_token * number_value_observation,
            shape_layers=[64],
            output_dimension=1,
            activation_function=self.activation_function,
        )

        self.observation_positional_encoding = None
        self.flatten_observation = None

    def forward(self, input_dict, state, seq_lens):
        self.flatten_observation = input_dict['obs_flat'].unsqueeze(-1)
        positional_encoding = torch.linspace(0, 1, self.flatten_observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(self.flatten_observation.shape[0], -1, -1).to(self.flatten_observation.device)
        self.observation_positional_encoding = torch.cat((self.flatten_observation, positional_encoding), dim=-1)

        # queries = self.action_query_projector(self.observation_positional_encoding)
        # keys = self.action_key_projector(self.observation_positional_encoding)
        # values = self.action_value_projector(self.observation_positional_encoding)
        # attention_output, attention_scores = self.multihead_attention(queries, keys, values)
        # embedding = attention_output.mean(dim=1)

        attention_output = self.self_attention(self.observation_positional_encoding)

        embedding = attention_output.view(attention_output.shape[0], -1)
        action = self.action_dense_layer(embedding)
        return action, []

    def value_function(self):
        positional_encoding = torch.linspace(0, 1, self.flatten_observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(self.flatten_observation.shape[0], -1, -1).to(
            self.flatten_observation.device)
        self.observation_positional_encoding = torch.cat((self.flatten_observation, positional_encoding), dim=-1)

        # queries = self.critic_query_projector(self.observation_positional_encoding)
        # keys = self.critic_key_projector(self.observation_positional_encoding)
        # values = self.critic_value_projector(self.observation_positional_encoding)
        # attention_output, _ = self.multihead_attention(queries, keys, values)
        # embedding = attention_output.mean(dim=1)

        attention_output = self.self_attention(self.observation_positional_encoding)

        embedding = attention_output.view(attention_output.shape[0], -1)
        value = self.critic_dense_layer(embedding)
        return torch.reshape(value, [-1])
