import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.preprocessors import get_preprocessor


class Dense(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)
        self.activation_function = kwargs.get('activation_function', nn.ReLU())
        self.configuration_hidden_layers = kwargs.get('configuration_hidden_layers', [64, 64])
        self.num_hidden_layers = len(self.configuration_hidden_layers)

        self.flatten_observation = None
        self.latent_space = None

        observation_size = get_preprocessor(observation_space)(observation_space).size
        action_size = num_outputs

        actor_layers = [nn.Linear(observation_size, self.configuration_hidden_layers[0]), self.activation_function]
        critic_layers = [nn.Linear(observation_size, self.configuration_hidden_layers[0]), self.activation_function]

        for i in range(0, self.num_hidden_layers - 1):
            actor_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i+1]))
            actor_layers.append(self.activation_function)

            critic_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i+1]))
            critic_layers.append(self.activation_function)

        critic_layers.append(nn.Linear(self.configuration_hidden_layers[-1], 1))

        self.actor_layers = nn.Sequential(*actor_layers)
        self.critic_layers = nn.Sequential(*critic_layers)

        self.action_layer = nn.Linear(self.configuration_hidden_layers[-1], action_size)

        self.hook_current_index_layer = None
        self.hook_activations = None
        self.already_initialized_get_latent_space = False

    def forward(self, input_dict, state, seq_lens):
        self.flatten_observation = input_dict['obs_flat']
        self.latent_space = self.actor_layers(self.flatten_observation)

        action = self.action_layer(self.latent_space)
        return action, []

    def value_function(self):
        value = self.critic_layers(self.flatten_observation)
        return torch.reshape(value, [-1])
