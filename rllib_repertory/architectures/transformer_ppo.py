from typing import Dict, Any, Optional

import gymnasium
import torch
from ray.experimental.array.remote import zeros_like
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from torch.nn import Parameter

from utilities.create_dense_architecture import create_dense_architecture


class TransformerPPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function = self.model_config.get('activation_function', nn.LeakyReLU())
        self.use_multiple_projectors = self.model_config.get('use_multiple_projectors', True)
        self.use_same_encoder_actor_critic = self.model_config.get('use_same_encoder_actor_critic', False)

        dimension_token = self.model_config.get('dimension_token', 16)
        number_heads = self.model_config.get('number_heads', 1)
        dimension_feedforward = self.model_config.get('dimension_feedforward', 32)
        number_transformer_layers = self.model_config.get('number_transformer_layers', 1)
        dropout = self.model_config.get('dropout', 0.1)

        action_token_projector_layer_shapes = self.model_config.get('action_token_projector_layer_shapes', [])
        critic_token_projector_layer_shapes = self.model_config.get('critic_token_projector_layer_shapes', [])
        action_dense_layer_shapes = self.model_config.get('action_dense_layer_shapes', [128, 64, 32, 16])
        critic_dense_layer_shapes = self.model_config.get('critic_dense_layer_shape', [128, 64, 32, 16])

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dimension_token,
            nhead=number_heads,
            dim_feedforward=dimension_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        number_value_observation = gymnasium.spaces.utils.flatten(self.observation_space, self.observation_space.sample()).shape[0]

        # Token Projectors
        if self.use_multiple_projectors:
            self.action_token_projectors = nn.ModuleList([
                create_dense_architecture(
                input_dimension=2,
                shape_layers=action_token_projector_layer_shapes,
                output_dimension=dimension_token,
                activation_function=self.activation_function,
                ) for _ in range(number_value_observation)
            ])
            self.critic_token_projectors = nn.ModuleList([
                create_dense_architecture(
                    input_dimension=2,
                    shape_layers=critic_token_projector_layer_shapes,
                    output_dimension=dimension_token,
                    activation_function=self.activation_function,
                ) for _ in range(number_value_observation)
            ])
        else:
            self.action_token_projector = create_dense_architecture(
                input_dimension=2,
                shape_layers=action_token_projector_layer_shapes,
                output_dimension=dimension_token,
                activation_function=self.activation_function,
            )
            self.critic_token_projector = create_dense_architecture(
                input_dimension=2,
                shape_layers=critic_token_projector_layer_shapes,
                output_dimension=dimension_token,
                activation_function=self.activation_function,
            )

        # Transformers
        self.action_context_token = Parameter(torch.randn(dimension_token), requires_grad=True)
        self.action_layer_transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=number_transformer_layers
        )
        self.critic_context_token = Parameter(torch.randn(dimension_token), requires_grad=True)
        self.critic_layer_transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=number_transformer_layers
        )

        # Dense
        self.action_dense_layers = create_dense_architecture(
            input_dimension=dimension_token,
            shape_layers=action_dense_layer_shapes,
            output_dimension=self.action_dist_cls.required_input_dim(space=self.action_space),
            activation_function=self.activation_function,
        )
        self.critic_dense_layers = create_dense_architecture(
            input_dimension=dimension_token,
            shape_layers=critic_dense_layer_shapes,
            output_dimension=1,
            activation_function=self.activation_function,
        )

        self.to(self.device)
        print(self.__dict__)
        print(self)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        observation = batch[Columns.OBS].unsqueeze(-1)

        positional_encoding = torch.linspace(0, 1, observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(observation.shape[0], -1, -1).to(observation.device)

        observation_positional_encoding = torch.cat((observation, positional_encoding), dim=-1).to(self.device)

        if self.use_multiple_projectors:
            projected_tokens = torch.cat(
                [projector(observation_positional_encoding[:, i, :]).unsqueeze(1) for i, projector in enumerate(self.action_token_projectors)],
                dim=1
            )
        else:
            projected_tokens = self.action_token_projector(observation_positional_encoding)

        tokens = torch.cat(
            [self.action_context_token.expand(observation.size(0), 1, self.action_context_token.size(0)), projected_tokens],
            dim=1
        )

        transformer_output = self.action_layer_transformer_encoder(
            tokens,
        )

        embedding = transformer_output[:, 0]
        action_distribution_inputs = self.action_dense_layers(embedding)

        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        observation = batch[Columns.OBS].unsqueeze(-1)

        positional_encoding = torch.linspace(0, 1, observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(observation.shape[0], -1, -1).to(
            observation.device)

        observation_positional_encoding = torch.cat((observation, positional_encoding), dim=-1).to(self.device)

        if self.use_same_encoder_actor_critic:
            if self.use_multiple_projectors:
                projected_tokens = torch.cat(
                    [projector(observation_positional_encoding[:, i, :]).unsqueeze(1) for i, projector in
                     enumerate(self.action_token_projectors)],
                    dim=1
                )
            else:
                projected_tokens = self.action_token_projector(observation_positional_encoding)

            tokens = torch.cat(
                [self.action_context_token.expand(observation.size(0), 1, self.action_context_token.size(0)),
                 projected_tokens],
                dim=1
            )

            transformer_output = self.action_layer_transformer_encoder(
                tokens,
            )
        else:
            if self.use_multiple_projectors:
                projected_tokens = torch.cat(
                    [projector(observation_positional_encoding[:, i, :]).unsqueeze(1) for i, projector in enumerate(self.critic_token_projectors)],
                    dim=1
                )
            else:
                projected_tokens = self.critic_token_projector(observation_positional_encoding)

            tokens = torch.cat(
                [self.critic_context_token.expand(observation.size(0), 1, self.critic_context_token.size(0)), projected_tokens],
                dim=1
            )
            transformer_output = self.critic_layer_transformer_encoder(
                tokens,
            )

        embedding = transformer_output[:, 0]
        return self.critic_dense_layers(embedding).squeeze(-1)