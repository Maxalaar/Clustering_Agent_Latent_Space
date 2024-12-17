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


# from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
# from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


# class TetrisPPOCNN(TorchRLModule, ValueFunctionAPI):
#     @override(TorchRLModule)
#     def setup(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.activation_function = self.model_config.get('activation_function', nn.LeakyReLU())
#
#         out_channels_1 = 4
#         out_channels_2 = 2
#
#         self.active_tetromino_mask_net = nn.Sequential(
#             nn.Conv2d(1, out_channels_1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         self.board_net = nn.Sequential(
#             nn.Conv2d(1, out_channels_1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         self.holder_net = nn.Sequential(
#             nn.Conv2d(1, out_channels_1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         self.queue_net = nn.Sequential(
#             nn.Conv2d(1, out_channels_1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         input_dense = out_channels_2 * 24 * 18 + out_channels_2 * 24 * 18 + out_channels_2 * 4 * 4 + out_channels_2 * 4 * 16
#         self.actor_layers = nn.Sequential(
#             nn.Linear(input_dense, 128),
#             self.activation_function,
#             nn.Linear(128, 64),
#             self.activation_function,
#             nn.Linear(64, self.action_space.n)
#         )
#
#         self.critic_layers = nn.Sequential(
#             nn.Linear(input_dense, 128),
#             self.activation_function,
#             nn.Linear(128, 64),
#             self.activation_function,
#             nn.Linear(64, 1)
#         )
#
#         self.to(self.device)
#
#     @override(TorchRLModule)
#     def _forward(self, batch, **kwargs):
#         active_tetromino_mask = batch[Columns.OBS]['active_tetromino_mask'].unsqueeze(1).float().to(self.device)
#         board = batch[Columns.OBS]['board'].unsqueeze(1).float().to(self.device)
#         holder = batch[Columns.OBS]['holder'].unsqueeze(1).float().to(self.device)
#         queue = batch[Columns.OBS]['queue'].unsqueeze(1).float().to(self.device)
#
#         active_tetromino_mask_out = self.active_tetromino_mask_net(active_tetromino_mask)
#         board_out = self.board_net(board)
#         holder_out = self.holder_net(holder)
#         queue_out = self.queue_net(queue)
#
#         combined = torch.cat([active_tetromino_mask_out, board_out, holder_out, queue_out], dim=1)
#
#         action_distribution_inputs = self.actor_layers(combined)
#         return {
#             Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
#         }
#
#     @override(ValueFunctionAPI)
#     def compute_values(
#             self,
#             batch: Dict[str, Any],
#             embeddings: Optional[Any] = None,
#     ) -> TensorType:
#         active_tetromino_mask = batch[Columns.OBS]['active_tetromino_mask'].unsqueeze(1).float()
#         board = batch[Columns.OBS]['board'].unsqueeze(1).float()
#         holder = batch[Columns.OBS]['holder'].unsqueeze(1).float()
#         queue = batch[Columns.OBS]['queue'].unsqueeze(1).float()
#
#         active_tetromino_mask_out = self.active_tetromino_mask_net(active_tetromino_mask)
#         board_out = self.board_net(board)
#         holder_out = self.holder_net(holder)
#         queue_out = self.queue_net(queue)
#
#         combined = torch.cat([active_tetromino_mask_out, board_out, holder_out, queue_out], dim=1)
#
#         return self.critic_layers(combined).squeeze(-1)


class TetrisPPOCNN(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function = self.model_config.get('activation_function', nn.ReLU())

        self.actor_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space.n),
        )

        self.critic_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.to(self.device)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        action_distribution_inputs = self.actor_layers(batch[Columns.OBS].to(self.device).float())

        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:

        return self.critic_layers(batch[Columns.OBS].to(self.device).float()).squeeze(-1)


class TetrisPPOTransformer(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function = self.model_config.get('activation_function', nn.LeakyReLU())

        dimension_token = self.model_config.get('dimension_token', 16)
        number_heads = self.model_config.get('number_heads', 1)
        dimension_feedforward = self.model_config.get('dimension_feedforward', 32)
        number_transformer_layers = self.model_config.get('number_transformer_layers', 1)
        dropout = self.model_config.get('dropout', 0.1)

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

        self.action_context_token = Parameter(torch.randn(dimension_token), requires_grad=True)
        self.action_token_projector = nn.Linear(2, dimension_token)
        self.action_layer_transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                                      num_layers=number_transformer_layers)
        self.action_dense_layers = create_dense_architecture(
            input_dimension=dimension_token,
            shape_layers=action_dense_layer_shapes,
            output_dimension=self.action_dist_cls.required_input_dim(space=self.action_space),
            activation_function=self.activation_function,
        )

        self.critic_context_token = Parameter(torch.randn(dimension_token), requires_grad=True)
        self.critic_token_projector = nn.Linear(2, dimension_token)
        self.critic_layer_transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                                      num_layers=number_transformer_layers)

        self.critic_dense_layers = create_dense_architecture(
            input_dimension=dimension_token,
            shape_layers=critic_dense_layer_shapes,
            output_dimension=1,
            activation_function=self.activation_function,
        )

        self.observation_positional_encoding = None
        self.to(self.device)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        observation = batch[Columns.OBS].unsqueeze(-1)
        src_key_padding_mask = torch.flatten(observation != 0, start_dim=1)
        src_key_padding_mask = torch.cat([torch.tensor([True]).expand(observation.size(0), 1).to(self.device), src_key_padding_mask.bool().to(self.device)], dim=1).to(self.device)

        positional_encoding = torch.linspace(0, 1, observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(observation.shape[0], -1, -1).to(
            observation.device)

        observation_positional_encoding = torch.cat((observation, positional_encoding), dim=-1).to(self.device)

        tokens = torch.cat([self.action_context_token.expand(observation.size(0), 1, self.action_context_token.size(0)), self.action_token_projector(observation_positional_encoding)], dim=1)
        transformer_output = self.action_layer_transformer_encoder(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
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
        src_key_padding_mask = torch.flatten(observation != 0, start_dim=1)
        src_key_padding_mask = torch.cat([torch.tensor([True]).expand(observation.size(0), 1).to(self.device), src_key_padding_mask.bool().to(self.device)], dim=1).to(self.device)

        positional_encoding = torch.linspace(0, 1, observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(observation.shape[0], -1, -1).to(
            observation.device)

        observation_positional_encoding = torch.cat((observation, positional_encoding), dim=-1).to(self.device)

        tokens = torch.cat(
            [self.critic_context_token.expand(observation.size(0), 1, self.action_context_token.size(0)), self.critic_token_projector(observation_positional_encoding)],
            dim=1
        )
        transformer_output = self.critic_layer_transformer_encoder(
            tokens,
            src_key_padding_mask=src_key_padding_mask,
        )

        embedding = transformer_output[:, 0]
        return self.critic_dense_layers(embedding).squeeze(-1)
