from typing import Dict, Any, Optional

import gymnasium
import torch
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

from utilities.create_dense_architecture import create_dense_architecture


# from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
# from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


# class TetrisPPO(TorchRLModule, ValueFunctionAPI):
#     @override(TorchRLModule)
#     def setup(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.activation_function = self.model_config.get('activation_function', nn.LeakyReLU())
#
#         self.active_tetromino_mask_net = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         self.board_net = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         self.holder_net = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         self.queue_net = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
#             self.activation_function,
#             nn.Flatten(start_dim=1)
#         )
#
#         input_dense = 1 * 24 * 18 + 1 * 24 * 18 + 1 * 4 * 4 + 1 * 4 * 16
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

class TetrisPPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function = self.model_config.get('activation_function', nn.LeakyReLU())

        dimension_token = self.model_config.get('dimension_token', 32)
        number_heads = self.model_config.get('number_heads', 1)
        dimension_feedforward = self.model_config.get('dimension_feedforward', 64)
        number_transformer_layers = self.model_config.get('number_transformer_layers', 1)
        dropout = self.model_config.get('dropout', 0.1)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dimension_token,
            nhead=number_heads,
            dim_feedforward=dimension_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        number_value_observation = gymnasium.spaces.utils.flatten(self.observation_space, self.observation_space.sample()).shape[0]

        self.action_token_projector = nn.Linear(2, dimension_token)
        self.action_layer_transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                                      num_layers=number_transformer_layers)
        self.action_dense_layer = create_dense_architecture(
            input_dimension=dimension_token * number_value_observation,
            shape_layers=[128, 64, 32, 16],
            output_dimension=self.action_dist_cls.required_input_dim(space=self.action_space),
            activation_function=self.activation_function,
        )

        self.critic_token_projector = nn.Linear(2, dimension_token)
        self.critic_layer_transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                                      num_layers=number_transformer_layers)

        self.critic_dense_layer = create_dense_architecture(
            input_dimension=dimension_token * number_value_observation,
            shape_layers=[128, 64, 32, 16],
            output_dimension=1,
            activation_function=self.activation_function,
        )

        self.observation_positional_encoding = None
        self.to(self.device)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        observation = batch[Columns.OBS].unsqueeze(-1)
        # print('observation : ' + str(observation.shape))
        positional_encoding = torch.linspace(0, 1, observation.shape[-2]).unsqueeze(-1)
        positional_encoding = positional_encoding.unsqueeze(0).expand(observation.shape[0], -1, -1).to(
            observation.device)
        # print('positional_encoding : ' + str(positional_encoding.shape))
        #
        # print('oki')
        observation_positional_encoding = torch.cat((observation, positional_encoding), dim=-1).to(self.device)
        # print('1) observation_positional_encoding : ' + str(observation_positional_encoding.shape))
        # index = torch.nonzero(observation)
        # # print('index : ' + str(index.shape))
        # observation_positional_encoding = observation_positional_encoding[index]
        # # print('2) observation_positional_encoding : ' + str(observation_positional_encoding.shape))

        tokens = self.action_token_projector(observation_positional_encoding)
        transformer_output = self.action_layer_transformer_encoder(tokens)
        # embedding = torch.mean(transformer_output, dim=1)
        embedding = transformer_output.view(transformer_output.shape[0], -1)
        action_distribution_inputs = self.action_dense_layer(embedding)

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
        # index = torch.nonzero(observation)[:, 1]
        # observation_positional_encoding = observation_positional_encoding[:, index, :]

        tokens = self.critic_token_projector(observation_positional_encoding)
        transformer_output = self.critic_layer_transformer_encoder(tokens)
        # embedding = torch.mean(transformer_output, dim=1)
        embedding = transformer_output.view(transformer_output.shape[0], -1)
        return self.critic_dense_layer(embedding).squeeze(-1)
