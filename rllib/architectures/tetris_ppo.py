from typing import Dict, Any, Optional

import torch
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
# from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
# from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


class TetrisPPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function = self.model_config.get('activation_function', nn.LeakyReLU())

        self.active_tetromino_mask_net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Flatten(start_dim=1)
        )

        self.board_net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Flatten(start_dim=1)
        )

        self.holder_net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Flatten(start_dim=1)
        )

        self.queue_net = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            self.activation_function,
            nn.Flatten(start_dim=1)
        )

        input_dense = 1 * 24 * 18 + 1 * 24 * 18 + 1 * 4 * 4 + 1 * 4 * 16
        self.actor_layers = nn.Sequential(
            nn.Linear(input_dense, 128),
            self.activation_function,
            nn.Linear(128, 64),
            self.activation_function,
            nn.Linear(64, self.action_space.n)
        )

        self.critic_layers = nn.Sequential(
            nn.Linear(input_dense, 128),
            self.activation_function,
            nn.Linear(128, 64),
            self.activation_function,
            nn.Linear(64, 1)
        )

        self.to(self.device)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        active_tetromino_mask = batch[Columns.OBS]['active_tetromino_mask'].unsqueeze(1).float().to(self.device)
        board = batch[Columns.OBS]['board'].unsqueeze(1).float().to(self.device)
        holder = batch[Columns.OBS]['holder'].unsqueeze(1).float().to(self.device)
        queue = batch[Columns.OBS]['queue'].unsqueeze(1).float().to(self.device)

        active_tetromino_mask_out = self.active_tetromino_mask_net(active_tetromino_mask)
        board_out = self.board_net(board)
        holder_out = self.holder_net(holder)
        queue_out = self.queue_net(queue)

        combined = torch.cat([active_tetromino_mask_out, board_out, holder_out, queue_out], dim=1)

        action_distribution_inputs = self.actor_layers(combined)
        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        active_tetromino_mask = batch[Columns.OBS]['active_tetromino_mask'].unsqueeze(1).float()
        board = batch[Columns.OBS]['board'].unsqueeze(1).float()
        holder = batch[Columns.OBS]['holder'].unsqueeze(1).float()
        queue = batch[Columns.OBS]['queue'].unsqueeze(1).float()

        active_tetromino_mask_out = self.active_tetromino_mask_net(active_tetromino_mask)
        board_out = self.board_net(board)
        holder_out = self.holder_net(holder)
        queue_out = self.queue_net(queue)

        combined = torch.cat([active_tetromino_mask_out, board_out, holder_out, queue_out], dim=1)

        return self.critic_layers(combined).squeeze(-1)
