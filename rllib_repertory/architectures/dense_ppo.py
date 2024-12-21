from typing import Dict, Any, Optional

import torch
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from torch.nn.functional import dropout

from utilities.create_dense_architecture import create_dense_architecture


# from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
# from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


class DensePPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function = self.model_config.get('activation_function', nn.ReLU())
        self.layer_normalization = self.model_config.get('layer_normalization', False)
        self.dropout = self.model_config.get('dropout', False)
        self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [64, 64])
        self.num_hidden_layers = len(self.configuration_hidden_layers)

        inpout_size = get_preprocessor(self.observation_space)(self.observation_space).size
        output_size = self.action_dist_cls.required_input_dim(space=self.action_space)

        self.actor_layers = create_dense_architecture(
            inpout_size,
            self.configuration_hidden_layers,
            output_size,
            self.activation_function,
            layer_normalization=self.layer_normalization,
            dropout=self.dropout,
        )
        self.critic_layers = create_dense_architecture(
            inpout_size,
            self.configuration_hidden_layers,
            1,
            self.activation_function,
            layer_normalization=self.layer_normalization,
            dropout=self.dropout,
        )
        print(self)
        self.to(self.device)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        action_distribution_inputs = self.actor_layers(batch[Columns.OBS].to(self.device))
        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        return self.critic_layers(batch[Columns.OBS].to(self.device)).squeeze(-1)
