from typing import Dict, Any, Optional

from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

# from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
# from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
# from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


class Dense(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.activation_function = self.model_config.get('activation_function', nn.ReLU())
        self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [64, 64])
        self.num_hidden_layers = len(self.configuration_hidden_layers)

        inpout_size = get_preprocessor(self.observation_space)(self.observation_space).size
        output_size = self.action_dist_cls.required_input_dim(space=self.action_space)

        actor_layers = [nn.Linear(inpout_size, self.configuration_hidden_layers[0]), self.activation_function]
        critic_layers = [nn.Linear(inpout_size, self.configuration_hidden_layers[0]), self.activation_function]

        for i in range(0, self.num_hidden_layers - 1):
            actor_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i + 1]))
            actor_layers.append(self.activation_function)

            critic_layers.append(
                nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i + 1]))
            critic_layers.append(self.activation_function)

        actor_layers.append(nn.Linear(self.configuration_hidden_layers[-1], output_size))
        critic_layers.append(nn.Linear(self.configuration_hidden_layers[-1], 1))

        self.actor_layers = nn.Sequential(*actor_layers)
        self.critic_layers = nn.Sequential(*critic_layers)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        action_distribution_inputs = self.actor_layers(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        return self.critic_layers(batch[Columns.OBS]).squeeze(-1)
