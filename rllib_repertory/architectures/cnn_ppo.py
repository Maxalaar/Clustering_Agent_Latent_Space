from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns

from utilities.create_cnn_architecture import create_cnn_architecture
from utilities.create_dense_architecture import create_dense_architecture

class CNNPPO(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_function_class = self.model_config.get("activation_function_class", nn.ReLU)
        self.layer_normalization = self.model_config.get("layer_normalization", False)
        self.dropout = self.model_config.get("dropout", False)
        self.configuration_hidden_layers = self.model_config.get("configuration_hidden_layers", [64, 64])
        self.configuration_cnn = self.model_config.get(
            "configuration_cnn",
            [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        )
        self.use_layer_normalization_cnn = self.model_config.get("use_layer_normalization_cnn", False)
        self.use_unified_cnn = self.model_config.get("use_unified_cnn", False)  # Option uu

        # Determine input channels for CNN
        in_channels = self.observation_space.shape[0] if len(self.observation_space.shape) == 3 else 1

        # Create CNN architecture
        self.shared_cnn = create_cnn_architecture(
            in_channels=in_channels,
            configuration_cnn=self.configuration_cnn,
            activation_function_class=self.activation_function_class,
            use_normalization=self.use_layer_normalization_cnn,
        )

        # Use shared CNN for both actor and critic if `use_unified_cnn` is True
        if self.use_unified_cnn:
            self.actor_cnn = self.shared_cnn
            self.critic_cnn = self.shared_cnn
        else:
            self.actor_cnn = create_cnn_architecture(
                in_channels=in_channels,
                configuration_cnn=self.configuration_cnn,
                activation_function_class=self.activation_function_class,
                use_normalization=self.use_layer_normalization_cnn,
            )
            self.critic_cnn = create_cnn_architecture(
                in_channels=in_channels,
                configuration_cnn=self.configuration_cnn,
                activation_function_class=self.activation_function_class,
                use_normalization=self.use_layer_normalization_cnn,
            )

        # Dynamically compute CNN output size
        dummy_input = torch.zeros(32, *self.observation_space.shape)
        cnn_output_size = self.actor_cnn(dummy_input).shape[1]
        print(f'CNN output size: {cnn_output_size}')

        self.output_size = self.action_dist_cls.required_input_dim(space=self.action_space)

        # Create dense layers for actor and critic
        self.actor_layers = create_dense_architecture(
            cnn_output_size,
            self.configuration_hidden_layers,
            self.output_size,
            self.activation_function_class,
            layer_normalization=self.layer_normalization,
            dropout=self.dropout,
        )
        self.critic_layers = create_dense_architecture(
            cnn_output_size,
            self.configuration_hidden_layers,
            1,
            self.activation_function_class,
            layer_normalization=self.layer_normalization,
            dropout=self.dropout,
        )

        self.to(self.device)
        print(self)

    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)
        cnn_features = self.actor_cnn(obs)
        action_distribution_inputs = self.actor_layers(cnn_features)
        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> Tensor:
        obs = batch[Columns.OBS].to(self.device)
        cnn_features = self.critic_cnn(obs)
        return self.critic_layers(cnn_features).squeeze(-1)
