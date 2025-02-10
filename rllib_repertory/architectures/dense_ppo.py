from typing import Dict, Any, Optional
import torch
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

from utilities.activation_functions import activation_functions
from utilities.create_dense_architecture import create_dense_architecture


class DensePPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function_class = self.model_config.get('activation_function_class', nn.ReLU)
        self.layer_normalization = self.model_config.get('layer_normalization', False)
        self.dropout = self.model_config.get('dropout', False)
        self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [64, 64])
        self.num_hidden_layers = len(self.configuration_hidden_layers)
        self.use_same_encoder_actor_critic = self.model_config.get('use_same_encoder_actor_critic', False)
        self.configuration_encoder_hidden_layers = self.model_config.get('configuration_encoder_hidden_layers', [64, 64])

        # Récupération du flag pour le mode création de dataset
        self.recorder_mode = self.model_config.get('recorder_mode', False)

        # Dictionnaire pour stocker les activations via hooks
        self.activations = {}

        self.inpout_size = get_preprocessor(self.observation_space)(self.observation_space).size
        self.output_size = self.action_dist_cls.required_input_dim(space=self.action_space)

        if self.use_same_encoder_actor_critic:
            self.encoder_layers = create_dense_architecture(
                self.inpout_size,
                self.configuration_encoder_hidden_layers[:-1],
                self.configuration_encoder_hidden_layers[-1],
                self.activation_function_class,
                layer_normalization=self.layer_normalization,
                dropout=self.dropout,
            )
            self.actor_layers = create_dense_architecture(
                self.configuration_encoder_hidden_layers[-1],
                self.configuration_hidden_layers,
                self.output_size,
                self.activation_function_class,
                layer_normalization=self.layer_normalization,
                dropout=self.dropout,
            )
            self.critic_layers = create_dense_architecture(
                self.configuration_encoder_hidden_layers[-1],
                self.configuration_hidden_layers,
                1,
                self.activation_function_class,
                layer_normalization=self.layer_normalization,
                dropout=self.dropout,
            )
        else:
            self.actor_layers = create_dense_architecture(
                self.inpout_size,
                self.configuration_hidden_layers,
                self.output_size,
                self.activation_function_class,
                layer_normalization=self.layer_normalization,
                dropout=self.dropout,
            )
            self.critic_layers = create_dense_architecture(
                self.inpout_size,
                self.configuration_hidden_layers,
                1,
                self.activation_function_class,
                layer_normalization=self.layer_normalization,
                dropout=self.dropout,
            )
        print(self)

        # En mode dataset, on enregistre les hooks pour capturer les activations
        if self.recorder_mode:
            self.initialisation_hooks()

        self.to(self.device)

    def initialisation_hooks(self):
        # Enregistrement des hooks
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output, name):
            # Stocker la sortie sans gradient
            self.activations[str(name) + '_' + str(module)] = output.detach()

        # Enregistrer un hook pour chaque module Linear
        for name, module in self.named_modules():
            # print(str(name) + ' : ' + str(module))
            if isinstance(module, activation_functions):
                module.register_forward_hook(lambda m, i, o, name=name: save_activation(m, i, o, name))

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)

        if self.use_same_encoder_actor_critic:
            encoder_output = self.encoder_layers(obs)
            action_distribution_inputs = self.actor_layers(encoder_output)
        else:
            action_distribution_inputs = self.actor_layers(obs)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

        if self.recorder_mode:
            self.activations['input'] = batch[Columns.OBS].to(self.device)
            self.activations['actor_output'] = action_distribution_inputs.to(self.device)

            if self.use_same_encoder_actor_critic:
                critic_output =  self.critic_layers(self.encoder_layers(batch[Columns.OBS].to(self.device))).squeeze(-1)
            else:
                critic_output = self.critic_layers(batch[Columns.OBS].to(self.device)).squeeze(-1)
            self.activations['critic_output'] = critic_output

            output['activations'] = {key: value.detach().cpu().numpy() for key, value in self.activations.items()}
            output['critic_values'] = critic_output.detach().cpu().numpy()

        return output

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        if self.use_same_encoder_actor_critic:
            return self.critic_layers(self.encoder_layers(batch[Columns.OBS].to(self.device))).squeeze(-1)
        else:
            return self.critic_layers(batch[Columns.OBS].to(self.device)).squeeze(-1)
