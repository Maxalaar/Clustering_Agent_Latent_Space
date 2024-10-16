from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class Transformer(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_configuration, name, **kwargs):
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs, model_configuration, name)
        nn.Module.__init__(self)

        # Paramètres pour le bloc Transformer
        d_model = kwargs.get('d_model', 16)
        nhead = kwargs.get('nhead', 4)
        dim_feedforward = kwargs.get('dim_feedforward', 512)
        num_transformer_layers = kwargs.get('num_transformer_layers', 2)
        dropout = kwargs.get('dropout', 0.1)

        # Crée le préprocesseur d'observation
        observation_size = get_preprocessor(observation_space)(observation_space).size

        # Encodeur linéaire par composante (chaque composante devient un token)
        self.input_layer = nn.Linear(1, d_model)

        # Bloc Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Couches finales
        self.action_layer = nn.Linear(d_model, num_outputs)
        self.critic_layer = nn.Linear(d_model, 1)

    def forward(self, input_dict, state, seq_lens):
        # Séparer chaque composante comme un token et les passer par l'encodeur linéaire
        obs = input_dict['obs_flat'].unsqueeze(-1)  # Ajouter une dimension pour chaque composante
        x = self.input_layer(obs)

        # Transformer Encoder avec les tokens
        self.transformer_output = self.transformer_encoder(x)

        # Calcul de l'action à partir du dernier token (ou moyenne des tokens)
        action = self.action_layer(x.mean(dim=1))  # Moyenne sur la dimension des tokens pour obtenir une seule sortie
        return action, []

    def value_function(self):
        # Calcul de la valeur en utilisant le dernier token (ou moyenne des tokens)
        value = self.critic_layer(self.transformer_output.mean(dim=1))
        return torch.reshape(value, [-1])
