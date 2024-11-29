from typing import Optional, List

import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam

from utilities.create_dense_architecture import create_dense_architecture


class SurrogatePolicy(pl.LightningModule):
    def __init__(
            self,
            input_dimension,
            output_dimension,
            shape_layers=None,
            activation_function=nn.LeakyReLU(),
            learning_rate: float = 1e-4,
            use_clusterization_loss: bool = False,
            clusterization_function=None,
            clusterization_function_configuration: dict = {},
            clusterization_loss: Optional[nn.Module] = None,
            clusterization_loss_configuration: dict = {},
            latent_space_to_clusterize: List[bool] = None,
    ):
        super(SurrogatePolicy, self).__init__()

        if shape_layers is None:
            shape_layers = [64, 64]

        self.save_hyperparameters()

        self.model = create_dense_architecture(
            input_dimension=input_dimension,
            shape_layers=shape_layers,
            output_dimension=output_dimension,
            activation_function=activation_function,
        )
        self.embeddings_in_clustering_space = []
        self.learning_rate = learning_rate
        self.prediction_loss_function = nn.MSELoss()

        self.use_clusterization_loss = use_clusterization_loss
        if self.use_clusterization_loss:
            self.clusterization_function = clusterization_function(logger=self.log, **clusterization_function_configuration)
            self.clusterization_loss = clusterization_loss(logger=self.log, **clusterization_loss_configuration)
            self.latent_spaces_to_clusterize = latent_space_to_clusterize
            self._register_hooks()

    def _register_hooks(self):
        if self.latent_spaces_to_clusterize is not None:
            for layer, track in zip(self.model.children(), self.latent_spaces_to_clusterize):
                if track:
                    layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.embeddings_in_clustering_space.append(output)

    def forward(self, x):
        return self.model(x)

    def get_embeddings_in_clustering_space(self):
        clustered_space_activations = torch.cat(self.embeddings_in_clustering_space, dim=1)
        self.embeddings_in_clustering_space = []
        return clustered_space_activations

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        y = y.to(y_hat.device)
        action_loss = self.prediction_loss_function(y_hat, y)

        if self.use_clusterization_loss:
            embeddings_in_clustering_space = self.get_embeddings_in_clustering_space()
            cluster_result = self.clusterization_function(embeddings_in_clustering_space)
            clustering_loss = self.clusterization_loss(
                embeddings=embeddings_in_clustering_space,
                **cluster_result,
            )
        else:
            clustering_loss = 0.0

        total_loss = action_loss + clustering_loss

        self.log('action_loss_train', action_loss, on_epoch=True)
        self.log('clusterization_loss_train', clustering_loss, on_epoch=True)
        self.log('total_loss_train', total_loss, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        y = y.to(y_hat.device)
        action_loss = self.prediction_loss_function(y_hat, y)

        if self.use_clusterization_loss:
            embeddings_in_clustering_space = self.get_embeddings_in_clustering_space()
            cluster_result = self.clusterization_function(embeddings_in_clustering_space)
            clustering_loss = self.clusterization_loss(
                embeddings=embeddings_in_clustering_space,
                **cluster_result,
            )
        else:
            clustering_loss = 0.0

        total_loss = action_loss + clustering_loss

        self.log('action_loss_validation', action_loss, on_epoch=True)
        self.log('clusterization_loss_validation', clustering_loss, on_epoch=True)
        self.log('total_loss_validation', total_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def projection_clustering_space(self, x):
        if self.use_clusterization_loss:
            self(x)
            return self.get_embeddings_in_clustering_space()
        else:
            raise RuntimeError(
                'Projection into the clustering space requires \'use_clusterization_loss\' to be enabled. Make sure this option is correctly set before calling this method.'
            )
