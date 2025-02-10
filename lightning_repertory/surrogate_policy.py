from typing import Optional, List

import numpy as np
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
            activation_function_class=nn.LeakyReLU,
            learning_rate: float = 1e-4,
            use_clusterization_loss: bool = False,
            clusterization_function=None,
            clusterization_function_configuration: dict = {},
            clusterization_loss: Optional[nn.Module] = None,
            clusterization_loss_configuration: dict = {},
            indexes_latent_space_to_clusterize: List[int] = None,
            action_loss_coefficient: float = 1.0,
            clusterization_loss_coefficient: float = 1.0,
    ):
        super(SurrogatePolicy, self).__init__()

        if shape_layers is None:
            shape_layers = [64, 64]

        self.save_hyperparameters()

        self.model = create_dense_architecture(
            input_dimension=input_dimension,
            shape_layers=shape_layers,
            output_dimension=output_dimension,
            activation_function_class=activation_function_class,
        )
        self.embeddings_in_clustering_space = []
        self.learning_rate = learning_rate
        self.prediction_loss_function = nn.MSELoss()

        self.use_clusterization_loss = use_clusterization_loss
        if self.use_clusterization_loss:
            self.hook_count = 0
            self.clusterization_function = clusterization_function(logger=self.log, **clusterization_function_configuration)
            self.clusterization_loss = clusterization_loss(logger=self.log, **clusterization_loss_configuration)
            self.indexes_latent_space_to_clusterize = np.array(indexes_latent_space_to_clusterize)
            self._register_hooks()

        self.action_loss_coefficient = action_loss_coefficient
        self.clusterization_loss_coefficient = clusterization_loss_coefficient

    def _register_hooks(self):
        if self.indexes_latent_space_to_clusterize is not None:
            print()
            print('Surrogate policy Architecture:')
            print(self.model)

            print()
            print('Latent spaces to clusterize:')
            for index in self.indexes_latent_space_to_clusterize:
                print(str(index) + ': ' + str(self.model[index]))
            print()

            children = list(self.model.children())
            for child in children:
                child.register_forward_hook(self._hook_fn)

    def forward(self, x):
        return self.model(x)

    def _hook_fn(self, module, input, output):
        if self.hook_count in self.indexes_latent_space_to_clusterize:
            self.embeddings_in_clustering_space.append(output)
        self.hook_count += 1

    def get_embeddings_in_clustering_space(self):
        clustered_space_activations = torch.cat(self.embeddings_in_clustering_space, dim=1)
        self.hook_count = 0
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

        total_loss = self.action_loss_coefficient * action_loss + self.clusterization_loss_coefficient * clustering_loss

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

        total_loss = self.action_loss_coefficient * action_loss + self.clusterization_loss_coefficient * clustering_loss

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
