from typing import Optional

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
            architecture_configuration: Optional[dict] = None,
            learning_rate: float = 1e-4,
            clusterization_loss=None,
            clusterization_loss_configuration: Optional[dict] = None,
    ):
        super(SurrogatePolicy, self).__init__()

        self.save_hyperparameters()

        self.prediction_loss_function = nn.MSELoss()
        self.clusterization_loss = clusterization_loss(logger=self.log, **clusterization_loss_configuration)
        self.activation_function = nn.LeakyReLU()
        self.learning_rate = learning_rate
        self.cluster_space_size = architecture_configuration.get('cluster_space_size', 16)
        self.embeddings_in_clustering_space = None

        self.projection_clustering_space = create_dense_architecture(
            input_dimension,
            architecture_configuration.get('projection_clustering_space_shape', [128, 64, 32]),
            self.cluster_space_size,
            self.activation_function,
        )

        self.projection_action_space = create_dense_architecture(
            self.cluster_space_size,
            architecture_configuration.get('projection_action_space_shape', [32, 64, 128]),
            output_dimension,
            self.activation_function,
        )

    def forward(self, x, use_noise: bool = False):
        self.embeddings_in_clustering_space = self.projection_clustering_space(x)

        # if use_noise:
        #     gaussian_noise = torch.randn_like(self.embeddings_in_clustering_space) * 0.1
        #     self.embeddings_in_clustering_space = gaussian_noise + self.embeddings_in_clustering_space

        action = self.projection_action_space(self.embeddings_in_clustering_space)
        return action

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x, use_noise=True)
        y = y.to(y_hat.device)
        action_loss = self.prediction_loss_function(y_hat, y)

        if self.clusterization_loss is None:
            clustering_loss = 0
        else:
            clustering_loss = self.clusterization_loss(
                embeddings=self.embeddings_in_clustering_space,
                current_global_step=self.global_step,
            )

        total_loss = action_loss + self.clusterization_loss_coefficient * clustering_loss

        self.log('action_loss_train', action_loss, on_epoch=True)
        self.log('clusterization_loss_train', clustering_loss, on_epoch=True)
        self.log('total_loss_train', total_loss, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        y = y.to(y_hat.device)
        action_loss = self.prediction_loss_function(y_hat, y)

        if self.clusterization_loss is None:
            clustering_loss = 0
        else:
            clustering_loss = self.clusterization_loss(
                embeddings=self.embeddings_in_clustering_space,
                current_global_step=self.global_step,
            )

        total_loss = action_loss + clustering_loss

        self.log('action_loss_validation', action_loss, on_epoch=True)
        self.log('clusterization_loss_validation', clustering_loss, on_epoch=True)
        self.log('total_loss_validation', total_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
