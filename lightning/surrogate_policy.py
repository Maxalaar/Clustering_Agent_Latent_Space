import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

from utilities.create_dense_architecture import create_dense_architecture


class SurrogatePolicy(pl.LightningModule):
    def __init__(
            self,
            input_dimension,
            output_dimension,
            cluster_space_size,
            projection_clustering_space_shape,
            projection_action_space_shape,
            learning_rate,
            clusterization_loss_function=None,
            clusterization_loss_function_arguments=None,
    ):
        super(SurrogatePolicy, self).__init__()

        self.save_hyperparameters()

        self.prediction_loss_function = nn.MSELoss()
        self.clusterization_loss_function = clusterization_loss_function
        self.clusterization_loss_function_arguments = clusterization_loss_function_arguments
        self.activation_function = nn.LeakyReLU()
        self.learning_rate = learning_rate
        self.cluster_space_size = cluster_space_size
        self.embeddings_in_clustering_space = None

        self.projection_clustering_space = create_dense_architecture(
            input_dimension,
            projection_clustering_space_shape,
            self.cluster_space_size,
            self.activation_function,
        )

        self.projection_action_space = create_dense_architecture(
            self.cluster_space_size,
            projection_action_space_shape,
            output_dimension,
            self.activation_function,
        )

    def forward(self, x):
        # x = torch.squeeze(x, dim=0)
        self.embeddings_in_clustering_space = self.projection_clustering_space(x)
        action = self.projection_action_space(self.embeddings_in_clustering_space)
        return action

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = torch.squeeze(x, dim=0)
        # y = torch.squeeze(y, dim=0)

        y_hat = self(x)
        y = y.to(y_hat.device)
        action_loss = self.prediction_loss_function(y_hat, y)

        if self.clusterization_loss_function is None:
            clustering_loss = 0
        else:
            clustering_loss = self.clusterization_loss_function(
                embeddings=self.embeddings_in_clustering_space,
                logger=self.log,
                **self.clusterization_loss_function_arguments,
            )

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

        if self.clusterization_loss_function is None:
            clustering_loss = 0
        else:
            clustering_loss = self.clusterization_loss_function(self.embeddings_in_clustering_space, **self.clusterization_loss_function_arguments)

        total_loss = action_loss + clustering_loss

        self.log('action_loss_validation', action_loss, on_epoch=True)
        self.log('clusterization_loss_validation', clustering_loss, on_epoch=True)
        self.log('total_loss_validation', total_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
