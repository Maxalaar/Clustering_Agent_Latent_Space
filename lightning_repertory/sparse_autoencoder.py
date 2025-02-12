import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.create_dense_architecture import create_dense_architecture


# def entropy_loss(x):
#     # Applique une sigmoïde pour obtenir des valeurs entre 0 et 1
#     p = torch.sigmoid(x)
#     # Calcul de l'entropie (évite les valeurs nulles pour ne pas avoir de log(0))
#     entropy = -torch.sum(p * torch.log(p + 1e-8))
#     return entropy
#
# def absolut_loss(x):
#     return torch.sum(torch.abs(x), dim=1).mean()

def cross_entropy_loss(x):
    # p = torch.softmax(x, dim=1)

    # Calcul de l'entropie pour chaque élément du batch
    entropy = -torch.sum(x * torch.log(x), dim=1)

    # Calcul de la perte moyenne sur tout le batch
    return entropy.mean()


class SparseAutoencoder(pl.LightningModule):
    def __init__(self, input_dimension, latent_dimension, sparsity_loss_coefficient=0.01):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dimension, latent_dimension, bias=True),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_dimension, latent_dimension, bias=True),
        # )

        self.encoder = create_dense_architecture(
            input_dimension=input_dimension,
            shape_layers=[32, 32],
            output_dimension=latent_dimension,
            activation_function_class=nn.LeakyReLU,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, input_dimension, bias=True),
            # nn.Sigmoid()  # Adaptez l'activation selon vos données
        )

        self.sparsity_coeff = sparsity_loss_coefficient

    def forward(self, x):
        # Aplatit l'input si nécessaire (pour les images)
        # x = x.view(x.size(0), -1)

        # latent = self.encoder(x)
        latent = torch.softmax(self.encoder(x), dim=1)

        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def training_step(self, batch, batch_idx):
        x, = batch
        latent, recon = self(x)

        # Loss de reconstruction
        recon_loss = F.mse_loss(recon, x)

        # Contrainte de sparsité (L1 sur les activations latentes)
        # sparsity_loss = latent.abs().mean()
        # sparsity_loss = entropy_loss(latent)
        # sparsity_loss = absolut_loss(latent)
        sparsity_loss = cross_entropy_loss(latent)

        # Loss totale
        total_loss = recon_loss + 0.01 * sparsity_loss

        # Logging
        self.log('total_loss_train', total_loss, on_epoch=True)
        self.log('mean_non_zero_train', (latent != 0).sum(dim=1).float().mean(), on_epoch=True)
        self.log('reconstruction_loss_train', recon_loss, on_epoch=True)
        self.log('sparsity_loss_train', sparsity_loss, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

        # x, = batch
        # latent, recon = self(x)
        #
        # recon_loss = F.mse_loss(recon, x)
        # sparsity_loss = cross_entropy_loss(latent)
        #
        # # On log seulement la reconstruction pour la validation
        # self.log('mean_non_zero_validation', (latent != 0).sum(dim=1).float().mean(), on_epoch=True)
        # self.log('reconstruction_loss_validation', recon_loss, on_epoch=True)
        # self.log('sparsity_loss_validation', sparsity_loss, on_epoch=True)
        #
        # return recon_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
