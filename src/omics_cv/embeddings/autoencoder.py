"""
PyTorch Autoencoder and Variational Autoencoder (VAE) for Omics representation learning (Task 4 Infrastructure).
Reduces high-dimensional transcriptomics/phosphoproteomics into dense latent embeddings.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class OmicsAutoencoder(nn.Module):
    """
    Standard / Denoising Autoencoder for omics feature representation learning.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build Encoder
        encoder_layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_d, h_d),
                nn.BatchNorm1d(h_d),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = h_d
        encoder_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder
        decoder_layers = []
        in_d = latent_dim
        for h_d in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_d, h_d),
                nn.BatchNorm1d(h_d),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = h_d
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class VariationalOmicsAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for high-dimensional biological data modeling.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder backbone
        self.encoder_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def vae_loss_function(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss
