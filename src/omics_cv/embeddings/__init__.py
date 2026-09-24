"""Latent-representation learning for omics feature matrices."""
from .autoencoder import OmicsAutoencoder, VariationalOmicsAutoencoder
from .trainer import AutoencoderTrainer

__all__ = ["OmicsAutoencoder", "VariationalOmicsAutoencoder", "AutoencoderTrainer"]
