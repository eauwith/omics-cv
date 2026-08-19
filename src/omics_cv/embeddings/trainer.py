"""
Autoencoder Trainer and Latent Feature Extractor (Task 4 Infrastructure).
"""

from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .autoencoder import OmicsAutoencoder, VariationalOmicsAutoencoder


class AutoencoderTrainer:
    """
    Trains Omics Autoencoders and extracts dense latent representations for downstream GNN or ML models.
    """

    def __init__(
        self,
        model: Union[OmicsAutoencoder, VariationalOmicsAutoencoder],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        epochs: int = 20,
        batch_size: int = 32,
        noise_factor: float = 0.0
    ) -> list:
        """
        Trains autoencoder with optional input denoising noise.
        """
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype(np.float32)
        else:
            X_data = X.astype(np.float32)

        tensor_x = torch.from_numpy(X_data)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0
            for (b_x,) in loader:
                b_x = b_x.to(self.device)
                
                # Apply noise for Denoising Autoencoder
                if noise_factor > 0.0:
                    noisy_x = b_x + noise_factor * torch.randn_like(b_x)
                else:
                    noisy_x = b_x

                self.optimizer.zero_grad()

                if isinstance(self.model, VariationalOmicsAutoencoder):
                    x_recon, mu, logvar = self.model(noisy_x)
                    loss = VariationalOmicsAutoencoder.vae_loss_function(x_recon, b_x, mu, logvar)
                else:
                    x_recon, _ = self.model(noisy_x)
                    loss = torch.nn.functional.mse_loss(x_recon, b_x)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(b_x)

            epoch_loss = total_loss / len(X_data)
            loss_history.append(epoch_loss)

        return loss_history

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Extracts dense latent embeddings for input dataset.
        """
        if isinstance(X, pd.DataFrame):
            X_data = X.values.astype(np.float32)
        else:
            X_data = X.astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_data).to(self.device)
            if isinstance(self.model, VariationalOmicsAutoencoder):
                mu, _ = self.model.encode(tensor_x)
                z = mu
            else:
                z = self.model.encode(tensor_x)
            return z.cpu().numpy()
