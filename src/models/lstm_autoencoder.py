"""
LSTM-Autoencoder for wind turbine thermal anomaly detection.

Phase 5.3: Architecture that learns to compress and reconstruct
normal-operation thermal sensor sequences. High reconstruction
error signals anomalous behavior.

The 32-dimensional bottleneck embedding is reused by the Similar
Fault Finder in Phase 7.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for multivariate time-series.

    Architecture
    ------------
    Encoder:
        LSTM(input_size → 128) → last hidden state → Linear(128 → 32) → ReLU
    Bottleneck:
        32-dimensional vector (the learned embedding)
    Decoder:
        Linear(32 → 128) → ReLU → repeat to seq_len → LSTM(128 → n_features)

    Parameters
    ----------
    n_features : int
        Number of input features (sensors) per timestep.
    hidden_size : int
        LSTM hidden dimension in encoder (default 128).
    bottleneck_size : int
        Bottleneck embedding dimension (default 32).
    seq_len : int
        Expected sequence length for decoder repeat (default 36).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        bottleneck_size: int = 32,
        seq_len: int = 36,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.seq_len = seq_len

        # --- Encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.encoder_fc = nn.Linear(hidden_size, bottleneck_size)
        self.encoder_relu = nn.ReLU()

        # --- Decoder ---
        self.decoder_fc = nn.Linear(bottleneck_size, hidden_size)
        self.decoder_relu = nn.ReLU()
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequences to bottleneck embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Bottleneck embedding, shape (batch, bottleneck_size).
        """
        # Run encoder LSTM, take the last hidden state
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: (1, batch, hidden_size)
        h_last = h_n.squeeze(0)              # (batch, hidden_size)
        embedding = self.encoder_relu(self.encoder_fc(h_last))  # (batch, 32)
        return embedding

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck embeddings back to sequences.

        Parameters
        ----------
        embedding : torch.Tensor
            Shape (batch, bottleneck_size).

        Returns
        -------
        torch.Tensor
            Reconstructed sequence, shape (batch, seq_len, n_features).
        """
        # Project bottleneck back to hidden dimension
        h = self.decoder_relu(self.decoder_fc(embedding))  # (batch, 128)

        # Repeat across time dimension to form decoder input sequence
        h_repeated = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, 128)

        # Decode with LSTM — output directly has n_features dimensions
        reconstruction, _ = self.decoder_lstm(h_repeated)  # (batch, seq_len, n_features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences, shape (batch, seq_len, n_features).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - reconstruction: shape (batch, seq_len, n_features)
            - embedding: shape (batch, bottleneck_size)
        """
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding


def train_autoencoder(
    train_data: np.ndarray,
    n_features: int,
    seq_len: int = 36,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[LSTMAutoencoder, dict]:
    """Train an LSTM autoencoder on normal-operation sequences.

    Parameters
    ----------
    train_data : np.ndarray
        Shape (n_samples, seq_len, n_features). Already normalized.
    n_features : int
        Number of features per timestep.
    seq_len : int
        Sequence length (default 36).
    epochs : int
        Maximum training epochs (default 50).
    batch_size : int
        Mini-batch size (default 256).
    lr : float
        Learning rate for Adam optimizer (default 0.001).
    patience : int
        Early stopping patience — stop if val loss doesn't improve
        for this many consecutive epochs (default 10).
    device : str
        Torch device (default 'cpu').

    Returns
    -------
    tuple[LSTMAutoencoder, dict]
        - Trained model (on CPU, eval mode).
        - Training history dict with keys:
          'train_loss', 'val_loss' (lists of per-epoch losses),
          'best_epoch' (int).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Train/val split (90/10) ---
    n_samples = len(train_data)
    n_val = max(1, int(n_samples * 0.1))
    n_train = n_samples - n_val

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.tensor(train_data[train_idx], dtype=torch.float32)
    X_val = torch.tensor(train_data[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # --- Model, loss, optimizer ---
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=128,
        bottleneck_size=32,
        seq_len=seq_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            reconstruction, _ = model(batch_x)
            loss = criterion(reconstruction, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_x)
            train_n += len(batch_x)

        train_loss = train_loss_sum / train_n

        # -- Validate --
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                reconstruction, _ = model(batch_x)
                loss = criterion(reconstruction, batch_x)
                val_loss_sum += loss.item() * len(batch_x)
                val_n += len(batch_x)

        val_loss = val_loss_sum / val_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # -- Early stopping check --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress every 5 epochs or at the end
        if epoch % 5 == 0 or epoch == 1 or epochs_no_improve >= patience:
            print(
                f"  Epoch {epoch:3d}/{epochs}  |  "
                f"Train Loss: {train_loss:.6f}  |  "
                f"Val Loss: {val_loss:.6f}  |  "
                f"Best: {best_val_loss:.6f} (ep {history['best_epoch']})"
            )

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.cpu().eval()
    return model, history


def compute_reconstruction_error(
    model: LSTMAutoencoder,
    data: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error.

    Parameters
    ----------
    model : LSTMAutoencoder
        Trained autoencoder (eval mode).
    data : np.ndarray
        Shape (n_samples, seq_len, n_features).
    batch_size : int
        Batch size for inference (default 512).
    device : str
        Torch device (default 'cpu').

    Returns
    -------
    np.ndarray
        Per-sample MSE error, shape (n_samples,).
    """
    model = model.to(device).eval()
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            reconstruction, _ = model(batch_x)
            # Per-sample MSE: mean over seq_len and n_features dimensions
            mse = ((reconstruction - batch_x) ** 2).mean(dim=(1, 2))  # (batch,)
            errors.append(mse.cpu().numpy())

    return np.concatenate(errors, axis=0)


def extract_embeddings(
    model: LSTMAutoencoder,
    data: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Extract bottleneck embeddings from the encoder.

    Parameters
    ----------
    model : LSTMAutoencoder
        Trained autoencoder (eval mode).
    data : np.ndarray
        Shape (n_samples, seq_len, n_features).
    batch_size : int
        Batch size for inference (default 512).
    device : str
        Torch device (default 'cpu').

    Returns
    -------
    np.ndarray
        Bottleneck embeddings, shape (n_samples, bottleneck_size).
    """
    model = model.to(device).eval()
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            emb = model.encode(batch_x)  # (batch, bottleneck_size)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
