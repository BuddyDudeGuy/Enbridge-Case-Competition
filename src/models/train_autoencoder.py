"""
Train LSTM-Autoencoders for all three wind farms.

Phase 5.4: Loads pre-built normal-operation sequences, trains one
autoencoder per farm, saves model weights + config + training curves.

Usage:
    py src/models/train_autoencoder.py
"""

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add project root to path for imports
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_autoencoder import (
    LSTMAutoencoder,
    train_autoencoder,
    compute_reconstruction_error,
)

# Farm configurations
FARMS = {
    "farm_a": {"file": "farm_a_train_X.npy", "n_features": 19},
    "farm_b": {"file": "farm_b_train_X.npy", "n_features": 22},
    "farm_c": {"file": "farm_c_train_X.npy", "n_features": 20},
}

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 256
LR = 0.001
PATIENCE = 10
HIDDEN_SIZE = 128
BOTTLENECK_SIZE = 32
DEVICE = "cpu"


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    seq_dir = PROJECT_ROOT / "data" / "processed" / "ae_sequences"
    models_dir = PROJECT_ROOT / "data" / "processed" / "models"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    # Ensure output directories exist
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Collect results for the training report
    report = {}
    histories = {}

    for farm_key, cfg in FARMS.items():
        farm_letter = farm_key.split("_")[1].upper()
        print(f"\n{'='*60}")
        print(f"  Training LSTM-Autoencoder for Farm {farm_letter}")
        print(f"{'='*60}")

        # --- Load data ---
        data_path = seq_dir / cfg["file"]
        X = np.load(data_path)
        n_features = cfg["n_features"]
        seq_len = X.shape[1]

        print(f"  Data shape: {X.shape}")
        print(f"  Features: {n_features}  |  Seq length: {seq_len}")
        print(f"  Sequences: {X.shape[0]:,}")

        assert X.shape[2] == n_features, (
            f"Expected {n_features} features, got {X.shape[2]}"
        )

        # --- Train ---
        t0 = time.time()
        model, history = train_autoencoder(
            train_data=X,
            n_features=n_features,
            seq_len=seq_len,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            patience=PATIENCE,
            device=DEVICE,
        )
        training_time = time.time() - t0

        best_epoch = history["best_epoch"]
        final_train = history["train_loss"][-1]
        final_val = history["val_loss"][-1]
        best_train = history["train_loss"][best_epoch - 1]
        best_val = history["val_loss"][best_epoch - 1]

        print(f"\n  Training complete in {training_time:.1f}s")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best train loss: {best_train:.6f}")
        print(f"  Best val loss:   {best_val:.6f}")
        print(f"  Final train loss: {final_train:.6f}")
        print(f"  Final val loss:   {final_val:.6f}")

        # --- Save model weights + config ---
        farm_model_dir = models_dir / farm_key
        farm_model_dir.mkdir(parents=True, exist_ok=True)

        # Save state dict
        weights_path = farm_model_dir / "lstm_ae.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"  Model saved to {weights_path}")

        # Save config so we can reconstruct the model later
        config = {
            "n_features": n_features,
            "hidden_size": HIDDEN_SIZE,
            "bottleneck_size": BOTTLENECK_SIZE,
            "seq_len": seq_len,
        }
        config_path = farm_model_dir / "lstm_ae_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Config saved to {config_path}")

        # --- Quick reconstruction error stats on training data ---
        recon_errors = compute_reconstruction_error(model, X, device=DEVICE)
        print(f"  Recon error — mean: {recon_errors.mean():.6f}, "
              f"std: {recon_errors.std():.6f}, "
              f"p95: {np.percentile(recon_errors, 95):.6f}, "
              f"max: {recon_errors.max():.6f}")

        # Store for report
        report[farm_key] = {
            "n_features": n_features,
            "seq_len": seq_len,
            "n_sequences": int(X.shape[0]),
            "epochs_trained": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "best_train_loss": float(best_train),
            "best_val_loss": float(best_val),
            "final_train_loss": float(final_train),
            "final_val_loss": float(final_val),
            "training_time_seconds": round(training_time, 1),
            "recon_error_mean": float(recon_errors.mean()),
            "recon_error_std": float(recon_errors.std()),
            "recon_error_p95": float(np.percentile(recon_errors, 95)),
            "recon_error_max": float(recon_errors.max()),
        }
        histories[farm_key] = history

    # --- Save training report ---
    report_path = reports_dir / "ae_training_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTraining report saved to {report_path}")

    # --- Plot training curves (1x3 subplot) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (farm_key, history) in enumerate(histories.items()):
        ax = axes[idx]
        farm_letter = farm_key.split("_")[1].upper()
        epochs_range = range(1, len(history["train_loss"]) + 1)

        ax.plot(epochs_range, history["train_loss"], label="Train", linewidth=1.5)
        ax.plot(epochs_range, history["val_loss"], label="Validation", linewidth=1.5)
        ax.axvline(
            x=history["best_epoch"],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Best (ep {history['best_epoch']})",
        )

        ax.set_title(f"Farm {farm_letter}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate best val loss
        best_ep = history["best_epoch"]
        best_val = history["val_loss"][best_ep - 1]
        ax.annotate(
            f"{best_val:.5f}",
            xy=(best_ep, best_val),
            xytext=(best_ep + 2, best_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="gray",
        )

    fig.suptitle("LSTM-Autoencoder Training Loss Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    curves_path = figures_dir / "ae_training_curves.png"
    fig.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {curves_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    for farm_key, r in report.items():
        farm_letter = farm_key.split("_")[1].upper()
        print(
            f"  Farm {farm_letter}: {r['n_sequences']:,} seqs × {r['n_features']} features  |  "
            f"Best val loss: {r['best_val_loss']:.6f} (ep {r['best_epoch']})  |  "
            f"Time: {r['training_time_seconds']}s"
        )
    print()


if __name__ == "__main__":
    main()
