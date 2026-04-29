"""
train.py
--------
Training loop and evaluation utilities for the Vanilla RNN volatility model.

Public API
----------
    train_one_epoch(model, loader, optimizer, device) -> float
    evaluate(model, loader, device)                   -> dict
    run_training(...)                                 -> dict
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =========================================================================
# 1. SINGLE EPOCH
# =========================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training DataLoader.

    Returns
    -------
    avg_mse : float  – mean MSE loss across all batches
    """
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for x, stock_id, y in loader:
        x        = x.to(device)         # (B, lookback, 1)
        stock_id = stock_id.to(device)  # (B, num_stocks)
        y        = y.to(device)         # (B,)

        optimizer.zero_grad()
        preds = model(x, stock_id)      # (B,)
        loss  = criterion(preds, y)
        loss.backward()

        # Gradient clipping — vanilla RNNs can explode without it
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================================
# 2. EVALUATION
# =========================================================================
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on a DataLoader (val or test set).

    Returns a dict with:
        mse         – MSE in log-vol space
        mae         – MAE in original vol space (after exp())
        correlation – Pearson r between pred and true log-vol
        preds_log   – numpy array of log-vol predictions
        targets_log – numpy array of log-vol targets
    """
    model.eval()
    criterion = nn.MSELoss()

    all_preds   = []
    all_targets = []
    total_mse   = 0.0

    with torch.no_grad():
        for x, stock_id, y in loader:
            x        = x.to(device)
            stock_id = stock_id.to(device)
            y        = y.to(device)

            preds = model(x, stock_id)
            total_mse += criterion(preds, y).item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_log   = np.concatenate(all_preds)
    targets_log = np.concatenate(all_targets)

    # MAE in original volatility space
    mae = np.mean(np.abs(np.exp(preds_log) - np.exp(targets_log)))

    # Pearson correlation in log-vol space
    correlation = float(np.corrcoef(preds_log, targets_log)[0, 1])

    return {
        "mse":         total_mse / len(loader),
        "mae":         mae,
        "correlation": correlation,
        "preds_log":   preds_log,
        "targets_log": targets_log,
    }


# =========================================================================
# 3. FULL TRAINING RUN
# =========================================================================
def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,                       # e.g. ReduceLROnPlateau; pass None to skip
    device: torch.device,
    num_epochs: int = 20,
    patience: int = 5,
    checkpoint_path: str = "best_model.pt",
    verbose: bool = True,
) -> dict:
    """
    Full training run with early stopping and model checkpointing.

    Parameters
    ----------
    model            : nn.Module
    train_loader     : DataLoader   training split
    val_loader       : DataLoader   validation split
    optimizer        : torch.optim  e.g. Adam
    scheduler        : LR scheduler or None
    device           : torch.device
    num_epochs       : int          max epochs to train
    patience         : int          early-stopping patience (epochs)
    checkpoint_path  : str          path to save the best model weights
    verbose          : bool         print per-epoch metrics

    Returns
    -------
    history : dict with keys
        train_mse, val_mse, val_mae, val_corr  (lists, one value per epoch)
    """
    history = {
        "train_mse": [],
        "val_mse":   [],
        "val_mae":   [],
        "val_corr":  [],
    }

    best_val_mse    = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        train_mse = train_one_epoch(model, train_loader, optimizer, device)

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, device)
        val_mse     = val_metrics["mse"]
        val_mae     = val_metrics["mae"]
        val_corr    = val_metrics["correlation"]

        # --- LR Scheduler ---
        if scheduler is not None:
            # ReduceLROnPlateau expects the metric; StepLR takes no arg
            try:
                scheduler.step(val_mse)
            except TypeError:
                scheduler.step()

        # --- Logging ---
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)
        history["val_corr"].append(val_corr)

        if verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train MSE: {train_mse:.5f} | "
                f"Val MSE: {val_mse:.5f} | "
                f"Val MAE: {val_mae:.5f} | "
                f"Val Corr: {val_corr:.4f} | "
                f"LR: {lr:.2e}"
            )

        # --- Checkpointing ---
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"  ✓ New best val MSE {best_val_mse:.5f} — model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping triggered after {epoch} epochs.")
                break

    # Reload best weights before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if verbose:
        print(f"\nTraining complete. Best Val MSE: {best_val_mse:.5f}")

    return history
