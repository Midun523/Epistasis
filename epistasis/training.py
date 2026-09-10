"""High-performance training, validation, and interpretation runner for EpistasisTransformer."""

from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from epistasis.models.partitioned_transformer import EpistasisTransformer
from epistasis.interpretability.gradient_cam import compute_attentive_class_activation_tokens
from epistasis.evaluation.metrics import evaluate_epistasis_detection, compute_prediction_metrics
from epistasis.utils.logger import get_logger

logger = get_logger("training")


def train_epistasis_transformer(
    model: EpistasisTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    warmup_epochs: int = 5,
    ema_decay: float = 0.7,
    device: Optional[str] = None,
    task: str = "classification",
) -> Tuple[EpistasisTransformer, Dict[str, List[float]]]:
    """
    Trains EpistasisTransformer with EMA-smoothed validation metric tracking,
    warmup guard, early stopping, and learning rate scheduling.

    Returns:
        (best_model, history_dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],
        "val_metric_smoothed": [],
    }

    best_metric = float("-inf")
    best_state = None
    no_improve_count = 0
    smoothed_metric = None

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x).squeeze(-1)
            loss = criterion(out, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # Validation Phase
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x).squeeze(-1)
                loss = criterion(out, batch_y)
                val_losses.append(loss.item())
                val_preds.append(out.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        avg_val_loss = float(np.mean(val_losses))
        val_preds_arr = np.concatenate(val_preds, axis=0)
        val_targets_arr = np.concatenate(val_targets, axis=0)

        val_metrics = compute_prediction_metrics(val_targets_arr, val_preds_arr, task=task)
        primary_val_metric = val_metrics["roc_auc"] if task == "classification" else val_metrics["r2"]

        # Exponential Moving Average (EMA) smoothing to guard against single-epoch AUC noise on small splits
        if smoothed_metric is None:
            smoothed_metric = primary_val_metric
        else:
            smoothed_metric = ema_decay * smoothed_metric + (1.0 - ema_decay) * primary_val_metric

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_metric"].append(primary_val_metric)
        history["val_metric_smoothed"].append(smoothed_metric)

        # Checkpoint selection and early stopping with warmup guard
        effective_warmup = min(warmup_epochs, max(1, epochs // 4))
        if epoch >= effective_warmup:
            if smoothed_metric > best_metric:
                best_metric = smoothed_metric
                best_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # During warmup, track latest state as baseline fallback
            best_state = copy.deepcopy(model.state_dict())
            best_metric = smoothed_metric

        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (Best Smoothed Val {'AUC' if task == 'classification' else 'R2'}: {best_metric:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def run_full_pipeline(
    data_dict: Dict,
    num_partitions: int = 6,
    combination_size: int = 2,
    sparsity_ratio: float = 0.90,
    epochs: int = 20,
    batch_size: int = 64,
    device: Optional[str] = None,
    alpha_acat: float = 0.5,
    aggregation: str = "gated",
    warmup_epochs: int = 5,
    ema_decay: float = 0.7,
    random_state: int = 42,
) -> Dict:
    """
    Runs end-to-end data splitting, training, interpretation, and detection evaluation for a dataset.
    """
    from epistasis.data.dataset import create_dataloaders

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_task = data_dict.get("metadata", {}).get("phenotype_type", "")
    y_arr = data_dict["y"]
    if meta_task in ("regression", "continuous") or np.issubdtype(y_arr.dtype, np.floating) or len(np.unique(y_arr)) > 2:
        task = "regression"
    else:
        task = "classification"
    n_snps = data_dict["X"].shape[1]
    causal_indices = data_dict["causal_indices"]

    train_loader, val_loader, test_loader, splits = create_dataloaders(
        data_dict, batch_size=batch_size, random_state=random_state
    )

    model = EpistasisTransformer(
        num_snps=n_snps,
        embed_dim=64,
        num_heads=4,
        num_partitions=num_partitions,
        combination_size=combination_size,
        sparsity_ratio=sparsity_ratio,
        aggregation=aggregation,
        num_layers=2,
        ff_dim=128,
        task=task,
    ).to(device)

    trained_model, history = train_epistasis_transformer(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        ema_decay=ema_decay,
        device=device,
        task=task,
    )

    # NOTE: ACAT scoring and detection-power evaluation must run on held-out
    # test data only, never on data_dict["X"] (the full dataset). Using the
    # full dataset silently includes the ~70% of samples the model was
    # trained on, which can make a memorizing/overfit model look like it has
    # correctly "detected" the causal SNPs even when validation AUC shows it
    # hasn't generalized at all -- confirmed empirically: a P=15 config with
    # best_val_auc=0.485 (worse than random) previously reported near-perfect
    # causal-SNP ranking, which becomes uninterpretable once you know ranking
    # was computed on data including the training set itself.
    X_test, y_test = splits["test"]

    # Compute Attentive Class Activation Tokens (ACAT) on held-out test data only
    combined_scores, attn_scores, grad_scores = compute_attentive_class_activation_tokens(
        trained_model, X_test, alpha=alpha_acat, batch_size=batch_size, device=device
    )

    # Evaluate detection power and ranking metrics
    eval_results = evaluate_epistasis_detection(
        scores=combined_scores,
        causal_indices=causal_indices,
        top_percent=0.05,
    )

    return {
        "model": trained_model,
        "history": history,
        "combined_scores": combined_scores,
        "attn_scores": attn_scores,
        "grad_scores": grad_scores,
        "detection_eval": eval_results,
    }
