"""Epoch-by-epoch verbose training logger for P=6 and P=10 on 1000 SNPs."""

import time
import torch
import torch.nn as nn
import numpy as np
from epistasis.data.gametes_simulator import EpistasisSimulator
from epistasis.data.dataset import create_dataloaders
from epistasis.models.partitioned_transformer import EpistasisTransformer
from epistasis.evaluation.metrics import compute_prediction_metrics, evaluate_epistasis_detection
from epistasis.interpretability.gradient_cam import compute_attentive_class_activation_tokens

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_and_log_epochs(p: int, seed: int = 42, epochs: int = 35):
    # Set seeds for absolute reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    print(f"\n================================================================================")
    print(f"=== EPOCH-BY-EPOCH LOG: P={p} Partitions (C(P,2)={p*(p-1)//2} combos) on 1000 SNPs ===")
    print(f"================================================================================")
    
    sim = EpistasisSimulator(n_snps=1000, n_samples=1600, order=2, maf=0.2, heritability=0.4, model_type="additive", random_state=seed)
    data = sim.generate_case_control()
    causal = data["causal_indices"]
    print(f"Dataset: 1000 SNPs, 1600 samples, Causal Indices: {causal}, Device: {device}")
    
    train_loader, val_loader, test_loader, splits = create_dataloaders(data, batch_size=64, random_state=seed)
    X_test, y_test = splits["test"]
    
    model = EpistasisTransformer(
        num_snps=1000,
        embed_dim=64,
        num_heads=4,
        num_partitions=p,
        combination_size=2,
        sparsity_ratio=0.85,
        aggregation="gated",
        num_layers=2,
        ff_dim=128,
        task="classification"
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    best_metric = float("-inf")
    best_state = None
    no_improve = 0
    patience = 15
    
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val AUC':<10} | {'LR':<10} | {'Test Mean Rank':<14} | {'Test Causal Ranks':<18} | {'Status'}")
    print("-" * 105)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            
        avg_train_loss = float(np.mean(train_losses))
        
        # Validation
        model.eval()
        val_losses, val_preds, val_targets = [], [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx).squeeze(-1)
                loss = criterion(out, by)
                val_losses.append(loss.item())
                val_preds.append(out.cpu().numpy())
                val_targets.append(by.cpu().numpy())
                
        avg_val_loss = float(np.mean(val_losses))
        val_preds_arr = np.concatenate(val_preds)
        val_targets_arr = np.concatenate(val_targets)
        val_metrics = compute_prediction_metrics(val_targets_arr, val_preds_arr, task="classification")
        val_auc = val_metrics["roc_auc"]
        
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        
        # Quick ACAT extraction on held-out test data
        combined_scores, _, _ = compute_attentive_class_activation_tokens(model, X_test, alpha=0.5, batch_size=64, device=device)
        eval_res = evaluate_epistasis_detection(combined_scores, causal, top_percent=0.05)
        
        is_best = val_auc > best_metric
        if is_best:
            best_metric = val_auc
            import copy
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            status_str = "--> Best Checkpoint"
        else:
            no_improve += 1
            status_str = f"No improve ({no_improve}/{patience})"
            
        print(f"{epoch:02d}/{epochs:<2} | {avg_train_loss:<10.4f} | {avg_val_loss:<10.4f} | {val_auc:<10.4f} | {curr_lr:<10.1e} | {eval_res['mean_causal_rank']:<14.1f} | {str(eval_res['causal_ranks']):<18} | {status_str}")
        
        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (Best Val AUC: {best_metric:.4f})")
            break
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    final_scores, _, _ = compute_attentive_class_activation_tokens(model, X_test, alpha=0.5, batch_size=64, device=device)
    final_eval = evaluate_epistasis_detection(final_scores, causal, top_percent=0.05)
    print(f"\nFinal Restored Best Checkpoint Evaluation (Held-out Test):")
    print(f"  Best Val AUC: {best_metric:.4f}")
    print(f"  Mean Causal Rank: {final_eval['mean_causal_rank']:.1f} / 1000")
    print(f"  Exact Causal Ranks: {final_eval['causal_ranks']}")
    print(f"  Detected in Top 5%: {final_eval['detected_all_top_5pct']}")

if __name__ == "__main__":
    train_and_log_epochs(p=6, seed=42, epochs=30)
    train_and_log_epochs(p=10, seed=42, epochs=30)
