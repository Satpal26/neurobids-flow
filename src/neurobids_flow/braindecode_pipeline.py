"""
NeuroBIDS-Flow — Braindecode EEGNet Pipeline
=============================================
EEGNet deep learning classifier on BIDS-EEG data.

Usage:
    python src/neurobids_flow/braindecode_pipeline.py --bids-root ./bids_output
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   NeuroBIDS-Flow — EEGNet Deep Learning Pipeline        ║
║   Passive BCI | Braindecode + PyTorch                   ║
╚══════════════════════════════════════════════════════════╝
"""


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(bids_root: str, task: str = "workload",
              tmin: float = 0.0, tmax: float = 3.0):
    from neurobids_flow.sklearn_pipeline import load_epochs_from_bids
    return load_epochs_from_bids(bids_root=bids_root, task=task,
                                 tmin=tmin, tmax=tmax)


# ── EEGNet Model ──────────────────────────────────────────────────────────────

def build_eegnet(n_channels: int, n_times: int, n_classes: int):
    """
    Build EEGNet model using Braindecode.
    Falls back to ShallowConvNet if unavailable.
    """
    # Try braindecode 1.x API (EEGNet renamed from EEGNetv4)
    try:
        from braindecode.models import EEGNet
        model = EEGNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            final_conv_length=1,
            final_layer_linear=True,
        )
        log.info("Model      : EEGNet (Braindecode 1.x)")
        return model
    except (TypeError, ImportError):
        pass

    # Try braindecode 0.x API
    try:
        from braindecode.models import EEGNetv4
        model = EEGNetv4(
            n_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=n_times,
            final_conv_length=1,
        )
        log.info("Model      : EEGNetv4 (Braindecode 0.x)")
        return model
    except Exception:
        pass

    log.warning("EEGNet unavailable — using ShallowConvNet fallback")
    return _build_fallback(n_channels, n_times, n_classes)


def _build_fallback(n_channels: int, n_times: int, n_classes: int):
    """Simple fallback ShallowConvNet."""
    class ShallowConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),
                nn.Conv2d(40, 40, (n_channels, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 75), stride=(1, 15)),
                nn.Dropout(0.5),
                nn.Flatten(),
            )
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_size = self.net(dummy).shape[1]
            self.classifier = nn.Linear(flat_size, n_classes)

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.classifier(self.net(x))

    log.info("Model      : ShallowConvNet (fallback)")
    return ShallowConvNet()


def _forward(model, X_batch):
    """Forward pass — handles both (batch, classes) and (batch, classes, times) outputs."""
    out = model(X_batch)
    if out.dim() == 3:
        out = out.mean(dim=-1)
    return out


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class EEGArrayDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = _forward(model, X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = _forward(model, X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(bids_root: str, task: str = "workload",
                 tmin: float = 0.0, tmax: float = 3.0,
                 n_epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 8, val_split: float = 0.2) -> dict:

    print(BANNER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device     : {device}")
    log.info(f"BIDS root  : {bids_root}")
    log.info(f"Task       : {task}")
    log.info(f"Epoch      : [{tmin}, {tmax}] s")
    print()

    log.info("Loading epochs from BIDS dataset...")
    X, y, subjects = load_data(bids_root, task, tmin, tmax)
    log.info(f"Loaded X={X.shape}, y={y.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y).astype(np.int64)
    n_classes = len(le.classes_)
    n_channels = X.shape[1]
    n_times = X.shape[2]

    log.info(f"Classes    : {list(le.classes_)} ({n_classes})")
    log.info(f"Shape      : ch={n_channels}, times={n_times}")
    print()

    dataset = EEGArrayDataset(X, y_enc)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info(f"Train/val  : {n_train}/{n_val} epochs")

    model = build_eegnet(n_channels, n_times, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters : {n_params:,}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    log.info(f"Training for {n_epochs} epochs...")
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>9} | {'Val Acc':>8}")
    print("─" * 55)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.3f} | "
                  f"{val_loss:>9.4f} | {val_acc:>7.3f}")

    print("─" * 55)

    results = {
        "best_val_acc": best_val_acc,
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "n_epochs_trained": n_epochs,
        "n_classes": n_classes,
        "class_labels": list(le.classes_),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_params": n_params,
        "device": str(device),
        "history": history,
    }

    print(f"\n  Best val accuracy  : {best_val_acc:.3f}")
    print(f"  Chance level       : {1.0 / n_classes:.3f}")
    print(f"  Device             : {device}")
    if best_val_acc > 1.0 / n_classes:
        print("  ✓ Above chance level")
    else:
        print("  ✗ At or below chance (expected for synthetic data)")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroBIDS-Flow EEGNet Pipeline")
    parser.add_argument("--bids-root", default="./bids_output")
    parser.add_argument("--task", default="workload")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_pipeline(
        bids_root=args.bids_root,
        task=args.task,
        tmin=args.tmin,
        tmax=args.tmax,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )