"""Training, evaluation and reproducibility helpers."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def evaluate(model, loader, device) -> Tuple[float, float]:
    """Return classification accuracy and macro-F1 on a data loader."""
    model.eval()
    predictions, targets = [], []
    correct = total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        predictions.extend(preds.cpu().tolist())
        targets.extend(y.cpu().tolist())

    accuracy = correct / max(total, 1)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return accuracy, macro_f1


def _sync_if_cuda(device) -> None:
    """Synchronize CUDA so wall-clock timings include queued GPU work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_model(model, train_loader, val_loader, device, epochs, lr, weight_decay):
    """Train a model and restore the checkpoint with best validation accuracy.

    ``total_train_time`` measures training only (validation is excluded). CUDA is
    explicitly synchronized around the timed section to avoid under-reporting
    asynchronous GPU work.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_val_acc = -1.0
    best_state = None
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        _sync_if_cuda(device)
        start = time.perf_counter()
        running_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        _sync_if_cuda(device)
        epoch_time = time.perf_counter() - start
        total_train_time += epoch_time

        val_acc, val_f1 = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(n_samples, 1),
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "epoch_time_sec": epoch_time,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, total_train_time


@dataclass
class ExperimentResult:
    """One row in the experiment-level results table."""

    dataset: str
    model_type: str
    kernel_size: int
    theoretical_receptive_field: int
    num_params: int
    best_val_accuracy: float
    best_val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    total_train_time_sec: float
    avg_epoch_time_sec: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int

    def to_dict(self):
        return asdict(self)
