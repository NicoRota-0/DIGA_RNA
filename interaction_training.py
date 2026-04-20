
"""Reusable training / testing utilities for RNA interaction classifier."""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from dataclasses import dataclass, field
from copy import deepcopy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


INPUT_LENGTH = 1280


def set_seed(seed: Optional[int] = None) -> None:
    """Set NumPy / PyTorch seeds."""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairs_to_arrays(pairs_list, embeddings_dict, label, min_length: int = INPUT_LENGTH):
    """
    Convert list of pair tuples into X/y arrays.

    Expected pair format:
        [((s1, s2), (t1, t2)), ...]
    """
    X_list = []
    y_list = []
    missing = 0

    for (s1, s2), (t1, t2) in pairs_list:
        if s1 not in embeddings_dict or s2 not in embeddings_dict:
            missing += 1
            continue

        e1 = np.asarray(embeddings_dict[s1], dtype=np.float32)
        e2 = np.asarray(embeddings_dict[s2], dtype=np.float32)

        if e1.shape[0] < min_length:
            e1 = np.pad(e1, (min_length - e1.shape[0], 0), mode="constant")
        if e2.shape[0] < min_length:
            e2 = np.pad(e2, (min_length - e2.shape[0], 0), mode="constant")

        X_list.append(np.concatenate([e1, e2], axis=0))
        y_list.append(int(label))

    if missing:
        print(f"Skipped {missing} pairs due to missing embeddings.")

    if not X_list:
        return None, None

    return np.stack(X_list), np.asarray(y_list, dtype=np.int64)


class BalancedBatchSampler:
    def __init__(self, labels, batch_size, neg_batch_ratio):
        self.labels = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
        self.batch_size = int(batch_size)
        self.neg_batch_ratio = float(neg_batch_ratio)
        self.neg_indices = np.where(self.labels == 0)[0]
        self.pos_indices = np.where(self.labels == 1)[0]
        self.num_neg_per_batch = int(self.batch_size * self.neg_batch_ratio)
        self.num_batches = len(self.neg_indices) // max(1, self.num_neg_per_batch)
        self.leftover_negatives = len(self.neg_indices) % max(1, self.num_neg_per_batch)

    def __iter__(self):
        neg_idx = np.random.permutation(self.neg_indices)
        pos_idx = self.pos_indices
        batch_start = 0

        for batch_num in range(self.num_batches):
            if batch_num < self.num_batches - 1 or self.leftover_negatives == 0:
                num_neg_in_batch = self.num_neg_per_batch
            else:
                num_neg_in_batch = self.leftover_negatives

            neg_batch = neg_idx[batch_start:batch_start + num_neg_in_batch]
            batch_start += num_neg_in_batch

            num_pos_in_batch = self.batch_size - num_neg_in_batch
            pos_batch = np.random.choice(pos_idx, num_pos_in_batch, replace=True)
            batch_indices = np.concatenate([neg_batch, pos_batch])
            np.random.shuffle(batch_indices)
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches if self.num_batches > 0 else 1


def get_data_loaders_without_replacement_from_arrays(
    X_train, y_train, X_valid, y_valid, batch_size, num_workers: int = 2
):
    """Build train / validation DataLoaders from NumPy arrays."""
    train_tensor = torch.from_numpy(X_train).float()
    train_labels = torch.from_numpy(y_train)
    train_dataset = TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(X_valid).float()
    valid_labels = torch.from_numpy(y_valid)
    valid_dataset = TensorDataset(valid_tensor, valid_labels)

    train_sampler = BalancedBatchSampler(train_labels, batch_size, neg_batch_ratio=0.5)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader


def make_test_loader(X_test, y_test, batch_size, num_workers: int = 2):
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def cosine_annealing_with_warmup(epoch, warmup_epochs=4, max_epochs=50, min_lr=1e-5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    return (cosine_decay * (1 - min_lr) + min_lr)


class InteractionNN(nn.Module):
    def __init__(self, input_dim=INPUT_LENGTH * 2, hidden_dim=1024, num_layers=4, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x).view(-1)


@dataclass
class TrainingConfig:
    input_dimension: int = INPUT_LENGTH * 2
    patience: int = 10
    batch_size: int = 512
    validation_split_ratio: float = 0.2
    learning_rate: float = 5e-4
    num_layers: int = 4
    dropout: float = 0.2
    hidden_dimension: int = 1024
    min_lr: float = 5e-5
    epochs: int = 50
    warmup_epochs: int = 4
    seed: Optional[int] = 42
    num_workers: int = 2


@dataclass
class TrainArtifacts:
    model: nn.Module
    scaler: StandardScaler
    threshold: float
    device: str
    test_loader: DataLoader
    val_loader: DataLoader
    train_loader: DataLoader
    val_probs: np.ndarray
    val_labels: np.ndarray
    history: List[Dict[str, float]] = field(default_factory=list)
    best_val_loss: float = float("inf")


@dataclass
class TestResults:
    probabilities: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray


def _train_val_split(X_all, y_all, validation_split_ratio: float, seed: Optional[int]):
    n_tot = len(y_all)
    n_train = int((1 - validation_split_ratio) * n_tot)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_tot)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return X_all[train_idx], y_all[train_idx], X_all[val_idx], y_all[val_idx]


def _best_threshold_mcc(val_probs, val_labels):
    thresholds = np.linspace(0, 1, 101)
    mcc = [matthews_corrcoef(val_labels, (val_probs > t).astype(int)) for t in thresholds]
    return thresholds[int(np.nanargmax(mcc))]


def train_model(fold, embeddings_dict, config: Optional[TrainingConfig] = None, device: Optional[str] = None):
    """
    Train model, compute validation threshold, and prepare test loader.
    """
    config = config or TrainingConfig()
    set_seed(config.seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    positive_train = fold["train"]["positives"]
    negative_train = fold["train"]["negatives"]
    positive_test = fold["test"]["positives"]
    negative_test = fold["test"]["negatives"]

    Xp, yp = pairs_to_arrays(positive_train, embeddings_dict, label=1)
    Xn, yn = pairs_to_arrays(negative_train, embeddings_dict, label=0)
    if Xp is None or Xn is None:
        raise ValueError('Training set empty after embedding filtering.')
    X_all = np.concatenate([Xp, Xn], axis=0)
    y_all = np.concatenate([yp, yn], axis=0)

    X_train, y_train, X_val, y_val = _train_val_split(X_all, y_all, config.validation_split_ratio, config.seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    Xp_test, yp_test = pairs_to_arrays(positive_test, embeddings_dict, label=1)
    Xn_test, yn_test = pairs_to_arrays(negative_test, embeddings_dict, label=0)
    if Xp_test is None or Xn_test is None:
        raise ValueError('Test set empty after embedding filtering.')
    X_test = np.concatenate([Xp_test, Xn_test], axis=0)
    y_test = np.concatenate([yp_test, yn_test], axis=0)
    X_test = scaler.transform(X_test)

    train_loader, val_loader = get_data_loaders_without_replacement_from_arrays(
        X_train, y_train, X_val, y_val, batch_size=config.batch_size, num_workers=config.num_workers
    )
    test_loader = make_test_loader(X_test, y_test, batch_size=config.batch_size, num_workers=config.num_workers)

    model = InteractionNN(
        input_dim=config.input_dimension,
        hidden_dim=config.hidden_dimension,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_annealing_with_warmup(
            e,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.epochs,
            min_lr=config.min_lr,
        ),
    )
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improvement = 0
    history = []

    for epoch in range(config.epochs):
        model.train()
        loss_train = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(xb).view(-1)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        loss_train /= max(1, len(train_loader))

        model.eval()
        loss_val = 0.0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device, dtype=torch.float32)

                output = model(xb).view(-1)
                loss = criterion(output, yb)

                loss_val += loss.item()
                val_probs.extend(output.detach().cpu().numpy())
                val_labels.extend(yb.detach().cpu().numpy())

        loss_val /= max(1, len(val_loader))

        history.append({"epoch": epoch + 1, "train_loss": loss_train, "val_loss": loss_val})

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        scheduler.step()
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train={loss_train:.4f} | Val={loss_val:.4f} | LR={lr_current:.2e}"
        )

        if no_improvement >= config.patience:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            output = model(xb).view(-1)
            val_probs.extend(output.detach().cpu().numpy())
            val_labels.extend(yb.numpy())

    val_probs = np.asarray(val_probs)
    val_labels = np.asarray(val_labels)
    best_thr = _best_threshold_mcc(val_probs, val_labels)
    print("Soglia ottimale:", best_thr)

    return TrainArtifacts(
        model=model,
        scaler=scaler,
        threshold=best_thr,
        device=device,
        test_loader=test_loader,
        val_loader=val_loader,
        train_loader=train_loader,
        val_probs=val_probs,
        val_labels=val_labels,
        history=history,
        best_val_loss=best_val_loss,
    )


def test_model(model, test_loader, threshold: float, device: Optional[str] = None):
    """
    Evaluate trained model on test set.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            output = model(xb).view(-1)
            probabilities.extend(output.detach().cpu().numpy())
            labels.extend(yb.numpy())

    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    predictions = (probabilities > threshold).astype(int)

    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
    }

    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions, normalize="true")
    except Exception:
        cm = np.zeros((2, 2), dtype=float)

    return TestResults(
        probabilities=probabilities,
        labels=labels,
        predictions=predictions,
        metrics=metrics,
        confusion_matrix=cm,
    )
