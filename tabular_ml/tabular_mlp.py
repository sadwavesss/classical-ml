"""Income prediction on UCI Adult with a PyTorch MLP, compared with simple baselines."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from tabular_ml.data import RESULTS_DIR, load_adult
from tabular_ml.plotting import plt, save

SEED = 42


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.net(x)


def encode(X: pd.DataFrame, y: pd.Series):
    X = pd.get_dummies(X, dtype=np.float32)
    labels = LabelEncoder().fit(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(np.float32), labels.transform(y), test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train).astype(np.float32), scaler.transform(X_test).astype(np.float32),
            y_train, y_test, list(labels.classes_))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total, probs = 0.0, 0, 0, []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
        probs.append(logits.softmax(1)[:, 1].cpu())
    return loss_sum / total, correct / total, torch.cat(probs).numpy()


def loader(X, y, **kwargs) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.as_tensor(y, dtype=torch.long)), **kwargs)


def predict_proba(model, X) -> np.ndarray:
    device = next(model.parameters()).device
    return evaluate(model, loader(X, np.zeros(len(X)), batch_size=1024), nn.CrossEntropyLoss(), device)[2]


def train(X_train, X_val, y_train, y_val, epochs=50, hidden_dim=64, lr=1e-3, batch_size=256):
    """Trains for a fixed number of epochs and also keeps the weights with the lowest validation loss."""
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    eval_train = loader(X_train, y_train, batch_size=1024)
    val_loader = loader(X_val, y_val, batch_size=1024)

    model = MLP(X_train.shape[1], hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        tr_loss, tr_acc, _ = evaluate(model, eval_train, criterion, device)
        va_loss, va_acc, _ = evaluate(model, val_loader, criterion, device)
        for key, value in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[key].append(value)
        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

    best_model = MLP(X_train.shape[1], hidden_dim).to(device)
    best_model.load_state_dict(best_state)
    return model, best_model, best_epoch, history


def plot_history(history: dict, best_epoch: int, out: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "acc")):
        ax.plot(epochs, history[f"train_{metric}"], label="train")
        ax.plot(epochs, history[f"val_{metric}"], label="validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"best val loss (epoch {best_epoch})")
        ax.set(xlabel="epoch", title="Cross-entropy loss" if metric == "loss" else "Accuracy")
        ax.grid(True)
        ax.legend()
    save(fig, out / "training_curves.png")


def scores(y_true, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "f1": f1_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, prob)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "tabular_mlp")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)

    X_full, X_test, y_full, y_test, classes = encode(*load_adult(args.data))
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.1, random_state=SEED,
                                                      stratify=y_full)
    print(f"train {X_train.shape}, val {X_val.shape}, test {X_test.shape}, classes {classes}")

    final_model, best_model, best_epoch, history = train(X_train, X_val, y_train, y_val, epochs=args.epochs)
    logreg = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    majority = np.bincount(y_train).argmax()

    metrics = {
        "Majority class": {"accuracy": accuracy_score(y_test, np.full_like(y_test, majority))},
        "Logistic Regression": scores(y_test, logreg.predict_proba(X_test)[:, 1]),
        f"MLP, {args.epochs} epochs": scores(y_test, predict_proba(final_model, X_test)),
        f"MLP, best val epoch ({best_epoch})": scores(y_test, predict_proba(best_model, X_test)),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    plot_history(history, best_epoch, args.out)
    (args.out / "metrics.json").write_text(json.dumps(
        {"train_size": len(X_train), "val_size": len(X_val), "test_size": len(X_test),
         "features": X_train.shape[1], "epochs": args.epochs, "best_epoch": best_epoch, "metrics": metrics},
        indent=2, default=float), encoding="utf-8")
    print(pd.DataFrame(metrics).T.round(4))


if __name__ == "__main__":
    main()
