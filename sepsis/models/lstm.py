"""Causal (unidirectional) LSTM / GRU that emits a sepsis risk at EVERY hour of a stay."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from ..config import DYNAMIC, LABEL, PID, TIME

N_FEATURES = 2 * len(DYNAMIC) + 3   # normalised values + measured-masks + age, gender, ICULOS


def compute_stats(pre_train: pd.DataFrame) -> dict:
    dyn = pre_train[DYNAMIC]
    std = dyn.std().replace(0, 1).fillna(1)
    return {"mean": dyn.mean().to_dict(), "std": std.to_dict(), "median": dyn.median().to_dict(),
            "age_mean": float(pre_train["Age"].mean()), "age_std": float(pre_train["Age"].std() or 1.0)}


def _row_features(pre: pd.DataFrame, s: dict) -> np.ndarray:
    n = len(DYNAMIC)
    X = np.zeros((len(pre), N_FEATURES), dtype=np.float32)
    for i, c in enumerate(DYNAMIC):
        v = pre[c].fillna(s["median"][c]).to_numpy(float)
        X[:, i] = np.clip((v - s["mean"][c]) / s["std"][c], -6, 6)
        X[:, n + i] = pre[f"{c}_measured"].to_numpy()
    X[:, 2 * n] = (pre["Age"].fillna(s["age_mean"]).to_numpy(float) - s["age_mean"]) / s["age_std"]
    X[:, 2 * n + 1] = pre["Gender"].fillna(0.5).to_numpy(float)
    X[:, 2 * n + 2] = pre[TIME].to_numpy(float) / 100.0
    return X


def to_sequences(pre: pd.DataFrame, stats: dict):
    """Pad patients to (N, T, F). Padding sits at the END, so a causal RNN never sees it
    before real data. Returns X, y, mask and the row positions of each patient."""
    pre = pre.reset_index(drop=True)
    X = _row_features(pre, stats)
    y = pre[LABEL].to_numpy(np.float32) if LABEL in pre else np.zeros(len(pre), np.float32)
    groups = sorted(pre.groupby(PID, sort=False).indices.values(), key=lambda a: a[0])
    T = max(len(ix) for ix in groups)
    Xs = np.zeros((len(groups), T, N_FEATURES), np.float32)
    Ys = np.zeros((len(groups), T), np.float32)
    M = np.zeros((len(groups), T), bool)
    for n, ix in enumerate(groups):
        Xs[n, :len(ix)], Ys[n, :len(ix)], M[n, :len(ix)] = X[ix], y[ix], True
    return Xs, Ys, M, groups


class SepsisRNN(nn.Module):
    def __init__(self, n_in=N_FEATURES, hidden=64, layers=2, dropout=0.2, cell="lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out).squeeze(-1)          # (N, T) logits


@torch.no_grad()
def predict_probs(model, X: np.ndarray, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def predict_rows(model, pre: pd.DataFrame, stats: dict, device: str = "cpu") -> np.ndarray:
    """Per-row probabilities aligned with `pre` (index positions)."""
    Xs, _, _, groups = to_sequences(pre, stats)
    probs = predict_probs(model, Xs, device=device)
    out = np.full(len(pre), np.nan)
    for n, ix in enumerate(groups):
        out[ix] = probs[n, :len(ix)]
    return out


def train_rnn(train_seq, val_seq, cell="lstm", epochs=25, batch_size=64, lr=1e-3,
              hidden=64, layers=2, dropout=0.2, patience=5, seed=42, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, mtr = (torch.from_numpy(a) for a in train_seq)
    Xva, yva, mva = val_seq
    pos = float(ytr[mtr].sum())
    neg = float(mtr.sum()) - pos
    pos_weight = torch.tensor(np.sqrt(neg / max(pos, 1.0)), dtype=torch.float32, device=device)

    model = SepsisRNN(hidden=hidden, layers=layers, dropout=dropout, cell=cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, mb = Xtr[idx].to(device), ytr[idx].to(device), mtr[idx].to(device)
            loss = (loss_fn(model(xb), yb) * mb).sum() / mb.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        pv = predict_probs(model, Xva, device=device)
        ap = average_precision_score(yva[mva], pv[mva])
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_auprc": float(ap)})
        if verbose:
            print(f"[{cell}] epoch {epoch:02d} loss={total/len(perm):.4f} val_auprc={ap:.4f}")
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), history
