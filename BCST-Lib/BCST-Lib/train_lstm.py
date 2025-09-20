# -*- coding: utf-8 -*-
"""
LSTM Portfolio Backtest (CLI args + optimizer choice)
Print gross, *annualized* ex-post metrics: SRp, SoRp, ORp, CSRp, ESRp.

Annualization:
- SRp/SoRp/CSRp/ESRp: multiply weekly ratios by sqrt(annualize_factor) (52 by default).
- ORp: 'block' annualization (default): compound weekly returns into annual blocks,
        compound rf benchmark in the same blocks, then compute Omega on those blocks.
        ('none' keeps weekly Omega as-is; not recommended if you need annualized Omega.)

Paper alignment notes:
- SoR uses rb = rf (risk-free) as downside benchmark.
- CSRp uses CVaR on portfolio loss (-r_p) with confidence θ (default 0.95).
- ESRp uses EVaR on loss (-r_p) with the same θ.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Optional optimizer & BN utilities
try:
    from optimizers.blosam import BLOSAM
except Exception:
    BLOSAM = None
try:
    from utility.bypass_bn import enable_running_stats, disable_running_stats
except Exception:
    def enable_running_stats(*args, **kwargs): pass
    def disable_running_stats(*args, **kwargs): pass


# ----------------------------- Reproducibility -----------------------------
def set_all_seeds(seed: int = 42, deterministic: bool = True):
    """Fix random seeds and optionally force deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device: str) -> str:
    """Return device string: 'cuda' if available and device='auto'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ----------------------------- I/O loaders -----------------------------
def load_headerless_excel_asset_and_index(excel_path: str,
                                          asset_sheet: str = "asset_return",
                                          index_sheet: str = "index_return"):
    """Read asset and index returns from headerless Excel sheets."""
    assets_df = pd.read_excel(excel_path, sheet_name=asset_sheet, header=None)
    index_df  = pd.read_excel(excel_path, sheet_name=index_sheet, header=None)

    # Coerce to numeric and drop all-empty rows
    assets_df = assets_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    index_df  = index_df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    # Align length
    T = min(len(assets_df), len(index_df))
    assets_df = assets_df.iloc[:T, :].copy()
    index_df  = index_df.iloc[:T, :].copy()

    # 1-based week index
    assets_df.index = np.arange(1, T+1)
    index_df.index  = np.arange(1, T+1)

    # Index return is single column
    index_sr = index_df.iloc[:, 0].astype(float)
    index_sr.name = "index_return"

    # Name asset columns
    assets_df.columns = [f"asset_{i+1}" for i in range(assets_df.shape[1])]
    return assets_df, index_sr


def load_headerless_txt_rf(txt_path: str):
    """Read risk-free returns from a headerless TXT (commas or whitespace)."""
    try:
        rf_df = pd.read_csv(txt_path, header=None, sep=None, engine="python")
    except Exception:
        rf_df = pd.read_csv(txt_path, header=None)

    if rf_df.shape[1] == 1 and len(rf_df) > 1:
        rf_sr = rf_df.iloc[:, 0].astype(float)
    else:
        vals = rf_df.values.flatten()
        vals = [v for v in vals if pd.notna(v)]
        rf_sr = pd.Series([float(v) for v in vals])

    rf_sr.name = "rf"
    rf_sr.index = np.arange(1, len(rf_sr)+1)
    return rf_sr


# ----------------------------- Preprocess -----------------------------
def winsorize_rowwise(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99):
    """Cross-sectional (per-week) winsorization at given quantiles."""
    def _clip_row(row):
        lo = row.quantile(lower_q)
        hi = row.quantile(upper_q)
        return row.clip(lower=lo, upper=hi)
    return df.apply(_clip_row, axis=1)


def preprocess_assets(assets_df: pd.DataFrame,
                      winsor_q: float = 0.01,
                      fill_method: str = "zero"):
    """Winsorize and fill NaNs."""
    if winsor_q is not None and winsor_q > 0:
        assets_df = winsorize_rowwise(assets_df, winsor_q, 1 - winsor_q)
    if fill_method == "zero":
        assets_df = assets_df.fillna(0.0)
    elif fill_method == "ffill":
        assets_df = assets_df.ffill().fillna(0.0)
    else:
        raise ValueError("fill_method must be 'zero' or 'ffill'")
    return assets_df


# ----------------------------- Metrics helpers -----------------------------
def sharpe_ratio(excess_returns: np.ndarray, eps: float = 1e-8):
    """Standard Sharpe: mean(excess) / std(excess)."""
    if excess_returns is None or len(excess_returns) == 0:
        return np.nan
    m = float(np.mean(excess_returns))
    s = float(np.std(excess_returns, ddof=0))
    return m / (s + eps)


def sortino_ratio_expost(port_ret: np.ndarray, rf: np.ndarray, eps: float = 1e-8):
    """
    Ex-post Sortino with rb = rf (per paper):
    SoRp = mean(port_ret - rf) / sqrt(mean( max(rf - port_ret, 0)^2 ))
    """
    if port_ret is None or len(port_ret) == 0:
        return np.nan
    excess = port_ret - rf
    lpm2 = np.mean(np.maximum(rf - port_ret, 0.0) ** 2)
    denom = math.sqrt(lpm2 + eps)
    return float(np.mean(excess)) / denom


def omega_ratio_expost(port_ret: np.ndarray, rf: np.ndarray, eps: float = 1e-12):
    """
    Ex-post Omega (via LPM1, rb = rf):
    ORp = mean(port_ret - rf) / mean( max(rf - port_ret, 0) ) + 1
    """
    if port_ret is None or len(port_ret) == 0:
        return np.nan
    excess = port_ret - rf
    lpm1 = np.mean(np.maximum(rf - port_ret, 0.0))
    return float(np.mean(excess)) / (lpm1 + eps) + 1.0


def cvar_loss(values: np.ndarray, alpha: float = 0.95):
    """
    CVaR (Expected Shortfall) of a LOSS series at level alpha:
    - Quantile q = VaR_alpha
    - CVaR = mean(loss | loss >= q)
    """
    if values is None or len(values) == 0:
        return np.nan
    q = np.quantile(values, alpha)
    tail = values[values >= q]
    if tail.size == 0:
        return float(q)
    return float(np.mean(tail))


def evar_loss(values: np.ndarray, alpha: float = 0.95):
    """
    EVaR (entropic VaR) of a LOSS series at level alpha using SAA:
    EVaR_alpha = min_{rho>0} rho * log( (1/((1-alpha)*m)) * sum_j exp(loss_j / rho) ).
    Use log-sum-exp for numerical stability and 1D search over rho.
    """
    vals = np.asarray(values, dtype=float)
    m = vals.size
    if m == 0:
        return np.nan

    def evar_obj(rho):
        z = vals / rho
        zmax = np.max(z)
        lse = zmax + np.log(np.mean(np.exp(z - zmax)))  # log((1/m) * sum exp(z))
        return rho * (lse - math.log(1 - alpha))

    rhos = np.logspace(-4, 0, 200)  # [1e-4, 1]
    vals_e = np.array([evar_obj(r) for r in rhos])
    i = int(np.argmin(vals_e))

    lo = rhos[max(0, i-1)]
    hi = rhos[min(len(rhos)-1, i+1)]
    for _ in range(30):  # golden-section refinement
        g = (math.sqrt(5) - 1) / 2
        x1 = hi - g * (hi - lo)
        x2 = lo + g * (hi - lo)
        f1, f2 = evar_obj(x1), evar_obj(x2)
        if f1 < f2:
            hi = x2
        else:
            lo = x1
    rho_star = 0.5 * (lo + hi)
    return float(evar_obj(rho_star))


def csr_expost(port_ret: np.ndarray, rf: np.ndarray, alpha: float = 0.95, eps: float = 1e-12):
    """
    Ex-post Conditional Sharpe ratio:
    CSRp = mean(port_ret - rf) / CVaR_alpha(loss), where loss = - port_ret.
    """
    excess = port_ret - rf
    loss = -port_ret
    cvar = cvar_loss(loss, alpha=alpha)
    return float(np.mean(excess)) / (cvar + eps)


def esr_expost(port_ret: np.ndarray, rf: np.ndarray, alpha: float = 0.95, eps: float = 1e-12):
    """
    Ex-post Entropic Sharpe ratio:
    ESRp = mean(port_ret - rf) / EVaR_alpha(loss), where loss = - port_ret.
    """
    excess = port_ret - rf
    loss = -port_ret
    evar = evar_loss(loss, alpha=alpha)
    return float(np.mean(excess)) / (evar + eps)


# ---- Omega annualization via compounding blocks ----
def block_compound_returns(r: np.ndarray, block: int):
    """
    Compound simple weekly returns into non-overlapping blocks of length 'block':
    R_block = prod(1 + r_t) - 1  for each block.
    """
    m = len(r) // block
    if m <= 0:
        return None
    rr = r[:m*block].reshape(m, block)
    return np.prod(1.0 + rr, axis=1) - 1.0


# ----------------------------- Covariance -----------------------------
def sample_cov(returns_window: pd.DataFrame):
    """Sample covariance on a window (rowvar=False since columns are assets)."""
    return np.cov(returns_window.values, rowvar=False)


def ewma_cov(returns_window: pd.DataFrame, halflife: float = 26):
    """EWMA covariance via simple recursion on demeaned returns."""
    X = returns_window.values
    X = X - X.mean(axis=0, keepdims=True)
    alpha = 1 - 0.5**(1/halflife)
    N = X.shape[1]
    S = np.zeros((N, N))
    w = 0.0
    for t in range(X.shape[0]):
        x = X[t:t+1].T
        S = (1 - alpha) * S + alpha * (x @ x.T)
        w = (1 - alpha) * w + alpha
    if w > 1e-12:
        S = S / w
    return S


def shrink_diag(S: np.ndarray, shrink: float = 0.0):
    """Diagonal shrinkage toward diag(S)."""
    if shrink <= 0.0:
        return S
    D = np.diag(np.diag(S))
    return (1 - shrink) * S + shrink * D


# ----------------------------- Projections & Optimizer -----------------------------
def project_to_simplex(z: np.ndarray) -> np.ndarray:
    """Projection to simplex {w>=0, sum w = 1}."""
    z = z.astype(float)
    n = z.size
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n+1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones_like(z) / n
    rho = ind[cond].max()
    theta = cssv[rho-1] / rho
    w = np.maximum(z - theta, 0.0)
    s = w.sum()
    return (np.ones_like(z)/n) if s <= 0 else (w / s)


def project_to_capped_simplex(z: np.ndarray, ub: float) -> np.ndarray:
    """Projection to {w>=0, sum w=1, w_i<=ub} via bisection on theta."""
    n = z.size
    lo, hi = -10.0, 10.0
    for _ in range(60):
        theta = 0.5 * (lo + hi)
        w = np.minimum(np.maximum(z - theta, 0.0), ub)
        s = w.sum()
        if s > 1.0:
            lo = theta
        else:
            hi = theta
    w = np.minimum(np.maximum(z - hi, 0.0), ub)
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / w.sum()


def optimize_markowitz(mu: np.ndarray,
                       Sigma: np.ndarray,
                       lam: float = 10.0,
                       w0: np.ndarray = None,
                       steps: int = 800,
                       lr: float = 0.05,
                       gamma_turnover: float = 0.0,
                       last_w: np.ndarray = None,
                       cap: float = None) -> np.ndarray:
    """
    Gradient ascent on: mu^T w - lam * w^T Σ w - gamma * ||w - last_w||_1
    s.t. w in simplex (and optionally w_i <= cap).
    """
    n = mu.size
    w = np.ones(n) / n if w0 is None else w0.astype(float)
    if last_w is None:
        last_w = np.zeros(n)
    Sigma = 0.5 * (Sigma + Sigma.T)
    proj = (lambda x: project_to_capped_simplex(x, cap)) if (cap is not None and cap > 0) else project_to_simplex
    w = proj(w)

    for _ in range(steps):
        grad = mu - 2 * lam * Sigma.dot(w)
        if gamma_turnover > 0.0:
            grad += - gamma_turnover * np.sign(w - last_w)
        w = proj(w + lr * grad)
    return w


# ----------------------------- Model -----------------------------
class LSTMForecaster(nn.Module):
    """Simple LSTM forecaster that outputs next-window mean returns per asset."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.head(h)


# ----------------------------- Dataset windows -----------------------------
def build_rebalance_points(T: int, lookback_L: int, hold_H: int):
    """Rebalance at t = L, L+H, L+2H, ... while t+H-1 <= T."""
    return list(range(lookback_L, T - hold_H + 1, hold_H))


def build_windows_from_rebals(returns_df: pd.DataFrame, rebal_points: list, lookback_L: int, hold_H: int):
    """
    Build (X, y) windows for training:
    - X: last L weeks (T×N -> L×N)
    - y: next H-week mean per asset (N-dim)
    """
    X_list, y_list, t_list = [], [], []
    for t in rebal_points:
        X_win = returns_df.iloc[t - lookback_L : t, :]
        Y_win = returns_df.iloc[t : t + hold_H, :]
        if X_win.isna().any().any() or Y_win.isna().any().any():
            continue
        X_list.append(X_win.values)
        y_list.append(Y_win.values.mean(axis=0))
        t_list.append(t)
    if len(X_list) == 0:
        raise ValueError("No valid windows; check NaNs or increase data length.")
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y, t_list


def fit_scaler_per_asset(X_train: np.ndarray):
    """Per-asset standardization from training windows only."""
    M, L, N = X_train.shape
    flat = X_train.reshape(M*L, N)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def transform_X(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Apply per-asset standardization."""
    return (X - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)


# ----------------------------- Training -----------------------------
def train_lstm_predict_mu(assets_df: pd.DataFrame,
                          lookback_L: int,
                          hold_H: int,
                          train_ratio: float = 0.7,
                          hidden_size: int = 64,
                          num_layers: int = 1,
                          dropout: float = 0.0,
                          lr: float = 1e-3,
                          batch_size: int = 64,
                          max_epochs: int = 200,
                          patience: int = 20,
                          device: str = "cpu",
                          verbose: bool = True,
                          persist_split: bool = True,
                          split_tag: str = None,
                          seed: int = 42,
                          huber_delta: float = 1.0,
                          opt_name: str = "blosam",
                          momentum: float = 0.9,
                          weight_decay: float = 5e-4,
                          blosam_rho: float = 0.05,
                          blosam_p: int = 2,
                          blosam_xi_lr_ratio: float = 3.0,
                          blosam_momentum_theta: float = 0.9,
                          blosam_adaptive: bool = True,
                          grad_clip: float = 1.0,
                          use_scheduler: bool = True,
                          scheduler_patience: int = 5,
                          min_lr: float = 1e-5):
    """Train LSTM to predict next H-week mean returns per asset."""
    set_all_seeds(seed, deterministic=True)
    rng = np.random.RandomState(seed)

    T, N = assets_df.shape
    rebal_points = build_rebalance_points(T, lookback_L, hold_H)
    if len(rebal_points) < 4:
        raise ValueError("Too few rebal points. Decrease L/H or provide more data.")
    split_idx_default = max(2, int(len(rebal_points) * train_ratio))

    X_all, y_all, t_all = build_windows_from_rebals(assets_df, rebal_points, lookback_L, hold_H)

    # Persist the split so repeated runs are comparable
    tag = split_tag or "default"
    split_file = f"split_rebals_L{lookback_L}_H{hold_H}_{tag}.npz"
    if persist_split and os.path.exists(split_file):
        npz = np.load(split_file, allow_pickle=True)
        t_saved = list(npz["t_all"])
        split_idx = int(npz["split_idx"])
        t_used = [t for t in t_all if t in t_saved]
        if len(t_used) < 4:
            t_used = t_all
            split_idx = split_idx_default
    else:
        t_used = t_all
        split_idx = split_idx_default
        if persist_split:
            np.savez_compressed(split_file, t_all=np.array(t_all), split_idx=split_idx)

    valid_positions = [rebal_points.index(t) for t in t_used]
    pos_map = {t:i for i,t in enumerate(t_all)}
    sel_idx = [pos_map[t] for t in t_used]
    X_all, y_all = X_all[sel_idx], y_all[sel_idx]

    train_mask = [pos < split_idx for pos in valid_positions]
    val_mask   = [pos >= split_idx for pos in valid_positions]
    if sum(train_mask) == 0 or sum(val_mask) == 0:
        raise ValueError("Empty train/val split; adjust train_ratio or check NaNs.")

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]

    # Normalize using training only
    mean, std = fit_scaler_per_asset(X_train)
    X_train_n = transform_X(X_train, mean, std)
    X_val_n   = transform_X(X_val,   mean, std)

    device = select_device(device)
    model = LSTMForecaster(input_size=N, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)

    # Optimizer selection
    if opt_name.lower() == "sgdm":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        is_sam_like = False
    elif opt_name.lower() == "blosam":
        if BLOSAM is None:
            raise ImportError("BLOSAM not found. Use --opt sgdm or install BLOSAM.")
        opt = BLOSAM(
            model.parameters(),
            lr=lr,
            rho=blosam_rho,
            p=blosam_p,
            xi_lr_ratio=blosam_xi_lr_ratio,
            momentum_theta=blosam_momentum_theta,
            weight_decay=weight_decay,
            adaptive=blosam_adaptive
        )
        is_sam_like = True
    else:
        raise ValueError("--opt must be 'sgdm' or 'blosam'.")

    loss_fn = nn.HuberLoss(delta=huber_delta)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=scheduler_patience,
                                  min_lr=min_lr, verbose=verbose) if use_scheduler else None

    def to_tensor(x): return torch.tensor(x, dtype=torch.float32, device=device)
    Xtr_t = to_tensor(X_train_n); ytr_t = to_tensor(y_train)
    Xva_t = to_tensor(X_val_n);   yva_t = to_tensor(y_val)

    best_val = float("inf")
    best_state = None
    wait = 0
    hist = {"train": [], "val": [], "lr": []}

    if verbose:
        print(f"[Fingerprint] T={T}, N={N}, windows={len(t_used)}, train={sum(train_mask)}, val={sum(val_mask)}, split_idx={split_idx}, opt={opt_name}")

    for epoch in range(1, max_epochs+1):
        model.train()
        idx = rng.permutation(len(X_train_n))
        batches = int(np.ceil(len(X_train_n) / batch_size))
        train_loss_acc = 0.0

        for b in range(batches):
            sel = idx[b*batch_size : (b+1)*batch_size]
            xb = Xtr_t[sel]; yb = ytr_t[sel]

            if is_sam_like:
                opt.zero_grad()
                enable_running_stats(model)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.mean().backward()
                opt.first_step(zero_grad=True)

                disable_running_stats(model)
                loss_fn(model(xb), yb).mean().backward()
                opt.second_step(zero_grad=True)

                train_loss_acc += loss.item() * len(sel)
            else:
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()
                train_loss_acc += loss.item() * len(sel)

        train_loss = train_loss_acc / len(X_train_n)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t)
            val_loss = loss_fn(val_pred, yva_t).item()

        hist["train"].append(train_loss)
        hist["val"].append(val_loss)
        if hasattr(opt, "param_groups"):
            hist["lr"].append(opt.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            lr_show = opt.param_groups[0]["lr"] if hasattr(opt, "param_groups") else lr
            print(f"[Epoch {epoch:03d}] train={train_loss:.6e} | val={val_loss:.6e} | lr={lr_show:.2e}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stop @ {epoch}, best val={best_val:.6e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model, (mean, std), hist, rebal_points, split_idx


# ----------------------------- Backtest -----------------------------
def rolling_backtest_lstm(args):
    """End-to-end rolling backtest and print gross *annualized* SRp/SoRp/ORp/CSRp/ESRp."""
    set_all_seeds(args.seed, deterministic=args.deterministic)

    # 1) Load data
    assets_df, index_sr = load_headerless_excel_asset_and_index(args.excel_path, args.asset_sheet, args.index_sheet)
    rf_sr = load_headerless_txt_rf(args.txt_path)

    # 2) Align length and drop all-NaN weeks
    T0 = min(len(assets_df), len(rf_sr))
    assets_df = assets_df.iloc[:T0, :].copy()
    rf_sr = rf_sr.iloc[:T0].copy()
    index_sr = index_sr.iloc[:T0].copy()

    valid_rows = ~assets_df.isna().all(axis=1)
    if valid_rows.sum() < len(valid_rows):
        assets_df = assets_df.loc[valid_rows].copy()
        rf_sr     = rf_sr.loc[valid_rows].copy()
        index_sr  = index_sr.loc[valid_rows].copy()

    # 3) Preprocess
    assets_df = preprocess_assets(assets_df, winsor_q=args.winsor_q, fill_method=args.fill_method)

    T, N = assets_df.shape
    if T < args.lookback_L + args.hold_H:
        raise ValueError(f"Data too short: T={T}, need at least {args.lookback_L + args.hold_H} weeks.")

    # 4) Train LSTM to predict next-window mean returns
    model, scaler, hist, rebal_points, split_idx = train_lstm_predict_mu(
        assets_df=assets_df,
        lookback_L=args.lookback_L,
        hold_H=args.hold_H,
        train_ratio=args.train_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        verbose=args.verbose,
        persist_split=not args.no_persist_split,
        split_tag=args.split_tag or os.path.basename(args.excel_path).split(".")[0],
        seed=args.seed,
        huber_delta=args.huber_delta,
        opt_name=args.opt,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        blosam_rho=args.blosam_rho,
        blosam_p=args.blosam_p,
        blosam_xi_lr_ratio=args.blosam_xi_lr_ratio,
        blosam_momentum_theta=args.blosam_momentum_theta,
        blosam_adaptive=not args.blosam_no_adaptive,
        grad_clip=args.grad_clip,
        use_scheduler=not args.no_scheduler,
        scheduler_patience=args.scheduler_patience,
        min_lr=args.min_lr
    )
    mean, std = scaler

    # 5) Rolling invest: predict µ, estimate Σ, optimize w, hold for H weeks
    port_ret_gross = np.full(T, np.nan)
    weights, weeks = [], []
    last_w = None

    model.eval()
    for t in rebal_points[split_idx:]:
        window = assets_df.iloc[t - args.lookback_L : t, :]
        Xn = ((window.values - mean.reshape(1, -1)) / std.reshape(1, -1))[None, ...]
        with torch.no_grad():
            mu_pred = model(torch.tensor(Xn, dtype=torch.float32)).numpy().reshape(-1)

        # Covariance
        if args.cov_method == "sample":
            S = sample_cov(window)
        elif args.cov_method == "ewma":
            S = ewma_cov(window, halflife=args.cov_halflife)
        else:
            raise ValueError("cov_method must be 'sample' or 'ewma'")
        S = shrink_diag(S, args.shrink)

        # Optimize Markowitz (gross, no costs)
        w0 = last_w if last_w is not None else np.ones(N)/N
        cap = None if (args.weight_cap is None or args.weight_cap < 0) else args.weight_cap
        w = optimize_markowitz(mu_pred, S, lam=args.lam, w0=w0, steps=800, lr=0.05,
                               gamma_turnover=args.gamma_turnover,
                               last_w=last_w if last_w is not None else np.zeros(N),
                               cap=cap)

        # Realized weekly returns over next H weeks (gross)
        future = assets_df.iloc[t : t + args.hold_H, :]
        pr = future.values.dot(w)  # weekly gross portfolio returns
        idx = future.index
        port_ret_gross[idx - 1] = pr

        weights.append(w)
        weeks.append(t)
        last_w = w

    # 6) Ex-post metrics (weekly first)
    rf = rf_sr.values[:T]
    valid = ~np.isnan(port_ret_gross)
    r = port_ret_gross[valid]
    rf_v = rf[valid]

    SRp_w  = sharpe_ratio(r - rf_v)
    SoRp_w = sortino_ratio_expost(r, rf_v)
    ORp_w  = omega_ratio_expost(r, rf_v)
    CSRp_w = csr_expost(r, rf_v, alpha=args.theta)
    ESRp_w = esr_expost(r, rf_v, alpha=args.theta)

    # Annualization
    scale = math.sqrt(args.annualize_factor)  # e.g., sqrt(52)
    SRp  = SRp_w * scale
    SoRp = SoRp_w * scale
    CSRp = CSRp_w * scale
    ESRp = ESRp_w * scale

    if args.omega_ann == "block":
        # Compound weekly returns & rf into annual blocks, then compute Omega on blocks
        r_blk  = block_compound_returns(r, args.annualize_factor)
        rf_blk = block_compound_returns(rf_v, args.annualize_factor)
        if r_blk is not None and rf_blk is not None and len(r_blk) == len(rf_blk) and len(r_blk) > 0:
            ORp = omega_ratio_expost(r_blk, rf_blk)
        else:
            ORp = ORp_w  # fallback to weekly Omega if not enough data
    else:
        ORp = ORp_w  # no annualization for Omega

    metrics = {
        "SRp":  SRp,
        "SoRp": SoRp,
        "ORp":  ORp,
        "CSRp": CSRp,
        "ESRp": ESRp
    }

    # Save optional artifacts
    weights_df = pd.DataFrame(weights, index=weeks, columns=assets_df.columns)
    weights_df.to_csv(args.out_weights)
    pd.Series(port_ret_gross, index=assets_df.index, name="r_gross").to_csv(args.out_r_gross)
    pd.DataFrame({"train":hist["train"], "val":hist["val"], "lr":hist["lr"]}).to_csv(args.out_train_hist, index=False)

    # Final print: ONLY the five gross *annualized* ex-post metrics
    print("=== Ex-post (gross, annualized) metrics ===")
    for k, v in metrics.items():
        print(f"{k:>5}: {v:.6f}")
    print("Saved CSVs:", args.out_weights, args.out_r_gross, args.out_train_hist)

    return metrics


# ----------------------------- CLI -----------------------------
def build_parser():
    p = argparse.ArgumentParser("LSTM Portfolio Backtest (gross *annualized* SRp/SoRp/ORp/CSRp/ESRp)")
    # data
    p.add_argument("--excel_path", type=str, required=True)
    p.add_argument("--txt_path", type=str, required=True)
    p.add_argument("--asset_sheet", type=str, default="asset_return")
    p.add_argument("--index_sheet", type=str, default="index_return")
    # windows
    p.add_argument("--lookback_L", type=int, default=52)
    p.add_argument("--hold_H", type=int, default=12)
    p.add_argument("--train_ratio", type=float, default=0.7)
    # model
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # optimizer choice
    p.add_argument("--opt", type=str, default="blosam", choices=["sgdm", "blosam"])
    # SGDM hyperparams
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    # BLOSAM hyperparams
    p.add_argument("--blosam_rho", type=float, default=0.05)
    p.add_argument("--blosam_p", type=int, default=2)
    p.add_argument("--blosam_xi_lr_ratio", type=int, default=2)
    p.add_argument("--blosam_momentum_theta", type=float, default=0.9)
    p.add_argument("--blosam_no_adaptive", action="store_true")
    # training stability
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--scheduler_patience", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-5)
    # covariance/optimization
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--shrink", type=float, default=0.1)
    p.add_argument("--cov_method", type=str, default="ewma", choices=["sample", "ewma"])
    p.add_argument("--cov_halflife", type=int, default=26)
    p.add_argument("--gamma_turnover", type=float, default=0.0)
    p.add_argument("--weight_cap", type=float, default=0.2, help="Per-asset upper bound; <0 disables cap")
    # preprocess
    p.add_argument("--winsor_q", type=float, default=0.01)
    p.add_argument("--fill_method", type=str, default="zero", choices=["zero", "ffill"])
    # reproducibility & split persistence
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--no_persist_split", action="store_true")
    p.add_argument("--split_tag", type=str, default=None)
    # ex-post metric hyperparams
    p.add_argument("--theta", type=float, default=0.95, help="CVaR/EVaR confidence level for CSRp/ESRp")
    p.add_argument("--annualize_factor", type=int, default=52, help="Weeks per year for annualization")
    p.add_argument("--omega_ann", type=str, default="none", choices=["block", "none"],
                   help="How to annualize Omega: 'block' compounding or 'none'")
    # outputs
    p.add_argument("--out_weights", type=str, default="weights_LSTM_OPT.csv")
    p.add_argument("--out_r_gross", type=str, default="portfolio_weekly_returns_gross_OPT.csv")
    p.add_argument("--out_train_hist", type=str, default="train_history.csv")
    # misc
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    rolling_backtest_lstm(args)
