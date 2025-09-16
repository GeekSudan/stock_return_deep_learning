# -*- coding: utf-8 -*-
"""
Weekly data pipeline (MV with LSTM μ, plus EW/MI baselines):
- Load Assets_Returns (T,N), Index_Returns (T,), and weekly risk-free (T,)
- Split timewise into train/val/test
- Train multi-output LSTM (predict next-week returns) with SGDM-style two-step optimizer (BLOSAM)
- Test with 4-week rebalancing:
    * MV: min x^T Σ x - λ μ^T x, with μ from LSTM prediction, Σ from past lookback window,
          constraints sum(x)=1, x>=0, solved via projected gradient descent
    * EW: equal weight
    * MI: market index as-is
- Compute 9 risk metrics (weekly -> annualized):
    SR, SoR, OR, CSR, ESR, CR, MR, PR, CPR
- Save: weekly_returns.csv, lstm_preds_test.csv, metrics.csv, alignment_preview.csv
"""

import argparse, os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 你项目里的 SAM/BLOSAM 工具
from bypass_bn import enable_running_stats, disable_running_stats
from blosam import BLOSAM

WEEKS_PER_YEAR = 52

# -------------------- reproducibility --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------- numerics & annualization --------------------
def _to_np(a): return np.asarray(a, dtype=np.float64)
def _sanitize(a):
    a = _to_np(a)
    a[~np.isfinite(a)] = 0.0
    return a

def annualize_mean(mean_weekly):   # weekly -> annualized arithmetic mean
    return mean_weekly * WEEKS_PER_YEAR

def annualize_vol(std_weekly):     # weekly -> annualized vol
    return std_weekly * math.sqrt(WEEKS_PER_YEAR)

def nav_from_returns(r):
    r = _sanitize(r)
    return np.cumprod(1.0 + r)

def drawdown_series(nav):
    nav = _sanitize(nav)
    peak = np.maximum.accumulate(nav)
    return (peak - nav) / np.maximum(peak, 1e-12)

# -------------------- risk metrics (weekly) --------------------
def sharpe_ratio(excess, annualize=True):
    ex = _sanitize(excess)
    mu = ex.mean()
    sd = ex.std(ddof=1)
    if sd < 1e-12: return 0.0
    if annualize: return annualize_mean(mu) / annualize_vol(sd)
    return mu / sd

def sortino_ratio(returns, rf_or_tau=0.0, annualize=True):
    r = _sanitize(returns)
    if np.ndim(rf_or_tau) == 0:
        tau = np.full_like(r, float(rf_or_tau))
    else:
        tau = _sanitize(rf_or_tau)
        if tau.size != r.size: tau = np.full_like(r, float(np.mean(tau)))
    ex2tau = r - tau
    downside = np.minimum(ex2tau, 0.0)
    dd_std = math.sqrt(np.mean(downside**2))
    if dd_std < 1e-12: return 0.0
    numer = np.mean(ex2tau)
    if annualize: return annualize_mean(numer) / annualize_vol(dd_std)
    return numer / dd_std

def omega_ratio(returns, tau=0.0):
    r = _sanitize(returns)
    if np.ndim(tau) == 0:
        t = float(tau)
        pos = np.maximum(r - t, 0.0).mean()
        neg = np.maximum(t - r, 0.0).mean()
    else:
        tau = _sanitize(tau)
        pos = np.maximum(r - tau, 0.0).mean()
        neg = np.maximum(tau - r, 0.0).mean()
    return pos / (neg + 1e-12)

def cvar_lower_tail(x, alpha=0.05):
    x = _sanitize(x)
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    if tail.size == 0: return 0.0
    return -tail.mean()

def conditional_sharpe_ratio(excess, alpha=0.05, annualize=True):
    ex = _sanitize(excess)
    numer = ex.mean()
    denom = cvar_lower_tail(ex, alpha=alpha)
    if denom < 1e-12: return 0.0
    if annualize: return annualize_mean(numer) / denom
    return numer / denom

def entropic_sharpe_ratio(excess, eta=5.0, annualize=True):
    ex = _sanitize(excess)
    m = np.mean(np.exp(eta * ex))
    if not np.isfinite(m) or m <= 0: return 0.0
    ce_week = (1.0 / eta) * np.log(m)
    sd_week = ex.std(ddof=1)
    if sd_week < 1e-12: return 0.0
    if annualize:
        ce_ann = annualize_mean(ce_week)
        sd_ann = annualize_vol(sd_week)
        return ce_ann / sd_ann
    return ce_week / sd_week

def calmar_ratio(returns):
    r = _sanitize(returns)
    nav = nav_from_returns(r)
    dd = drawdown_series(nav)
    mdd = dd.max() if dd.size else 0.0
    if nav.size == 0 or nav[-1] <= 0: return 0.0
    Tn = r.size
    cagr = nav[-1]**(WEEKS_PER_YEAR / max(Tn,1)) - 1.0
    if mdd < 1e-12: return 0.0
    return cagr / mdd

def ulcer_index(returns):
    nav = nav_from_returns(_sanitize(returns))
    dd = drawdown_series(nav)
    return math.sqrt(np.mean(dd**2))

def martin_ratio(returns, rf=0.0):
    r = _sanitize(returns)
    if np.ndim(rf) == 0:
        ex = r - float(rf)
    else:
        rf = _sanitize(rf)
        ex = r - rf
    ui = ulcer_index(r)
    if ui < 1e-12: return 0.0
    return annualize_mean(ex.mean()) / ui

def pain_index(returns):
    nav = nav_from_returns(_sanitize(returns))
    dd = drawdown_series(nav)
    return dd.mean()

def pain_ratio(returns):
    r = _sanitize(returns)
    nav = nav_from_returns(r)
    Tn = r.size
    cagr = nav[-1]**(WEEKS_PER_YEAR / max(Tn,1)) - 1.0
    pi = pain_index(r)
    if pi < 1e-12: return 0.0
    return cagr / pi

def conditional_pain_ratio(returns, alpha=0.05):
    r = _sanitize(returns)
    nav = nav_from_returns(r)
    dd = drawdown_series(nav)
    if dd.size == 0: return 0.0
    q = np.quantile(dd, 1.0 - alpha)
    worst = dd[dd >= q]
    cond = worst.mean() if worst.size > 0 else dd.mean()
    if cond < 1e-12: return 0.0
    Tn = r.size
    cagr = nav[-1]**(WEEKS_PER_YEAR / max(Tn,1)) - 1.0
    return cagr / cond

def compute_all_metrics(series_ret, rf_weekly, alpha=0.05, eta=5.0, tau_for_omega="rf"):
    r = _sanitize(series_ret)
    if np.ndim(rf_weekly) == 0:
        ex = r - float(rf_weekly)
        tau_series = float(rf_weekly)
    else:
        rf = _sanitize(rf_weekly)
        ex = r - rf
        tau_series = rf if tau_for_omega == "rf" else 0.0
    metrics = {
        "SR":  sharpe_ratio(ex, annualize=True),
        "SoR": sortino_ratio(r, rf_or_tau=rf_weekly, annualize=True),
        "OR":  omega_ratio(r, tau=tau_series),
        "CSR": conditional_sharpe_ratio(ex, alpha=alpha, annualize=True),
        "ESR": entropic_sharpe_ratio(ex, eta=eta, annualize=True),
        "CR":  calmar_ratio(r),
        "MR":  martin_ratio(r, rf=rf_weekly),
        "PR":  pain_ratio(r),
        "CPR": conditional_pain_ratio(r, alpha=alpha),
    }
    return metrics

# -------------------- simplex projection (sum=1, x>=0) --------------------
def project_to_simplex(v):
    v = _sanitize(v)
    v[v < 0] = 0.0
    s = v.sum()
    if s <= 1e-12:
        n = v.size
        return np.ones(n) / n
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, v.size + 1)) > (cssv - 1))[0]
    if rho.size == 0:
        return np.ones_like(v) / v.size
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

# -------------------- MV optimization by projected GD --------------------
def mv_optimize_weights(mu, Sigma, lam=1.0, iters=600, lr=0.2, tol=1e-8):
    """
    Solve: min_x  x^T Σ x - lam * μ^T x
           s.t.   sum x = 1, x >= 0
    by projected gradient descent onto simplex.
    """
    N = mu.size
    x = np.ones(N) / N
    Sigma = _sanitize(Sigma)
    mu = _sanitize(mu)
    for _ in range(iters):
        grad = 2.0 * Sigma.dot(x) - lam * mu
        x_new = x - lr * grad
        x_new = project_to_simplex(x_new)
        if np.linalg.norm(x_new - x, 1) < tol:
            x = x_new
            break
        x = x_new
    return x

# -------------------- LSTM (multi-output) --------------------
class MultiOutputLSTM(nn.Module):
    def __init__(self, num_assets, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_assets, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, num_assets)

    def forward(self, x):
        # x: (B, L, N)
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :])  # (B, N)
        return y

def make_windows(returns_2d, start_idx, end_idx, lookback=30):
    R = _sanitize(returns_2d)
    T, N = R.shape
    Xs, Ys = [], []
    s = max(start_idx, lookback)
    e = min(end_idx, T - 1)
    for t in range(s, e):
        Xs.append(R[t - lookback:t, :])
        Ys.append(R[t, :])  # predict next-week returns
    if not Xs:
        return np.zeros((0, lookback, N)), np.zeros((0, N))
    return np.stack(Xs), np.stack(Ys)

def train_lstm_sgdm(R_train, R_val, lookback=30,
                    hidden=32, layers=1, dropout=0.0,
                    lr=1e-2, momentum=0.9, weight_decay=0.0,
                    adaptive=True, rho=0.05, p=2, xi_lr_ratio=10,
                    epochs=50, batch_size=64, device="cpu"):
    """
    使用 BLOSAM 两步更新（兼容 BN）：第二次前向必须重新用 xb，而不是用 pred。
    """
    T, N = R_train.shape
    model = MultiOutputLSTM(N, hidden=hidden, layers=layers, dropout=dropout).to(device)
    opt = BLOSAM(model.parameters(), lr=lr, rho=rho, adaptive=adaptive, p=p,
                 xi_lr_ratio=xi_lr_ratio, momentum_theta=momentum, weight_decay=weight_decay)
    # opt = torch.optim.SGD(
    #         model.parameters(),
    #         lr=lr, momentum=momentum, weight_decay=weight_decay
    #     )

    loss_fn = nn.L1Loss()  # MAE

    # windows（验证集首窗借用训练尾部上下文）
    Xtr, Ytr = make_windows(R_train, 0, T, lookback=lookback)
    Xvl, Yvl = make_windows(np.vstack([R_train[-lookback:], R_val]),
                            lookback, lookback + R_val.shape[0], lookback=lookback)

    def to_tensor(x): return torch.tensor(x, dtype=torch.float32, device=device)
    Xtr, Ytr = to_tensor(Xtr), to_tensor(Ytr)
    Xvl, Yvl = to_tensor(Xvl), to_tensor(Yvl)

    best_val = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(Xtr.shape[0], device=device)
        Xtr_shuf, Ytr_shuf = Xtr[idx], Ytr[idx]
        for i in range(0, Xtr_shuf.shape[0], batch_size):
            xb = Xtr_shuf[i:i+batch_size]
            yb = Ytr_shuf[i:i+batch_size]

            opt.zero_grad()

            # ---- SAM/BLOSAM 第一步：启用 BN 统计，正常前向-反传 ----
            enable_running_stats(model)
            pred = model(xb)                # 输入必须是 xb：(B, L, N)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.first_step(zero_grad=True)

            # ---- SAM/BLOSAM 第二步：关闭 BN 统计，重新用 xb 前向-反传 ----
            disable_running_stats(model)
            pred2 = model(xb)               # 关键修复：再次用 xb，而不是 pred
            loss2 = loss_fn(pred2, yb)
            loss2.backward()
            opt.second_step(zero_grad=True)

            # pred = model(xb)
            # loss = loss_fn(pred, yb)
            # loss.backward()
            # opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xvl), Yvl).item() if Xvl.shape[0] > 0 else 0.0

        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def rolling_predict(model, R_full, test_start, test_end, lookback=30, device="cpu"):
    """
    For each t in [test_start, test_end):
        feed window [t-lookback, t) -> predict returns at week t
    """
    T, N = R_full.shape
    preds = np.zeros((T, N), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for t in range(test_start, test_end):
            if t - lookback < 0: continue
            x = R_full[t - lookback:t, :][None, :, :]  # (1, L, N)
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            y = model(xt).cpu().numpy()[0]            # (N,)
            preds[t, :] = y
    return preds

# -------------------- strategies on test (4-week rebalance) --------------------
def run_strategies_weekly(R_full, I_full, rf_full,
                          train_ratio=0.6, val_ratio=0.15,
                          lookback_lstm=30, lookback_cov=52,
                          rebalance_every=4, mv_lambda=1.0,
                          lstm_hidden=32, lstm_layers=1, lstm_dropout=0.0,
                          lr=1e-2, rho=0.05, adaptive=True, p=2, xi_lr_ratio=10,
                          momentum=0.9, weight_decay=0.0,
                          epochs=50, batch_size=64, device="cpu", seed=42):
    """
    R_full: (T,N) assets weekly returns
    I_full: (T,)   index weekly returns
    rf_full: (T,)  weekly risk-free
    """
    set_seed(seed)

    T, N = R_full.shape
    assert I_full.shape[0] == T, "Index length must match assets length"
    assert rf_full.shape[0] == T, "Risk-free length must match assets length"

    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))
    test_start, test_end = val_end, T

    # LSTM + BLOSAM
    model = train_lstm_sgdm(R_full[:train_end], R_full[train_end:val_end],
                            lookback=lookback_lstm,
                            hidden=lstm_hidden, layers=lstm_layers, dropout=lstm_dropout,
                            lr=lr, momentum=momentum, weight_decay=weight_decay,
                            adaptive=adaptive, rho=rho, p=p, xi_lr_ratio=xi_lr_ratio,
                            epochs=epochs, batch_size=batch_size, device=device)

    # rolling predict next-week returns across test
    preds = rolling_predict(model, R_full, test_start=test_start, test_end=test_end,
                            lookback=lookback_lstm, device=device)  # (T,N)

    # init series
    ew_ret = np.zeros(T)
    mi_ret = _sanitize(I_full.copy())
    mv_ret = np.zeros(T)

    # EW: equal weight at each rebalance
    t = test_start
    while t < test_end:
        w_ew = np.ones(N) / N
        t_next = min(test_end, t + rebalance_every)
        ew_ret[t:t_next] = (R_full[t:t_next] @ w_ew)
        t = t_next
    # pre-test fill (仅保证 CSV 连续性，不参与指标计算)
    if test_start > 0:
        ew_ret[:test_start] = (R_full[:test_start] @ (np.ones(N)/N))

    # MV: μ from LSTM predictions, Σ from past lookback_cov realized returns
    t = test_start
    while t < test_end:
        mu = preds[t, :]
        cov_start = max(0, t - lookback_cov)
        Sigma = np.cov(R_full[cov_start:t].T) if (t - cov_start) >= 2 else np.eye(N) * 1e-4
        w_mv = mv_optimize_weights(mu, Sigma, lam=mv_lambda, iters=600, lr=0.2)
        t_next = min(test_end, t + rebalance_every)
        mv_ret[t:t_next] = (R_full[t:t_next] @ w_mv)
        t = t_next
    # pre-test fill
    if test_start > 0:
        mv_ret[:test_start] = (R_full[:test_start] @ (np.ones(N)/N))

    # metrics on pure test span
    ew_test = ew_ret[test_start:test_end]
    mi_test = mi_ret[test_start:test_end]
    mv_test = mv_ret[test_start:test_end]
    rf_test = rf_full[test_start:test_end]

    m_ew = compute_all_metrics(ew_test, rf_test)
    m_mi = compute_all_metrics(mi_test, rf_test)
    m_mv = compute_all_metrics(mv_test, rf_test)

    out = {
        "spans": {"train": (0, train_end), "val": (train_end, val_end), "test": (test_start, test_end)},
        "preds": preds,
        "ew_ret_full": ew_ret, "mi_ret_full": mi_ret, "mv_ret_full": mv_ret,
        "metrics": {"EW": m_ew, "MI": m_mi, "MV": m_mv}
    }
    return out

# -------------------- tbill loader (txt -> weekly aligned) --------------------
def load_tbill_series(tbill_path, T, yearly_tbill=True):
    """
    Read risk-free series from txt; return length = T weekly series.
    - If yearly_tbill=True: convert annual yield (decimal) to weekly: (1+a)^(1/52)-1
    - If already weekly rates: set yearly_tbill=False
    """
    if not os.path.isfile(tbill_path):
        return np.zeros(T, dtype=np.float64)

    arr = []
    with open(tbill_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                arr.append(float(line))
            except:
                continue
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(T, dtype=np.float64)

    if yearly_tbill:
        arr = (1.0 + arr)**(1.0 / WEEKS_PER_YEAR) - 1.0  # annual -> weekly

    # align to T
    if arr.size >= T:
        return arr[:T]
    reps = int(np.ceil(T / arr.size))
    return np.tile(arr, reps)[:T]

# -------------------- main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--excel_path", type=str, required=True, help="Excel file with Assets_Returns & Index_Returns")
    p.add_argument("--assets_sheet", type=str, default="Assets_Returns")
    p.add_argument("--index_sheet", type=str, default="Index_Returns")
    p.add_argument("--tbill_path", type=str, required=True)
    p.add_argument("--yearly_tbill", action="store_true", help="if txt values are annual yields (decimals)")
    p.add_argument("--train_ratio", type=float, default=0.60)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--lookback_lstm", type=int, default=30)
    p.add_argument("--lookback_cov", type=int, default=52)
    p.add_argument("--rebalance_every", type=int, default=4)
    p.add_argument("--mv_lambda", type=float, default=1.0)
    # LSTM + BLOSAM
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    # SAM/BLOSAM 超参（修正类型）
    p.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM/ASAM.")
    p.add_argument("--adaptive", action="store_true", help="Use Adaptive SAM (ASAM).")
    p.add_argument("--p", default=2, type=int, help="Norm for SAM neighborhood.")
    p.add_argument("--xi_lr_ratio", default=10, type=int, help="xi_lr_ratio to lr")
    p.add_argument("--outdir", type=str, default="./outputs_weekly")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # read excel
    assets = pd.read_excel(args.excel_path, sheet_name=args.assets_sheet).values  # (T,N)
    index_ = pd.read_excel(args.excel_path, sheet_name=args.index_sheet).values.squeeze()  # (T,)
    if index_.ndim > 1: index_ = index_.reshape(-1)
    T, N = assets.shape

    # risk-free (align to T)
    rf_weekly_full = load_tbill_series(args.tbill_path, T=T, yearly_tbill=args.yearly_tbill)

    # alignment preview (first 10 rows)
    preview_len = min(10, T)
    prev = pd.DataFrame(
        np.column_stack([assets[:preview_len, :min(N,5)], index_[:preview_len], rf_weekly_full[:preview_len]]),
        columns=[*(f"A{i+1}" for i in range(min(N,5))), "Index", "RF_weekly"]
    )
    prev.to_csv(os.path.join(args.outdir, "alignment_preview.csv"), index=False)

    # run strategies
    out = run_strategies_weekly(
        assets, index_, rf_weekly_full,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        lookback_lstm=args.lookback_lstm, lookback_cov=args.lookback_cov,
        rebalance_every=args.rebalance_every, mv_lambda=args.mv_lambda,
        lstm_hidden=args.hidden, lstm_layers=args.layers, lstm_dropout=args.dropout,
        lr=args.lr, rho=args.rho, adaptive=args.adaptive, p=args.p, xi_lr_ratio=args.xi_lr_ratio,
        momentum=args.momentum, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device, seed=args.seed
    )

    # save per-week returns
    df = pd.DataFrame({
        "EW_ret": out["ew_ret_full"],
        "MI_ret": out["mi_ret_full"],
        "MV_ret": out["mv_ret_full"],
        "rf_weekly": rf_weekly_full
    })
    df.to_csv(os.path.join(args.outdir, "weekly_returns.csv"), index=False)

    # save test preds
    test_s, test_e = out["spans"]["test"]
    pd.DataFrame(out["preds"][test_s:test_e, :]).to_csv(
        os.path.join(args.outdir, "lstm_preds_test.csv"), index=False
    )

    # save metrics
    rows = []
    for name, m in out["metrics"].items():
        row = {"strategy": name}
        row.update(m)
        rows.append(row)
    md = pd.DataFrame(rows)
    md.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

    # print nice summary
    print(f"[splits] train={out['spans']['train']}, val={out['spans']['val']}, test={out['spans']['test']}")
    for name, m in out["metrics"].items():
        print(name, " ".join([f"{k}={v:.4f}" for k, v in m.items()]))

if __name__ == "__main__":
    main()
