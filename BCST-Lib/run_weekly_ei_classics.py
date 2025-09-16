# -*- coding: utf-8 -*-
"""
Weekly Enhanced Indexation (classical, non-DL) under the same setup:
- Input: Excel (Assets_Returns T×N, Index_Returns T×1), txt risk-free (weekly or annual -> weekly)
- Split: timewise train/val/test (default 60/15/25), 4-week rebalancing on test
- At each rebalance: fit an EI model on a rolling in-sample window, get weights, hold for next 4 weeks
- Models implemented (cvxpy):
  * CZεSD  : LP minimizing total shortfall below benchmark + tiny secondary objective
  * RMZ-SSD: SSD via constraints on sum of k smallest excess returns >= 0, ∀k
  * LR-ASSD: like RMZ-SSD with uniform slack θ; minimize θ - α * mean(excess)
  * L-SSD  : compact SSD（对等概率离散样本与 RMZ 约束等价实现）
  * KP-SSD : spectral (Power3) weighted SSD: ∑ w_k * sum_smallest(excess, k) ≥ 0
- Performance metrics: SR, SoR, OR, CSR, ESR, CR, MR, PR, CPR (weekly -> annualized)
- Outputs: weekly_returns.csv (per strategy), metrics.csv, alignment_preview.csv
"""

import argparse, os, math, random
import numpy as np
import pandas as pd
import cvxpy as cp

WEEKS_PER_YEAR = 52

# ================= Utilities =================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)

def _to_np(a): return np.asarray(a, dtype=np.float64)
def _san(a):
    a = _to_np(a); a[~np.isfinite(a)] = 0.0; return a

def annualize_mean(mw): return mw * WEEKS_PER_YEAR
def annualize_vol(sw):  return sw * math.sqrt(WEEKS_PER_YEAR)

def nav_from_returns(r):
    r = _san(r); return np.cumprod(1.0 + r)

def drawdown_series(nav):
    nav = _san(nav)
    peak = np.maximum.accumulate(nav)
    return (peak - nav) / np.maximum(peak, 1e-12)

# ----- risk metrics -----
def sharpe_ratio(excess, annualize=True):
    ex = _san(excess); mu = ex.mean(); sd = ex.std(ddof=1)
    if sd < 1e-12: return 0.0
    return (annualize_mean(mu) / annualize_vol(sd)) if annualize else (mu / sd)

def sortino_ratio(returns, rf_or_tau=0.0, annualize=True):
    r = _san(returns)
    if np.ndim(rf_or_tau) == 0: tau = np.full_like(r, float(rf_or_tau))
    else:
        tau = _san(rf_or_tau)
        if tau.size != r.size: tau = np.full_like(r, float(np.mean(tau)))
    ex2tau = r - tau
    downside = np.minimum(ex2tau, 0.0)
    dd_std = math.sqrt(np.mean(downside**2))
    if dd_std < 1e-12: return 0.0
    numer = np.mean(ex2tau)
    return (annualize_mean(numer) / annualize_vol(dd_std)) if annualize else (numer / dd_std)

def omega_ratio(returns, tau=0.0):
    r = _san(returns)
    if np.ndim(tau) == 0:
        t = float(tau); pos = np.maximum(r - t, 0.0).mean(); neg = np.maximum(t - r, 0.0).mean()
    else:
        tau = _san(tau); pos = np.maximum(r - tau, 0.0).mean(); neg = np.maximum(tau - r, 0.0).mean()
    return pos / (neg + 1e-12)

def cvar_lower_tail(x, alpha=0.05):
    x = _san(x); q = np.quantile(x, alpha); tail = x[x <= q]
    if tail.size == 0: return 0.0
    return -tail.mean()

def conditional_sharpe_ratio(excess, alpha=0.05, annualize=True):
    ex = _san(excess); numer = ex.mean(); denom = cvar_lower_tail(ex, alpha=alpha)
    if denom < 1e-12: return 0.0
    return (annualize_mean(numer) / denom) if annualize else (numer / denom)

def entropic_sharpe_ratio(excess, eta=5.0, annualize=True):
    """
    ESR = CE(eta) / sigma, CE(eta) = (1/eta) * log E[exp(eta * excess)]
    数值稳健：裁剪 eta*ex 防止 exp 溢出
    """
    ex = _san(excess)
    z = np.clip(eta * ex, -50.0, 50.0)
    m = np.mean(np.exp(z))
    if not np.isfinite(m) or m <= 0:
        return 0.0
    ce_week = (1.0 / eta) * np.log(m)
    sd_week = ex.std(ddof=1)
    if sd_week < 1e-12: return 0.0
    if annualize:
        return annualize_mean(ce_week) / annualize_vol(sd_week)
    return ce_week / sd_week

def calmar_ratio(returns):
    r = _san(returns)
    nav = nav_from_returns(r); dd = drawdown_series(nav)
    mdd = dd.max() if dd.size else 0.0
    if nav.size == 0 or nav[-1] <= 0: return 0.0
    Tn = r.size; cagr = nav[-1]**(WEEKS_PER_YEAR / max(Tn,1)) - 1.0
    if mdd < 1e-12: return 0.0
    return cagr / mdd

def ulcer_index(returns):
    nav = nav_from_returns(_san(returns)); dd = drawdown_series(nav)
    return math.sqrt(np.mean(dd**2))

def martin_ratio(returns, rf=0.0):
    r = _san(returns)
    ex = r - (float(rf) if np.ndim(rf)==0 else _san(rf))
    ui = ulcer_index(r); 
    if ui < 1e-12: return 0.0
    return annualize_mean(ex.mean()) / ui

def pain_index(returns):
    nav = nav_from_returns(_san(returns)); dd = drawdown_series(nav)
    return dd.mean()

def pain_ratio(returns):
    r = _san(returns); nav = nav_from_returns(r); Tn = r.size
    cagr = nav[-1]**(WEEKS_PER_YEAR / max(Tn,1)) - 1.0
    pi = pain_index(r)
    if pi < 1e-12: return 0.0
    return cagr / pi

def conditional_pain_ratio(returns, alpha=0.05):
    r = _san(returns); nav = nav_from_returns(r); dd = drawdown_series(nav)
    if dd.size == 0: return 0.0
    q = np.quantile(dd, 1.0 - alpha); worst = dd[dd >= q]
    cond = worst.mean() if worst.size > 0 else dd.mean()
    if cond < 1e-12: return 0.0
    Tn = r.size; cagr = nav[-1]**(WEEKS_PER_YEAR / max(Tn,1)) - 1.0
    return cagr / cond

def compute_all_metrics(series_ret, rf_weekly, alpha=0.05, eta=5.0, tau_for_omega="rf"):
    r = _san(series_ret)
    if np.ndim(rf_weekly) == 0:
        ex = r - float(rf_weekly); tau_series = float(rf_weekly)
    else:
        rf = _san(rf_weekly); ex = r - rf; tau_series = rf if tau_for_omega=="rf" else 0.0
    return {
        "SR": sharpe_ratio(ex, True),
        "SoR": sortino_ratio(r, rf_or_tau=rf_weekly, annualize=True),
        "OR": omega_ratio(r, tau=tau_series),
        "CSR": conditional_sharpe_ratio(ex, alpha=alpha, annualize=True),
        "ESR": entropic_sharpe_ratio(ex, eta=eta, annualize=True),
        "CR": calmar_ratio(r),
        "MR": martin_ratio(r, rf=rf_weekly),
        "PR": pain_ratio(r),
        "CPR": conditional_pain_ratio(r, alpha=alpha),
    }

# ================= Data helpers =================
def load_tbill_series(tbill_path, T, yearly_tbill=True):
    if not os.path.isfile(tbill_path):
        return np.zeros(T, dtype=np.float64)
    arr = []
    with open(tbill_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: arr.append(float(line))
            except: continue
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0: return np.zeros(T, dtype=np.float64)
    if yearly_tbill:
        arr = (1.0 + arr)**(1.0 / WEEKS_PER_YEAR) - 1.0
    if arr.size >= T: return arr[:T]
    reps = int(np.ceil(T / arr.size))
    return np.tile(arr, reps)[:T]

# ================= EI optimizers (cvxpy) =================
def _safe_norm_weights(w, n, name, verbose=True):
    """
    规范化权重；若无效则回退等权并打印警告。
    """
    if w is None or not np.all(np.isfinite(w)):
        if verbose:
            print(f"[WARN] {name} solver returned invalid weights, fallback to equal weight for this rebalance.")
        return np.ones(n)/n
    w = np.maximum(np.asarray(w).reshape(-1), 0.0)
    s = w.sum()
    if s <= 1e-12:
        if verbose:
            print(f"[WARN] {name} weights nearly zero-sum, fallback to equal weight.")
        return np.ones(n)/n
    return w / s

# ======== DCP-safe SSD implementations via threshold + hinge linearization ========

def _build_thresholds(rb_hist, qgrid=None):
    """基准窗口的阈值集合（分位数网格）"""
    if qgrid is None:
        # 25 个阈值：2%~98% 每 4%
        qgrid = np.linspace(0.02, 0.98, 25)
    th = np.quantile(_san(rb_hist), qgrid)
    # 去重 & 排序，避免重复阈值导致冗余约束
    th = np.unique(np.round(th, 12))
    return th

def _bench_shortfall_rhs(rb_hist, thresholds):
    """右端项： E[max(0, η - r_b)] 对每个 η"""
    rb = _san(rb_hist)
    rhs = []
    for eta in thresholds:
        rhs.append(np.maximum(eta - rb, 0.0).mean())
    return np.asarray(rhs)

def _ssd_lp_core(Rwin, Iwin, thresholds, eps_relax=0.0, extra_constraints=None,
                 objective_kind="max_mean_excess", secondary_eps=1e-6, spectral_weights=None):
    """
    通用 SSD LP 核心：
      变量: x (N, nonneg), z (K,M, nonneg) 线性化 max(0, η - Rw)
      约束: z_{k,s} ≥ η_k - (Rwin x)_s,  z_{k,s} ≥ 0,  sum(x)=1
            (1/M)sum_s z_{k,s} ≤ RHS_k + eps_relax * |RHS_k|
      目标: 
        - objective_kind == "max_mean_excess": 最大化 mean(Rx - I)
        - objective_kind == "min_shortfall":   最小化 ∑_{k} α_k * (1/M)∑_s z_{k,s} - ε * mean(Rx - I)
          （α_k 默认为 1；可用 spectral_weights 传入 KP 的谱权重）
    """
    R = _san(Rwin); rb = _san(Iwin)
    M, N = R.shape; K = thresholds.size

    # 右端项
    rhs = _bench_shortfall_rhs(rb, thresholds)
    if eps_relax > 0:
        rhs = rhs + eps_relax * np.maximum(np.abs(rhs), 1e-12)

    x = cp.Variable(N, nonneg=True)
    z = cp.Variable((K, M), nonneg=True)
    Rx = R @ x
    cons = [cp.sum(x) == 1.0]

    # 线性化 hinge & SSD 约束
    for k in range(K):
        eta = thresholds[k]
        cons += [
            z[k, :] >= eta - Rx,   # z >= η - Rx
            z[k, :] >= 0,
            cp.sum(z[k, :]) / M <= rhs[k]
        ]
    if extra_constraints:
        cons += extra_constraints

    # 目标
    delta = Rx - rb
    if objective_kind == "max_mean_excess":
        obj = cp.Maximize(cp.sum(delta) / M)
    elif objective_kind == "min_shortfall":
        # 可选谱权重（用于 KP）：α_k >= 0，默认全 1
        if spectral_weights is None:
            alpha = np.ones(K)
        else:
            alpha = np.asarray(spectral_weights, dtype=np.float64)
            if alpha.size != K:
                raise ValueError("spectral_weights length must match thresholds length")
        shortfalls = []
        for k in range(K):
            shortfalls.append((alpha[k] / M) * cp.sum(z[k, :]))
        # 主目标：最小化加权短缺；次目标：鼓励更高平均超额（打破并列最优）
        obj = cp.Minimize(cp.sum(shortfalls) - secondary_eps * (cp.sum(delta) / M))
    else:
        raise ValueError("unknown objective_kind")

    prob = cp.Problem(obj, cons)
    status = "unknown"
    try:
        # 线性规划/凸规划：ECOS 或 SCS 都可
        prob.solve(solver=cp.ECOS, verbose=False)
        status = prob.status
        if x.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
            status = prob.status
    except Exception as e:
        status = f"exception:{type(e).__name__}"

    return x.value, status

# ---- 各方法基于相同核心，区别在阈值集/松弛/权重/目标 ----

def cz_eps_sd_weights(Rwin, Iwin, extra_constraints=None, sec_eps=1e-6):
    """
    CZεSD：最小化“总体短缺”（所有阈值等权）并在目标里加次级项 -ε·mean(excess)
    """
    th = _build_thresholds(Iwin)
    w, st = _ssd_lp_core(Rwin, Iwin, th, eps_relax=0.0, extra_constraints=extra_constraints,
                         objective_kind="min_shortfall", secondary_eps=sec_eps, spectral_weights=None)
    return w, st

def rmz_ssd_weights(Rwin, Iwin, extra_constraints=None):
    """
    RMZ-SSD：SSD 约束 + 最大化 mean(excess)
    """
    th = _build_thresholds(Iwin)
    w, st = _ssd_lp_core(Rwin, Iwin, th, eps_relax=0.0, extra_constraints=extra_constraints,
                         objective_kind="max_mean_excess")
    return w, st

def lr_assd_weights(Rwin, Iwin, extra_constraints=None, alpha_small=1e-4, eps_relax=1e-3):
    """
    LR-ASSD：允许 ε 松弛（近似 SSD），并“最小化短缺 - α·mean(excess)”
    这里把“统一 θ”改为对 RHS 做相对 ε 松弛（数值更稳，效果等价于‘几乎支配’）
    """
    th = _build_thresholds(Iwin)
    w, st = _ssd_lp_core(Rwin, Iwin, th, eps_relax=eps_relax, extra_constraints=extra_constraints,
                         objective_kind="min_shortfall", secondary_eps=alpha_small, spectral_weights=None)
    return w, st

def l_ssd_weights(Rwin, Iwin, extra_constraints=None):
    """
    L-SSD：与 RMZ-SSD 相同的 SSD 约束 + 最大化 mean(excess)
    """
    return rmz_ssd_weights(Rwin, Iwin, extra_constraints=extra_constraints)

def kp_ssd_weights(Rwin, Iwin, gamma=3.0, extra_constraints=None):
    """
    KP-SSD（谱支配 Power-γ）：把谱权重离散在阈值上，最小化加权短缺 - ε·mean(excess)
    """
    th = _build_thresholds(Iwin)
    K = th.size
    # 构造 Power-γ 的离散权重：α_k ~ ((k/K)^γ - ((k-1)/K)^γ), k=1..K
    ks = np.arange(1, K+1, dtype=np.float64)
    alpha = (ks / K)**gamma - ((ks-1) / K)**gamma
    # 归一化（防数值波动）
    alpha = alpha / np.maximum(alpha.sum(), 1e-12)
    w, st = _ssd_lp_core(Rwin, Iwin, th, eps_relax=0.0, extra_constraints=extra_constraints,
                         objective_kind="min_shortfall", secondary_eps=1e-6, spectral_weights=alpha)
    return w, st


# ================= Rolling backtest (no DL) =================
def run_ei_strategies_weekly(R_full, I_full, rf_full,
                             train_ratio=0.60, val_ratio=0.15,  # val仅占位，窗口选取用 lookback_sd
                             lookback_sd=104,                   # SD模型回看窗（默认2年=104周）
                             rebalance_every=4,
                             device_seed=42):
    """
    与之前 LSTM 实验相同：时间切分 & 4周再平衡。
    在测试期每个再平衡时点，用最近 lookback_sd 周作为 in-sample，拟合权重并持有到下一个再平衡。
    """
    set_seed(device_seed)
    R_full = _san(R_full); I_full = _san(I_full); rf_full = _san(rf_full)
    T, N = R_full.shape
    # 切分
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))
    test_start, test_end = val_end, T

    # 结果时间序列
    rets = {
        "EW": np.zeros(T),
        "MI": I_full.copy(),
        "CZepsSD": np.zeros(T),
        "RMZ_SSD": np.zeros(T),
        "LR_ASSD": np.zeros(T),
        "L_SSD": np.zeros(T),
        "KP_SSD": np.zeros(T),
    }

    # 先把测试前段填等权（只为 CSV 连续，指标不会用到）
    if test_start > 0:
        w_eq = np.ones(N)/N
        for k in rets:
            if k != "MI":
                rets[k][:test_start] = (R_full[:test_start] @ w_eq)

    # 测试期滚动
    t = test_start
    step_prints = {0, rebalance_every, 2*rebalance_every}  # 前三次再平衡点打印诊断
    while t < test_end:
        win_end = t
        win_start = max(0, win_end - lookback_sd)
        Rwin = _san(R_full[win_start:win_end, :])
        Iwin = _san(I_full[win_start:win_end])
        m, n = Rwin.shape
        if m < 4:
            # 窗口太短时退化为等权
            w_cz = w_rmz = w_lr = w_l = w_kp = np.ones(n)/n
            st_cz = st_rmz = st_lr = st_l = st_kp = "fallback_equal_weight"
        else:
            # 求解各模型
            w_cz,  st_cz  = cz_eps_sd_weights(Rwin, Iwin, sec_eps=1e-6)
            w_rmz, st_rmz = rmz_ssd_weights(Rwin, Iwin)
            w_lr,  st_lr  = lr_assd_weights(Rwin, Iwin, alpha_small=1e-4)
            w_l,   st_l   = l_ssd_weights(Rwin, Iwin)
            w_kp,  st_kp  = kp_ssd_weights(Rwin, Iwin, gamma=3.0)

            # 规范化 / 回退保护
            w_cz  = _safe_norm_weights(w_cz,  n, "CZepsSD")
            w_rmz = _safe_norm_weights(w_rmz, n, "RMZ_SSD")
            w_lr  = _safe_norm_weights(w_lr,  n, "LR_ASSD")
            w_l   = _safe_norm_weights(w_l,   n, "L_SSD")
            w_kp  = _safe_norm_weights(w_kp,  n, "KP_SSD")

        # 诊断打印：前3个再平衡点
        if (t - test_start) in step_prints:
            def _brief(name, w, st):
                ww = np.round(w[:min(5, w.size)], 4)
                print(f"[{name}] status={st} w[:5]={ww} sum={w.sum():.4f}")
            _brief("CZepsSD", w_cz, st_cz)
            _brief("RMZ_SSD", w_rmz, st_rmz)
            _brief("LR_ASSD", w_lr, st_lr)
            _brief("L_SSD",   w_l,  st_l)
            _brief("KP_SSD",  w_kp, st_kp)

        t_next = min(test_end, t + rebalance_every)
        Rblk = R_full[t:t_next, :]
        # 应用权重获得未来rebalance区间收益
        rets["EW"][t:t_next]       = (Rblk @ (np.ones(n)/n))
        rets["CZepsSD"][t:t_next]  = (Rblk @ w_cz)
        rets["RMZ_SSD"][t:t_next]  = (Rblk @ w_rmz)
        rets["LR_ASSD"][t:t_next]  = (Rblk @ w_lr)
        rets["L_SSD"][t:t_next]    = (Rblk @ w_l)
        rets["KP_SSD"][t:t_next]   = (Rblk @ w_kp)
        t = t_next

    # 指标（仅测试窗）
    rf_test = rf_full[test_start:test_end]
    metrics = {}
    for name, series in rets.items():
        ser = series[test_start:test_end]
        metrics[name] = compute_all_metrics(ser, rf_test)

    spans = {"train": (0, train_end), "val": (train_end, val_end), "test": (test_start, test_end)}
    return rets, metrics, spans

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_path", required=True, type=str)
    ap.add_argument("--assets_sheet", default="Assets_Returns", type=str)
    ap.add_argument("--index_sheet",  default="Index_Returns",   type=str)
    ap.add_argument("--tbill_path",   required=True, type=str)
    ap.add_argument("--yearly_tbill", action="store_true")
    ap.add_argument("--train_ratio",  type=float, default=0.60)
    ap.add_argument("--val_ratio",    type=float, default=0.15)
    ap.add_argument("--lookback_sd",  type=int,   default=104)  # 2y window
    ap.add_argument("--rebalance_every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="./outputs_ei_classics")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True); set_seed(args.seed)

    # 读数据
    assets = pd.read_excel(args.excel_path, sheet_name=args.assets_sheet).values  # (T,N)
    index_ = pd.read_excel(args.excel_path, sheet_name=args.index_sheet).values.squeeze()
    if index_.ndim > 1: index_ = index_.reshape(-1)
    T, N = assets.shape
    rf_weekly = load_tbill_series(args.tbill_path, T=T, yearly_tbill=args.yearly_tbill)

    # 对齐预览
    preview_len = min(10, T)
    prev = pd.DataFrame(
        np.column_stack([assets[:preview_len, :min(N,5)], index_[:preview_len], rf_weekly[:preview_len]]),
        columns=[*(f"A{i+1}" for i in range(min(N,5))), "Index", "RF_weekly"]
    )
    prev.to_csv(os.path.join(args.outdir, "alignment_preview.csv"), index=False)

    # 回测
    rets, metrics, spans = run_ei_strategies_weekly(assets, index_, rf_weekly,
                                                    train_ratio=args.train_ratio,
                                                    val_ratio=args.val_ratio,
                                                    lookback_sd=args.lookback_sd,
                                                    rebalance_every=args.rebalance_every,
                                                    device_seed=args.seed)
    # 保存收益
    df = pd.DataFrame({
        "EW_ret": rets["EW"],
        "MI_ret": rets["MI"],
        "CZepsSD_ret": rets["CZepsSD"],
        "RMZ_SSD_ret": rets["RMZ_SSD"],
        "LR_ASSD_ret": rets["LR_ASSD"],
        "L_SSD_ret": rets["L_SSD"],
        "KP_SSD_ret": rets["KP_SSD"],
        "rf_weekly": rf_weekly
    })
    df.to_csv(os.path.join(args.outdir, "weekly_returns.csv"), index=False)

    # 保存指标
    rows = []
    for name, m in metrics.items():
        row = {"strategy": name}; row.update(m); rows.append(row)
    md = pd.DataFrame(rows)
    md.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

    # 控制台摘要
    print(f"[splits] train={spans['train']}, val={spans['val']}, test={spans['test']}")
    for name, m in metrics.items():
        print(name, " ".join([f"{k}={v:.4f}" for k,v in m.items()]))

if __name__ == "__main__":
    main()
