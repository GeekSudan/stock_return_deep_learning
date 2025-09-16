# -*- coding: utf-8 -*-
# classics_fair_nd.py
# 经典策略（MV/多种SSD）在与 ND 代码完全一致的实验设置下的公平对比版本。
# - 数据读取/切分/评估 与 nd_train.py 对齐
# - 基准 EW/MI 与 ND 的口径一致
# - 使用与 ND 相同的 ex_post_metrics_generic（SR/SoR/OR/CSR/ESR，含 EVaR/CVaR 细节）
# - 滚动 OOS、支持 --rebalance；支持保存逐周超额与权重

import argparse, os, math, re
import numpy as np
import pandas as pd

EPS = 1e-12
WEEKS_PER_YEAR = 52

# ------------------------
# 通用数值工具（对齐 ND）
# ------------------------
def sanitize(a, fill=0.0):
    a = np.asarray(a, dtype=np.float64)
    a[~np.isfinite(a)] = fill
    return a

def safe_div(a, b, eps=EPS):
    return a / (b if abs(b) > eps else (eps if b >= 0 else -eps))

# ------------------------
# 与 ND 一致的后验指标（SR/SoR/Ω/CSR/ESR）
# ------------------------
def ex_post_metrics_generic(returns, rf=None, mode="weekly",
                            threshold="zero", periods_per_year=52, theta=0.95):
    """
    returns: 1D np.array
      - if threshold='rf': returns = TOTAL returns; subtract rf per period inside.
      - if threshold='zero': returns = EXCESS; threshold=0.
    rf: 1D np.array of risk-free (same length) when threshold='rf', else None
    """
    r = sanitize(returns)
    if threshold == "rf":
        if rf is None:
            raise ValueError("eval_threshold=rf 需要传入 rf 序列")
        exc = r - sanitize(rf)
    else:
        exc = r

    mu_w = exc.mean()
    std_w = exc.std(ddof=1)
    sr_w = safe_div(mu_w, std_w + EPS)

    dn = np.maximum(-exc, 0.0)
    sdn_w = math.sqrt((dn**2).mean() + EPS)
    sor_w = safe_div(mu_w, sdn_w)

    lpm1_w = dn.mean() + EPS
    omega_w = mu_w / lpm1_w + 1.0

    m = len(exc)
    losses = -exc
    q = int(math.ceil((1.0 - theta) * m))
    if q > 0:
        worst = np.partition(losses, -q)[-q:]
        cvar_w = worst.mean() + EPS
    else:
        cvar_w = max(losses.max(), 0.0) + EPS
    csr_w = mu_w / (cvar_w + EPS)

    # EVaR via log-sum-exp（与 ND 对齐的稳定实现）
    def evar_weekly(x, theta=0.95):
        x = sanitize(x)
        best = None
        for rho in np.logspace(-3, 0, 40):  # 1e-3 .. 1
            logits = -x / max(rho, 1e-6)
            mlog = np.max(logits)
            lse = mlog + math.log(np.exp(logits - mlog).sum() + EPS)
            logS = lse - math.log(m * (1.0 - theta) + EPS)
            val = rho * logS
            best = val if (best is None or val < best) else best
        return best if best is not None else 0.0
    evar_w = evar_weekly(exc, theta=theta) + EPS
    esr_w = mu_w / evar_w

    if mode == "weekly":
        return dict(SRp=sr_w, SoRp=sor_w, ORp=omega_w, CSRp=csr_w, ESRp=esr_w)

    # Annualize（Sharpe/Sortino 乘 sqrt(年化频率）；其余按照 ND 的“量级保持”做 sqrt 缩放）
    mu_a  = mu_w * periods_per_year
    std_a = std_w * math.sqrt(periods_per_year)
    sdn_a = sdn_w * math.sqrt(periods_per_year)
    lpm1_a = lpm1_w * math.sqrt(periods_per_year)
    cvar_a = cvar_w * math.sqrt(periods_per_year)
    evar_a = evar_w * math.sqrt(periods_per_year)

    sr_a  = safe_div(mu_a, std_a + EPS)
    sor_a = safe_div(mu_a, sdn_a + EPS)
    omega_a = mu_a / (lpm1_a + EPS) + 1.0
    csr_a = mu_a / (cvar_a + EPS)
    esr_a = mu_a / (evar_a + EPS)
    return dict(SRp=sr_a, SoRp=sor_a, ORp=omega_a, CSRp=csr_a, ESRp=esr_a)

# ------------------------
# 读数据（与 ND 对齐）
# ------------------------
def read_assets_index_from_excel(path, assets_sheet="Assets_Returns", index_sheet="Index_Returns"):
    df_assets = pd.read_excel(path, sheet_name=assets_sheet, engine="openpyxl")
    df_index  = pd.read_excel(path, sheet_name=index_sheet,  engine="openpyxl")
    assets    = df_assets.values.astype(np.float64)               # (T, N)
    index_ret = df_index.values.squeeze().astype(np.float64)      # (T,)
    return assets, index_ret

# ------------------------
# 基础：单纯形映射 + EWMA 协方差
# ------------------------
def project_to_simplex(v):
    v = np.maximum(sanitize(v), 0.0)
    s = v.sum()
    if s <= 1e-12: return None
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0]
    if rho.size == 0: return None
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

def ewma_cov(X, span=20, eps=1e-6):
    X = sanitize(X)
    lam = math.exp(-1.0 / max(span, 1))
    mu = np.zeros(X.shape[1], dtype=np.float64)
    cov = np.eye(X.shape[1], dtype=np.float64) * 1e-6
    for t in range(X.shape[0]):
        x = X[t]
        mu = lam * mu + (1 - lam) * x
        d = (x - mu)[:, None]
        cov = lam * cov + (1 - lam) * (d @ d.T)
    cov += np.eye(X.shape[1]) * eps
    return cov

# ------------------------
# 经典策略的“求权重器”（在历史窗 R_est 上）
# ------------------------
def mv_weight(R_hist, use_ewma=False, span=20, lam=10.0):
    mu = R_hist.mean(axis=0)
    if use_ewma:
        Sigma = ewma_cov(R_hist, span=span)
    else:
        Sigma = np.cov(R_hist.T, ddof=1) + 1e-6*np.eye(R_hist.shape[1])
    try:
        raw = np.linalg.pinv(lam * Sigma) @ mu
    except Exception:
        raw = mu.copy()
    w = project_to_simplex(raw)
    return w if w is not None else np.ones(R_hist.shape[1]) / R_hist.shape[1]

def projected_gd_simplex(grad_fn, w0, steps=300, lr=0.2):
    w = w0.copy()
    for _ in range(steps):
        g = grad_fn(w)
        w = w - lr * g
        w = project_to_simplex(w)
        if w is None:
            w = np.ones_like(w0) / w0.size
    return w

# —— 下列 SSD 族与原脚本一致，但**只用历史窗 R_hist**构造梯度；与 ND 同样只用“过去” —— #
def lssd_weight(R_hist, tau=0.0, steps=300, lr=0.5):
    T, N = R_hist.shape
    def grad_fn(w):
        r = R_hist @ w
        d_r = -2.0 * np.clip(tau - r, 0.0, None)     # d/d r of (tau - r)_+^2
        return (R_hist.T @ (d_r / T))
    w0 = np.ones(N) / N
    return projected_gd_simplex(grad_fn, w0, steps=steps, lr=lr)

def lr_assd_weight(R_hist, taus=None, steps=300, lr=0.2):
    T, N = R_hist.shape
    if taus is None:
        qs = np.linspace(0.1, 0.9, 9)
        taus = np.quantile(R_hist @ (np.ones(N)/N), qs)
    taus = np.asarray(taus, dtype=np.float64)
    def grad_fn(w):
        r = R_hist @ w; g = np.zeros(N)
        for tau in taus:
            d_r = - (r < tau).astype(np.float64) / T   # d/d r of (tau - r)_+
            g += R_hist.T @ d_r
        return g / len(taus)
    w0 = np.ones(N) / N
    return projected_gd_simplex(grad_fn, w0, steps=steps, lr=lr)

def rmz_ssd_weight(R_hist, taus=None, steps=300, lr=0.2, kappa=0.5):
    T, N = R_hist.shape
    if taus is None:
        qs = np.linspace(0.05, 0.95, 10)
        taus = np.quantile(R_hist @ (np.ones(N)/N), qs)
    taus = np.asarray(taus, dtype=np.float64)
    w_tau = np.exp(-kappa * np.abs(taus))
    def grad_fn(w):
        r = R_hist @ w; g = np.zeros(N)
        for wt, tau in zip(w_tau, taus):
            d_r = -2.0 * np.clip(tau - r, 0.0, None)  # (tau - r)_+^2
            g += wt * (R_hist.T @ (d_r / T))
        return g / len(taus)
    w0 = np.ones(N) / N
    return projected_gd_simplex(grad_fn, w0, steps=steps, lr=lr)

def pk_ssd_weight(R_hist, taus=None, steps=300, lr=0.2, p=2.0):
    T, N = R_hist.shape
    if taus is None:
        qs = np.linspace(0.05, 0.95, 10)
        taus = np.quantile(R_hist @ (np.ones(N)/N), qs)
    taus = np.asarray(taus, dtype=np.float64)
    tau_w = np.abs(taus) ** p
    def grad_fn(w):
        r = R_hist @ w; g = np.zeros(N)
        for wt, tau in zip(tau_w, taus):
            d_r = - (r < tau).astype(np.float64) / T  # (tau - r)_+
            g += wt * (R_hist.T @ d_r)
        return g / len(taus)
    w0 = np.ones(N) / N
    return projected_gd_simplex(grad_fn, w0, steps=steps, lr=lr)

def czesd_weight(R_hist, taus=None, steps=300, lr=0.2, zcap=2.0):
    T, N = R_hist.shape
    if taus is None:
        qs = np.linspace(0.05, 0.95, 10)
        taus = np.quantile(R_hist @ (np.ones(N)/N), qs)
    taus = np.asarray(taus, dtype=np.float64)
    def grad_fn(w):
        r = R_hist @ w; g = np.zeros(N)
        for tau in taus:
            short = np.clip(tau - r, 0.0, None)     # (tau - r)_+
            active = (short > 0).astype(np.float64) * (short < zcap)
            d_r = - active / T
            g += R_hist.T @ d_r
        return g / len(taus)
    w0 = np.ones(N) / N
    return projected_gd_simplex(grad_fn, w0, steps=steps, lr=lr)

# ------------------------
# 主流程（完全对齐 ND 的实验设置）
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    # 数据与切分
    ap.add_argument("--excel_path", type=str, required=True)
    ap.add_argument("--assets_sheet", type=str, default="Assets_Returns")
    ap.add_argument("--index_sheet", type=str, default="Index_Returns")
    ap.add_argument("--tbill_path", type=str, required=True)
    ap.add_argument("--yearly_tbill", action="store_true",
                    help="若 txt 中是年化 T-bill，则转换为周频 (1+r)^(1/52)-1")
    ap.add_argument("--drop_last_if_odd", action="store_true",
                    help="奇数总周数时丢最后一周，使前后半段等长")
    ap.add_argument("--rebalance", type=int, default=1, help="再平衡周期（周）")

    # 经典策略开关
    ap.add_argument("--run_mv", action="store_true")
    ap.add_argument("--run_lssd", action="store_true")
    ap.add_argument("--run_lr_assd", action="store_true")
    ap.add_argument("--run_rmz_ssd", action="store_true")
    ap.add_argument("--run_pk_ssd", action="store_true")
    ap.add_argument("--run_czesd", action="store_true")

    # MV/SSD 参数
    ap.add_argument("--mv_use_ewma", action="store_true")
    ap.add_argument("--cov_span", type=int, default=20)
    ap.add_argument("--mv_lambda", type=float, default=10.0)
    ap.add_argument("--ssd_steps", type=int, default=300)
    ap.add_argument("--ssd_lr", type=float, default=0.2)
    ap.add_argument("--lssd_tau", type=float, default=0.0)
    ap.add_argument("--rmz_kappa", type=float, default=0.5)
    ap.add_argument("--pk_p", type=float, default=2.0)
    ap.add_argument("--cz_zcap", type=float, default=2.0)

    # 评估设置（对齐 ND）
    ap.add_argument("--eval_mode", choices=["weekly","annualized"], default="weekly")
    ap.add_argument("--eval_sample", choices=["oos","full"], default="oos")
    ap.add_argument("--eval_threshold", choices=["zero","rf"], default="zero")
    ap.add_argument("--theta", type=float, default=0.95)

    # 保存
    ap.add_argument("--save_weights_dir", default=None)
    ap.add_argument("--save_excess_csv", default=None)
    ap.add_argument("--save_metrics_csv", default=None)

    args = ap.parse_args()

    # —— 读入数据（与 ND 对齐）——
    assets, index_ret = read_assets_index_from_excel(args.excel_path, args.assets_sheet, args.index_sheet)

    with open(args.tbill_path, "r", encoding="utf-8") as f:
        rf_vals = np.array([float(x) for x in re.split(r"[,\s]+", f.read().strip()) if x != ""], dtype=np.float64)

    if args.yearly_tbill:
        rf_vals = (1.0 + rf_vals) ** (1.0 / WEEKS_PER_YEAR) - 1.0

    T_assets, N = assets.shape
    T_index = index_ret.shape[0]
    T_rf    = rf_vals.shape[0]
    T_all   = min(T_assets, T_index, T_rf)
    if (T_assets != T_all) or (T_index != T_all) or (T_rf != T_all):
        print(f"[WARN] Length mismatch: assets={T_assets}, index={T_index}, tbill={T_rf}. Truncate to {T_all}.")
        assets    = assets[:T_all, :]
        index_ret = index_ret[:T_all]
        rf_vals   = rf_vals[:T_all]

    if args.drop_last_if_odd and (T_all % 2 == 1):
        print("[INFO] Odd number of weeks detected; drop last week to make even.")
        assets    = assets[:-1, :]
        index_ret = index_ret[:-1]
        rf_vals   = rf_vals[:-1]

    T = assets.shape[0]
    T_half = T // 2
    seq_train = assets[:T_half, :]
    seq_test  = assets[T_half:, :]
    rf_test   = rf_vals[T_half:]
    mi_test   = index_ret[T_half:]

    # —— 基准 EW/MI（与 ND 口径一致）——
    if args.eval_sample == "full":
        ew_base = assets.mean(axis=1)
        mi_base = index_ret
        rf_eval = rf_vals
    else:
        ew_base = seq_test.mean(axis=1)
        mi_base = mi_test
        rf_eval = rf_test

    if args.eval_threshold == "rf":
        ew_metrics = ex_post_metrics_generic(ew_base, rf=rf_eval,
                                             mode=args.eval_mode, threshold="rf",
                                             theta=args.theta, periods_per_year=WEEKS_PER_YEAR)
        mi_metrics = ex_post_metrics_generic(mi_base, rf=rf_eval,
                                             mode=args.eval_mode, threshold="rf",
                                             theta=args.theta, periods_per_year=WEEKS_PER_YEAR)
    else:
        ew_metrics = ex_post_metrics_generic(ew_base - rf_eval, rf=None,
                                             mode=args.eval_mode, threshold="zero",
                                             theta=args.theta, periods_per_year=WEEKS_PER_YEAR)
        mi_metrics = ex_post_metrics_generic(mi_base - rf_eval, rf=None,
                                             mode=args.eval_mode, threshold="zero",
                                             theta=args.theta, periods_per_year=WEEKS_PER_YEAR)

    print(f"T(all)={T}, N(assets)={N}, OOS weeks={len(seq_test)}")
    print(f"\n== EW ({args.eval_sample}, {args.eval_mode}, thr={args.eval_threshold}) 指标 ==")
    for k, v in ew_metrics.items(): print(f"  {k}: {v:.6f}")
    print(f"\n== MI ({args.eval_sample}, {args.eval_mode}, thr={args.eval_threshold}) 指标 ==")
    for k, v in mi_metrics.items(): print(f"  {k}: {v:.6f}")

    # —— 经典策略：滚动 OOS，与 ND 的“构造权重→下一周实现收益”一致 —— #
    strategies = []
    if args.run_mv:        strategies.append("MV")
    if args.run_lssd:      strategies.append("L-SSD")
    if args.run_lr_assd:   strategies.append("LR-ASSD")
    if args.run_rmz_ssd:   strategies.append("RMZ-SSD")
    if args.run_pk_ssd:    strategies.append("PK-SSD")
    if args.run_czesd:     strategies.append("CZeSD")

    weights_hist = {s: [] for s in strategies}
    excess_hist  = {s: [] for s in strategies}

    def solve_one(R_est, name):
        if name == "MV":
            return mv_weight(R_est, use_ewma=args.mv_use_ewma, span=args.cov_span, lam=args.mv_lambda)
        elif name == "L-SSD":
            return lssd_weight(R_est, tau=args.lssd_tau, steps=args.ssd_steps, lr=args.ssd_lr)
        elif name == "LR-ASSD":
            return lr_assd_weight(R_est, taus=None, steps=args.ssd_steps, lr=args.ssd_lr)
        elif name == "RMZ-SSD":
            return rmz_ssd_weight(R_est, taus=None, steps=args.ssd_steps, lr=args.ssd_lr, kappa=args.rmz_kappa)
        elif name == "PK-SSD":
            return pk_ssd_weight(R_est, taus=None, steps=args.ssd_steps, lr=args.ssd_lr, p=args.pk_p)
        elif name == "CZeSD":
            return czesd_weight(R_est, taus=None, steps=args.ssd_steps, lr=args.ssd_lr, zcap=args.cz_zcap)
        else:
            raise ValueError(name)

    # 初始化为等权
    x_state = {s: np.ones(N)/N for s in strategies}

    for i in range(T - T_half):
        t_global = T_half + i
        R_est = assets[:t_global, :]  # 仅用到 t_global-1 的历史
        rf_t  = rf_vals[t_global]

        if (i % args.rebalance) == 0:
            for s in strategies:
                x_state[s] = solve_one(R_est, s)

        r_next = assets[t_global, :]   # 当前周真实收益
        for s in strategies:
            weights_hist[s].append(x_state[s].copy())
            excess_hist[s].append(float(r_next @ x_state[s]) - rf_t)

    for s in strategies:
        weights_hist[s] = np.array(weights_hist[s])
        excess_hist[s]  = np.array(excess_hist[s])

    # —— 事后指标（OOS）（与 ND 完全相同口径）——
    if strategies:
        print(f"\n== Classics (OOS, {args.eval_mode}, thr={args.eval_threshold}) 指标 ==")
    classics_summary = {}
    for s in strategies:
        if args.eval_threshold == "rf":
            r_total = excess_hist[s] + rf_test
            classics_summary[s] = ex_post_metrics_generic(r_total, rf=rf_test,
                                    mode=args.eval_mode, threshold="rf",
                                    theta=args.theta, periods_per_year=WEEKS_PER_YEAR)
        else:
            classics_summary[s] = ex_post_metrics_generic(excess_hist[s], rf=None,
                                    mode=args.eval_mode, threshold="zero",
                                    theta=args.theta, periods_per_year=WEEKS_PER_YEAR)
        print(f"--- {s} ---")
        for k, v in classics_summary[s].items():
            print(f"  {k}: {v:.6f}")

    # —— 保存（字段布局与 ND 对齐）——
    if args.save_weights_dir and strategies:
        os.makedirs(args.save_weights_dir, exist_ok=True)
        asset_names = np.array([f"A{i}" for i in range(N)])
        for s in strategies:
            np.savez(os.path.join(args.save_weights_dir, f"weights_{s}.npz"),
                     W=weights_hist[s], assets=asset_names)
        print(f"\nSaved weights to: {args.save_weights_dir}")

    if args.save_excess_csv and strategies:
        out = {
            "EW_excess": seq_test.mean(axis=1) - rf_test,
            "MI_excess": mi_test - rf_test,
        }
        for s in strategies:
            out[f"{s}_excess"] = excess_hist[s]
        pd.DataFrame(out).to_csv(args.save_excess_csv, index=False)
        print(f"Saved excess to: {args.save_excess_csv}")

    if args.save_metrics_csv:
        rows = []
        # 基准（与 ND 相同）
        rows.append({"Strategy": "EW", "Sample": args.eval_sample, "Mode": args.eval_mode,
                     "Threshold": args.eval_threshold, **ew_metrics})
        rows.append({"Strategy": "MI", "Sample": args.eval_sample, "Mode": args.eval_mode,
                     "Threshold": args.eval_threshold, **mi_metrics})
        # 经典策略
        for s in strategies:
            rows.append({"Strategy": s, "Sample": "oos", "Mode": args.eval_mode,
                         "Threshold": args.eval_threshold, **classics_summary[s]})
        pd.DataFrame(rows).to_csv(args.save_metrics_csv, index=False)
        print(f"Saved metrics to: {args.save_metrics_csv}")

if __name__ == "__main__":
    main()
