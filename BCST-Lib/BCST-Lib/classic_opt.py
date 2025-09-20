# baselines_bcst_annualized.py
# -*- coding: utf-8 -*-
"""
Traditional (non-DL) BCST-style baselines with annualized ex-post metrics:
- EW, MI, M–V, L-SSD, LR-ASSD, RMZ-SSD, PK-SSD, CZeSD
Rolling scheme: 52-week in-sample, 12-week hold, rebalance every 12 weeks.
Data: headerless Excel (asset_return sheet: rows=weeks, cols=assets; index_return sheet: 1 col),
      headerless TXT risk-free (comma or whitespace separated).

Outputs per method:
- Out-of-sample weekly portfolio gross returns (CSV)
- Rebalance weights (CSV; MI has no weights)
- Printed ex-post gross *annualized* metrics: SRp, SoRp, ORp, CSRp, ESRp

Annualization:
- SRp/SoRp/CSRp/ESRp: weekly ratio × sqrt(annualize_factor) (default 52).
- ORp: 'block' annualization compounds weekly returns and rf into annual blocks,
        then Omega is computed on those annual blocks. Use --omega_ann none to skip annualization.
"""

import argparse, os, math, json
import numpy as np
import pandas as pd
import cvxpy as cp

# ===================== I/O: headerless loaders =====================
def load_excel_assets_index(excel_path, asset_sheet="asset_return", index_sheet="index_return"):
    assets = pd.read_excel(excel_path, sheet_name=asset_sheet, header=None).apply(pd.to_numeric, errors="coerce")
    index  = pd.read_excel(excel_path, sheet_name=index_sheet, header=None).apply(pd.to_numeric, errors="coerce")
    assets = assets.dropna(how="all"); index = index.dropna(how="all")
    T = min(len(assets), len(index))
    assets = assets.iloc[:T, :].copy()
    index_sr = index.iloc[:T, 0].astype(float).reset_index(drop=True)
    assets.columns = [f"asset_{i+1}" for i in range(assets.shape[1])]
    assets.index = np.arange(1, len(assets)+1)
    index_sr.index = np.arange(1, len(index_sr)+1)
    return assets, index_sr

def load_txt_rf(txt_path):
    try:
        df = pd.read_csv(txt_path, header=None, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(txt_path, header=None)
    if df.shape[1] == 1 and len(df) > 1:
        rf = df.iloc[:,0].astype(float).reset_index(drop=True)
    else:
        vals = df.values.flatten()
        rf = pd.Series([float(v) for v in vals if pd.notna(v)])
    rf.index = np.arange(1, len(rf)+1)
    return rf

# ===================== Metrics (weekly → annualized) =====================
def sharpe_weekly(excess):
    if excess.size == 0: return np.nan
    return np.mean(excess) / (np.std(excess, ddof=0) + 1e-12)

def sortino_weekly(port_r, rf):
    if port_r.size == 0: return np.nan
    excess = port_r - rf
    downside = np.maximum(rf - port_r, 0.0)
    denom = math.sqrt(np.mean(downside**2) + 1e-12)
    return np.mean(excess) / denom

def omega_weekly(port_r, rf):
    if port_r.size == 0: return np.nan
    excess = port_r - rf
    lpm1 = np.mean(np.maximum(rf - port_r, 0.0))
    return np.mean(excess) / (lpm1 + 1e-12) + 1.0

def cvar_loss(values, alpha=0.95):
    if values.size == 0: return np.nan
    q = np.quantile(values, alpha)
    tail = values[values >= q]
    return float(np.mean(tail)) if tail.size > 0 else float(q)

def evar_loss(values, alpha=0.95):
    # EVaR_alpha = min_{rho>0} rho * log( (1/((1-alpha)*m)) * sum exp(loss_j / rho) )
    vals = np.asarray(values, dtype=float)
    m = vals.size
    if m == 0: return np.nan
    def obj(rho):
        z = vals / rho
        zmax = np.max(z)
        lse = zmax + np.log(np.mean(np.exp(z - zmax)))
        return rho * (lse - math.log(1 - alpha))
    rhos = np.logspace(-4, 0, 200)
    f = np.array([obj(r) for r in rhos]); i = int(np.argmin(f))
    lo = rhos[max(0, i-1)]; hi = rhos[min(len(rhos)-1, i+1)]
    for _ in range(30):
        g = (math.sqrt(5) - 1) / 2
        x1 = hi - g*(hi - lo); x2 = lo + g*(hi - lo)
        f1, f2 = obj(x1), obj(x2)
        if f1 < f2: hi = x2
        else:       lo = x1
    rho_star = 0.5*(lo + hi)
    return float(obj(rho_star))

def csr_weekly(port_r, rf, alpha=0.95):
    excess = port_r - rf
    loss = -port_r
    cvar = cvar_loss(loss, alpha=alpha)
    return np.mean(excess) / (cvar + 1e-12)

def esr_weekly(port_r, rf, alpha=0.95):
    excess = port_r - rf
    loss = -port_r
    evar = evar_loss(loss, alpha=alpha)
    return np.mean(excess) / (evar + 1e-12)

def block_compound(r, block):
    m = len(r) // block
    if m <= 0: return None
    arr = r[:m*block].reshape(m, block)
    return np.prod(1.0 + arr, axis=1) - 1.0

def summarize_annualized(port_r, idx_r, rf, annualize_factor=52, theta=0.95, omega_ann="block"):
    mask = ~np.isnan(port_r)
    r = port_r[mask]; ri = idx_r[mask]; rfv = rf[mask]
    # weekly metrics
    SRp_w  = sharpe_weekly(r - rfv)
    SoRp_w = sortino_weekly(r, rfv)
    CSRp_w = csr_weekly(r, rfv, alpha=theta)
    ESRp_w = esr_weekly(r, rfv, alpha=theta)
    # annualize ratios via sqrt(T)
    scale = math.sqrt(annualize_factor)
    SRp  = SRp_w * scale
    SoRp = SoRp_w * scale
    CSRp = CSRp_w * scale
    ESRp = ESRp_w * scale
    # Omega: block annualization or none
    if omega_ann == "block":
        r_blk  = block_compound(r,  annualize_factor)
        rf_blk = block_compound(rfv, annualize_factor)
        if r_blk is not None and rf_blk is not None and len(r_blk)==len(rf_blk) and len(r_blk)>0:
            ORp = omega_weekly(r_blk, rf_blk)  # same formula on annual blocks
        else:
            ORp = omega_weekly(r, rfv)  # fallback: weekly Omega
    else:
        ORp = omega_weekly(r, rfv)      # no annualization
    return dict(SRp=SRp, SoRp=SoRp, ORp=ORp, CSRp=CSRp, ESRp=ESRp)

# ===================== Covariance helpers (for M–V) =====================
def sample_cov(R):
    return np.cov(R, rowvar=False)

def ewma_cov(R, halflife=26):
    X = R - R.mean(axis=0, keepdims=True)
    alpha = 1 - 0.5**(1/halflife)
    n = X.shape[1]
    S = np.zeros((n, n)); w = 0.0
    for t in range(X.shape[0]):
        x = X[t:t+1].T
        S = (1 - alpha)*S + alpha*(x @ x.T)
        w = (1 - alpha)*w + alpha
    if w > 1e-12: S = S / w
    return S

def shrink_diag(S, shrink=0.0):
    if shrink <= 0.0: return S
    D = np.diag(np.diag(S))
    return (1 - shrink)*S + shrink*D

# ===================== z-grid / shortfall for SSD =====================
def z_grid_from_index(idx_in, kind="quantile", K=21):
    idx = np.asarray(idx_in, dtype=float)
    if kind == "unique":
        z = np.unique(np.round(idx, 10))
        return np.sort(z)
    qs = np.linspace(0.0, 1.0, K)
    return np.quantile(idx, qs)

def benchmark_shortfall(idx_in, z_list):
    I = np.asarray(idx_in, dtype=float).reshape(-1,1)  # m x 1
    Z = np.asarray(z_list, dtype=float).reshape(1,-1)  # 1 x K
    short = np.maximum(Z - I, 0.0)                    # m x K
    return short.mean(axis=0)                         # K,

# ===================== Solvers per method =====================
def simplex_constraints(n, weight_cap=None):
    x = cp.Variable(n, nonneg=True)
    cons = [cp.sum(x) == 1]
    if weight_cap is not None and weight_cap > 0:
        cons += [x <= weight_cap]
    return x, cons

def solve_EW(R, weight_cap=None):
    n = R.shape[1]
    w = np.ones(n)/n
    if weight_cap is not None and weight_cap > 0:
        w = np.minimum(w, weight_cap)
        w = w / w.sum()
    return w

def solve_MV(R, lam=8.0, cov_method="ewma", cov_halflife=26, shrink=0.2, weight_cap=None):
    mu = np.mean(R, axis=0)
    if cov_method == "ewma":
        S = ewma_cov(R, halflife=cov_halflife)
    else:
        S = sample_cov(R)
    S = 0.5*(S + S.T)
    S = shrink_diag(S, shrink)
    n = len(mu)
    x, cons = simplex_constraints(n, weight_cap)
    # Minimize lam x' S x - mu' x  (convex)
    obj = cp.Minimize(lam * cp.quad_form(x, S) - mu @ x)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    val = x.value if x.value is not None else np.ones(n)/n
    return np.maximum(val, 0)/max(np.sum(val), 1e-12)

def _ssd_constraints(R, x, z_list, bench_b):
    m = R.shape[0]; K = len(z_list)
    U = cp.Variable((m, K), nonneg=True)
    cons = []
    Rxc = R @ x
    for k, zk in enumerate(z_list):
        cons += [U[:,k] >= zk - Rxc,
                 cp.sum(U[:,k])/m <= bench_b[k]]
    return U, cons

def solve_RMZ_SSD(R, I, Kz=41, weight_cap=None):
    m, n = R.shape
    x, cons = simplex_constraints(n, weight_cap)
    z_list = z_grid_from_index(I, "quantile", K=Kz)
    b = benchmark_shortfall(I, z_list)
    U, ssd_cons = _ssd_constraints(R, x, z_list, b)
    cons += ssd_cons
    obj = cp.Maximize(cp.sum(R @ x - I)/m)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    val = x.value if x.value is not None else np.ones(n)/n
    return np.maximum(val, 0)/max(np.sum(val), 1e-12)

def solve_L_SSD(R, I, Kz=9, weight_cap=None):
    m, n = R.shape
    x, cons = simplex_constraints(n, weight_cap)
    z_list = z_grid_from_index(I, "quantile", K=Kz)
    b = benchmark_shortfall(I, z_list)
    U, ssd_cons = _ssd_constraints(R, x, z_list, b)
    cons += ssd_cons
    obj = cp.Maximize(cp.sum(R @ x - I)/m)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    val = x.value if x.value is not None else np.ones(n)/n
    return np.maximum(val, 0)/max(np.sum(val), 1e-12)

def solve_LR_ASSD(R, I, Kz=21, gamma_slack=100.0, weight_cap=None):
    m, n = R.shape
    x, cons = simplex_constraints(n, weight_cap)
    z_list = z_grid_from_index(I, "quantile", K=Kz)
    b = benchmark_shortfall(I, z_list)
    U = cp.Variable((m, Kz), nonneg=True)
    s = cp.Variable(Kz, nonneg=True)
    Rxc = R @ x
    for k, zk in enumerate(z_list):
        cons += [U[:,k] >= zk - Rxc,
                 cp.sum(U[:,k])/m <= b[k] + s[k]]
    obj = cp.Maximize(cp.sum(Rxc - I)/m - gamma_slack*cp.sum(s))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    val = x.value if x.value is not None else np.ones(n)/n
    return np.maximum(val, 0)/max(np.sum(val), 1e-12)

def solve_PK_SSD(R, I, Kz=21, gamma=3.0, weight_cap=None):
    # Power-weighted shortfall minimization (Post–Kopa spirit)
    m, n = R.shape
    x, cons = simplex_constraints(n, weight_cap)
    z_list = z_grid_from_index(I, "quantile", K=Kz)
    b = benchmark_shortfall(I, z_list)
    U = cp.Variable((m, Kz), nonneg=True)
    Rxc = R @ x
    for k, zk in enumerate(z_list):
        cons += [U[:,k] >= zk - Rxc]
    ks = np.arange(1, Kz+1)
    w = (ks / Kz) ** (gamma - 1.0)
    w = w / np.sum(w)
    meanU = cp.sum(U, axis=0) / m
    obj = cp.Minimize(cp.sum(cp.multiply(w, meanU - b)))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    val = x.value if x.value is not None else np.ones(n)/n
    return np.maximum(val, 0)/max(np.sum(val), 1e-12)

def solve_CZeSD(R, I, weight_cap=None):
    # Bruni et al. (2017): LP minimizing cumulative zero-order shortfall
    m, n = R.shape
    x, cons = simplex_constraints(n, weight_cap)
    y = cp.Variable(m, nonneg=True)
    delta = R @ x - I
    cons += [y + delta >= 0]           # y_t >= -delta_t
    obj = cp.Minimize(cp.sum(y))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    val = x.value if x.value is not None else np.ones(n)/n
    return np.maximum(val, 0)/max(np.sum(val), 1e-12)

# ===================== Rolling backtest =====================
def rolling_points(T, L=52, H=12):
    return list(range(L, T - H + 1, H))

def alias_method(name):
    n = name.strip()
    if n in ["MV", "M-V", "M–V"]: return "M–V"
    return n

def backtest_method(A_df, idx_sr, rf_sr, method_name,
                    L=52, H=12,
                    mv_lambda=8.0, cov_method="ewma", cov_halflife=26, shrink=0.2,
                    zK_dense=41, zK_mid=21, zK_coarse=9,
                    gamma_slack=100.0, pk_gamma=3.0,
                    weight_cap=None):
    A = A_df.values
    I = idx_sr.values
    rf_full = rf_sr.values[:len(I)]
    T, N = A.shape

    port = np.full(T, np.nan)
    Wlist, Tlist = [], []

    for t in rolling_points(T, L, H):
        ins_R = A[t-L:t, :]
        ins_I = I[t-L:t]
        fut_R = A[t:t+H, :]
        fut_I = I[t:t+H]

        mth = alias_method(method_name)
        if mth == "EW":
            w = solve_EW(ins_R, weight_cap)
        elif mth == "MI":
            w = None
        elif mth == "M–V":
            w = solve_MV(ins_R, lam=mv_lambda, cov_method=cov_method,
                         cov_halflife=cov_halflife, shrink=shrink, weight_cap=weight_cap)
        elif mth == "L-SSD":
            w = solve_L_SSD(ins_R, ins_I, Kz=zK_coarse, weight_cap=weight_cap)
        elif mth == "LR-ASSD":
            w = solve_LR_ASSD(ins_R, ins_I, Kz=zK_mid, gamma_slack=gamma_slack, weight_cap=weight_cap)
        elif mth == "RMZ-SSD":
            w = solve_RMZ_SSD(ins_R, ins_I, Kz=zK_dense, weight_cap=weight_cap)
        elif mth == "PK-SSD":
            w = solve_PK_SSD(ins_R, ins_I, Kz=zK_mid, gamma=pk_gamma, weight_cap=weight_cap)
        elif mth == "CZeSD":
            w = solve_CZeSD(ins_R, ins_I, weight_cap=weight_cap)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Out-of-sample weekly gross returns
        if mth == "MI":
            pr = fut_I.copy()
        else:
            pr = fut_R @ w

        idx = np.arange(t, t+H) + 1
        port[idx-1] = pr
        if mth != "MI":
            Wlist.append(w); Tlist.append(t)

    # Annualized metrics (gross)
    mets = summarize_annualized(port, idx_sr.values, rf_full,
                                annualize_factor=args.annualize_factor,
                                theta=args.theta,
                                omega_ann=args.omega_ann)

    # Collect weights
    Wdf = None if alias_method(method_name) == "MI" else pd.DataFrame(Wlist, index=Tlist, columns=A_df.columns)
    return dict(returns=pd.Series(port, index=A_df.index, name=alias_method(method_name)),
                weights=Wdf, metrics=mets)

# ===================== CLI =====================
def build_parser():
    p = argparse.ArgumentParser("BCST-style baselines (annualized ex-post SRp/SoRp/ORp/CSRp/ESRp)")
    # data
    p.add_argument("--excel_path", type=str, required=True)
    p.add_argument("--txt_path", type=str, required=True)
    p.add_argument("--asset_sheet", type=str, default="asset_return")
    p.add_argument("--index_sheet", type=str, default="index_return")
    # methods (use exact abbreviations as requested)
    p.add_argument("--methods", nargs="+",
                   default=["EW","MI","M–V","L-SSD","LR-ASSD","RMZ-SSD","PK-SSD","CZeSD"])
    # rolling scheme
    p.add_argument("--lookback_L", type=int, default=52)
    p.add_argument("--hold_H", type=int, default=12)
    # M–V params (align with previous settings)
    p.add_argument("--mv_lambda", type=float, default=8.0)
    p.add_argument("--cov_method", type=str, default="ewma", choices=["ewma","sample"])
    p.add_argument("--cov_halflife", type=int, default=26)
    p.add_argument("--shrink", type=float, default=0.2)
    # SSD params
    p.add_argument("--zK_dense", type=int, default=41)
    p.add_argument("--zK_mid", type=int, default=21)
    p.add_argument("--zK_coarse", type=int, default=9)
    p.add_argument("--gamma_slack", type=float, default=100.0)
    p.add_argument("--pk_gamma", type=float, default=3.0)
    # constraints
    p.add_argument("--weight_cap", type=float, default=0.2, help="Per-asset upper bound; <=0 disables cap")
    # ex-post/annualization
    p.add_argument("--annualize_factor", type=int, default=52, help="Weeks per year for annualization")
    p.add_argument("--theta", type=float, default=0.95, help="CVaR/EVaR confidence level for CSRp/ESRp")
    p.add_argument("--omega_ann", type=str, default="block", choices=["block","none"],
                   help="How to annualize Omega: 'block' compounding (default) or 'none'")
    # output
    p.add_argument("--out_dir", type=str, default="results_baselines")
    return p

def save_results(out_dir, name, res):
    os.makedirs(out_dir, exist_ok=True)
    ret_path = os.path.join(out_dir, f"oos_returns_{name}.csv")
    res["returns"].to_csv(ret_path)
    if res["weights"] is not None:
        w_path = os.path.join(out_dir, f"weights_{name}.csv")
        res["weights"].to_csv(w_path)
    met_path = os.path.join(out_dir, f"metrics_{name}.json")
    with open(met_path, "w") as f:
        json.dump(res["metrics"], f, indent=2)
    print(f"Saved: {ret_path}" + ("" if res["weights"] is None else f", {w_path}"))
    print(f"Metrics JSON: {met_path}")
    print("Annualized (gross) ex-post metrics:")
    for k,v in res["metrics"].items():
        print(f"  {k:>4}: {v:.6f}")

if __name__ == "__main__":
    args = build_parser().parse_args()

    A_df, idx_sr = load_excel_assets_index(args.excel_path, args.asset_sheet, args.index_sheet)
    rf_sr = load_txt_rf(args.txt_path)

    # align and drop all-NaN weeks
    T0 = min(len(A_df), len(idx_sr), len(rf_sr))
    A_df = A_df.iloc[:T0,:].copy(); idx_sr = idx_sr.iloc[:T0].copy(); rf_sr = rf_sr.iloc[:T0].copy()
    valid = ~A_df.isna().all(axis=1)
    A_df = A_df.loc[valid].copy(); idx_sr = idx_sr.loc[valid].copy(); rf_sr = rf_sr.loc[valid].copy()

    for m in args.methods:
        res = backtest_method(
            A_df, idx_sr, rf_sr, m,
            L=args.lookback_L, H=args.hold_H,
            mv_lambda=args.mv_lambda, cov_method=args.cov_method,
            cov_halflife=args.cov_halflife, shrink=args.shrink,
            zK_dense=args.zK_dense, zK_mid=args.zK_mid, zK_coarse=args.zK_coarse,
            gamma_slack=args.gamma_slack, pk_gamma=args.pk_gamma,
            weight_cap=(None if args.weight_cap is None or args.weight_cap <= 0 else args.weight_cap)
        )
        save_results(args.out_dir, alias_method(m), res)
