# run_baselines.py
import argparse
import numpy as np
import pandas as pd
from baseline_optimizers import (meanvar_max_utility, meanvar_min_var_given_ret,
                                 L_SSD, KP_SSD, CZ_eSD, reference_series,
                                 sharpe_ratio, sortino_ratio, annualized_return)

def load_excel_returns(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    num_df = df.select_dtypes(include=[float, int])
    R = num_df.values.astype(float)
    if R.ndim == 1:
        R = R[:, None]
    return R  # [T,N]

def ewma_cov(R, lam=0.94):
    r = R - R.mean(0, keepdims=True)
    n = r.shape[1]
    S, w, denom = np.zeros((n,n)), 1.0, 0.0
    for t in range(r.shape[0]-1, -1, -1):
        S += w * np.outer(r[t], r[t])
        denom += w
        w *= lam
    S /= max(denom, 1e-12)
    S = (S + S.T) / 2
    # PSD 修正
    d, U = np.linalg.eigh(S)
    d[d < 1e-10] = 1e-10
    return (U * d) @ U.T

def backtest(R, w, start=0):
    r = (R[start:] @ w).ravel()
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default="0")
    ap.add_argument("--rf", type=float, default=0.0)
    ap.add_argument("--mv_gamma", type=float, default=5.0)
    ap.add_argument("--ssd_ref", default="EW")  # "EW" / "asset:k" / "weights"
    ap.add_argument("--ssd_q", type=int, default=60)
    args = ap.parse_args()

    try:
        sheet = int(args.sheet)
    except:
        sheet = args.sheet

    R = load_excel_returns(args.excel, sheet=sheet)  # [T,N]
    T, N = R.shape
    print(f"Loaded: T={T}, N={N}")

    # 划分一个“样本内/样本外”示例（前 60% 估参，后 40% 回测）
    split = int(0.6 * T)
    R_in, R_out = R[:split], R[split:]

    mu = R_in.mean(0)
    V = ewma_cov(R_in)

    # 1) MeanVar（效用最大化）
    w_mv = meanvar_max_utility(mu, V, gamma=args.mv_gamma, allow_short=False)
    r_mv = backtest(R_out, w_mv, start=0)

    # 2) L-SSD（严格二阶占优）
    w_lssd = L_SSD(R_in, ref=args.ssd_ref, q=args.ssd_q, allow_short=False)
    r_lssd = backtest(R_out, w_lssd, start=0)

    # 3) KP-SSD（同框架近似实现）
    w_kpssd = KP_SSD(R_in, ref=args.ssd_ref, q=args.ssd_q, allow_short=False)
    r_kpssd = backtest(R_out, w_kpssd, start=0)

    # 4) CZ-εSD（加入 ε 松弛）
    w_czesd = CZ_eSD(R_in, ref=args.ssd_ref, q=args.ssd_q, allow_short=False, epsilon=0.001)
    r_czesd = backtest(R_out, w_czesd, start=0)

    # 输出指标（样本外）
    def show(name, r):
        print(f"{name:8s}  SR={sharpe_ratio(r, rf=args.rf):.4f}  SoR={sortino_ratio(r, rb=args.rf):.4f}  AnnRet≈{annualized_return(r):.2%}")

    print("\n=== Out-of-sample performance (approx.) ===")
    show("MeanVar", r_mv)
    show("L-SSD", r_lssd)
    show("KP-SSD", r_kpssd)
    show("CZ-εSD", r_czesd)

    # 若你有“市场指数/等权”的对照：也可直接算
    r_ref = reference_series(R_out, ref="EW")
    show("EW (ref)", r_ref)

    # 打印前几个权重
    def headw(name, w):
        print(f"{name:8s}  w[:min(10,N)]= {np.round(w[:min(10,N)], 4)}  sum={w.sum():.4f}")
    headw("MeanVar", w_mv)
    headw("L-SSD", w_lssd)
    headw("KP-SSD", w_kpssd)
    headw("CZ-εSD", w_czesd)

if __name__ == "__main__":
    main()
