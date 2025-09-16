# op_sr_min.py
# 目标：在与 blosam_sr_min.py 相同的设置下，复现“论文式 OP（优化组合）”
# 只计算 SR，对比 OP / EW / MI；支持遗忘因子 δ（in-sample 网格搜索）

import argparse
import numpy as np
import pandas as pd
import cvxpy as cp


# ===============================
# IO：读取两个 sheet 并对齐
# ===============================
def load_sheet(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    num = df.select_dtypes(include=[float, int])
    R = num.to_numpy(dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    return R

def align_two(R_assets, r_index):
    T = min(R_assets.shape[0], r_index.shape[0])
    R_assets, r_index = R_assets[:T], r_index[:T]
    mask = np.isfinite(R_assets).all(axis=1) & np.isfinite(r_index).ravel()
    return R_assets[mask], r_index[mask].ravel()


# ===============================
# 指标：Sharpe（可年化）
# ===============================
def sharpe_ratio_np(r, rf=0.0, per_year=52, annualize=False):
    r = np.asarray(r, dtype=float)
    ex = r.mean() - rf
    sd = r.std() + 1e-12
    sr = ex / sd
    return sr * (np.sqrt(per_year) if annualize else 1.0)


# ===============================
# 指数加权的矩（δ 遗忘因子）
# ===============================
def exp_weights(m, delta):
    """长度 m 的权重，最近一期权重大：w_t ∝ delta^(m-1-t)"""
    idx = np.arange(m, dtype=float)
    w = delta ** (m - 1 - idx)
    w /= w.sum()
    return w

def ew_mean_cov(X, delta=1.0, ridge=1e-8):
    """
    对窗口 X (m×n) 计算指数加权均值/协方差。
    delta=1.0 时退化为样本均值+样本协方差（带微小 ridge）。
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    w = exp_weights(m, float(delta))
    mu = (w[:, None] * X).sum(axis=0)
    Xc = X - mu
    # 加权协方差：sum_j w_j * x_j x_j^T
    S = (Xc * w[:, None]).T @ Xc
    S = 0.5 * (S + S.T)
    # 对角加载，避免病态
    S += ridge * np.trace(S) / n * np.eye(n)
    return mu, S


# ===============================
# OP（最大 Sharpe；Schaible 变换；支持 long-only）
# max_y mu^T y - rf * eta
# s.t. sum(y) = eta, y>=0(可选), y^T Σ y <= 1
# 令 w = y/eta
# ===============================
def solve_max_sharpe(mu, Sigma, rf=0.0, allow_short=False,
                     scs_max_iters=4000, scs_eps=1e-5):
    n = len(mu)
    y = cp.Variable(n)
    eta = cp.Variable()
    cons = [cp.sum(y) == eta, cp.quad_form(y, Sigma) <= 1.0]
    if not allow_short:
        cons += [y >= 0]
    obj = cp.Maximize(mu @ y - rf * eta)
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.SCS, verbose=False,
                   max_iters=scs_max_iters, eps=scs_eps, warm_start=True)
    except Exception:
        return np.ones(n) / n
    # 回退：求解失败则等权
    if y.value is None or eta.value is None or eta.value <= 0:
        return np.ones(n) / n
    w = np.array(y.value / eta.value).ravel()
    if not allow_short:
        w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones(n) / n


# ===============================
# δ 网格搜索（in-sample），以 SR 为准则
# ===============================
def grid_search_delta(R, window, rf, grid, rebalance=1,
                      allow_short=False, scs_max_iters=4000, scs_eps=1e-5):
    """
    在 IS（前半段）用滚动窗口模拟，选择使 SR 最大的 δ 。
    简化：每期再平衡 rebalance=1（与论文“递归再平衡”一致）。
    """
    T, N = R.shape
    split = T // 2
    best_delta, best_sr = None, -1e9
    for delta in grid:
        t = window
        rets = []
        while t < split:
            hist = R[t-window:t, :]
            mu, Sigma = ew_mean_cov(hist, delta=delta)
            w = solve_max_sharpe(mu, Sigma, rf=rf, allow_short=allow_short,
                                 scs_max_iters=scs_max_iters, scs_eps=scs_eps)
            # 每期再平衡（IS）
            rets.append(float((w * R[t, :]).sum()))
            t += rebalance
        sr = sharpe_ratio_np(np.array(rets), rf=rf, per_year=52, annualize=False)
        if sr > best_sr:
            best_sr, best_delta = sr, delta
    return best_delta if best_delta is not None else 1.0


# ===============================
# OOS 回测（与 LSTM 极简脚本同口径）
# ===============================
def run_op_oos(R, start, window, rf, delta,
               rebalance=1, allow_short=False, scs_max_iters=4000, scs_eps=1e-5):
    T, N = R.shape
    t = start
    out = []
    while t < T:
        hist = R[t-window:t, :]
        mu, Sigma = ew_mean_cov(hist, delta=delta)
        w = solve_max_sharpe(mu, Sigma, rf=rf, allow_short=allow_short,
                             scs_max_iters=scs_max_iters, scs_eps=scs_eps)
        for _ in range(rebalance):
            if t >= T: break
            out.append(float((w * R[t, :]).sum()))
            t += 1
    return np.array(out, dtype=float)


# ===============================
# 主程序
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet_assets", default="Assets_Returns")
    ap.add_argument("--sheet_index",  default="Index_Returns")

    # 回测设置（与 LSTM 极简版保持一致）
    ap.add_argument("--window", type=int, default=52, help="滚动窗口长度")
    ap.add_argument("--rebalance", type=int, default=1, help="每期再平衡频率（论文常用 1）")
    ap.add_argument("--split_ratio", type=float, default=0.5, help="前多少比例作为 IS（用于 δ 网格搜索）")

    # Sharpe 口径
    ap.add_argument("--rf", type=float, default=0.0, help="无风险利率（每期）")
    ap.add_argument("--per_year", type=int, default=52, help="年化频次：周=52，月=12，日=252")
    ap.add_argument("--annualize", action="store_true", help="是否对 Sharpe 年化")

    # 遗忘因子 δ
    ap.add_argument("--delta", type=float, default=None, help="固定 δ（不搜索）；不填则做网格搜索")
    ap.add_argument("--delta_grid", default="0.90,0.92,0.94,0.96,0.97,0.98,0.985,0.99,0.995",
                    help="IS 网格搜索的 δ 候选，逗号分隔")

    # 约束 & 求解器
    ap.add_argument("--allow_short", action="store_true", help="是否允许卖空（默认 long-only）")
    ap.add_argument("--scs_max_iters", type=int, default=4000)
    ap.add_argument("--scs_eps", type=float, default=1e-5)

    args = ap.parse_args()

    # 读并对齐
    R_assets = load_sheet(args.excel, args.sheet_assets)
    R_index  = load_sheet(args.excel, args.sheet_index)
    if R_index.shape[1] != 1:  # 只取第一列指数
        R_index = R_index[:, [0]]
    R, r_idx = align_two(R_assets, R_index[:,0])

    T, N = R.shape
    print(f"Loaded Assets_Returns: T={T}, N={N}; Index_Returns: T={T}, N=1\n")

    # 划分 IS/OOS 起点
    split = int(T * args.split_ratio)
    if split <= args.window:
        raise ValueError("split_ratio 太小，IS 长度不足形成一个 window。")
    start = max(args.window, split)

    # δ：固定或 IS 网格搜索
    if args.delta is not None:
        delta = float(args.delta)
        print(f"[δ] fixed delta = {delta:.4f}")
    else:
        grid = [float(x) for x in args.delta_grid.split(",") if x.strip()]
        print(f"[δ] grid search on IS: {grid}")
        delta = grid_search_delta(
            R[:split], window=args.window, rf=args.rf, grid=grid,
            rebalance=1, allow_short=args.allow_short,
            scs_max_iters=args.scs_max_iters, scs_eps=args.scs_eps
        )
        print(f"[δ] best (IS) delta = {delta:.4f}")

    # OOS：OP / EW / MI
    op_r = run_op_oos(
        R, start=start, window=args.window, rf=args.rf, delta=delta,
        rebalance=args.rebalance, allow_short=args.allow_short,
        scs_max_iters=args.scs_max_iters, scs_eps=args.scs_eps
    )
    end = start + len(op_r)
    ew_r = R[start:end, :].mean(axis=1)
    mi_r = r_idx[start:end]

    # SR（可年化）
    sr_op = sharpe_ratio_np(op_r, rf=args.rf, per_year=args.per_year, annualize=args.annualize)
    sr_ew = sharpe_ratio_np(ew_r, rf=args.rf, per_year=args.per_year, annualize=args.annualize)
    sr_mi = sharpe_ratio_np(mi_r, rf=args.rf, per_year=args.per_year, annualize=args.annualize)

    title = f"SR (annualized={args.annualize}, per_year={args.per_year})  OOS=[{start},{end})  delta={delta:.4f}"
    print("\n=== Sharpe Ratio (OP vs EW & MI) ===")
    print(title)
    print("-" * len(title))
    print(f"OP     : {sr_op:.4f}")
    print(f"EW     : {sr_ew:.4f}")
    print(f"MI     : {sr_mi:.4f}")


if __name__ == "__main__":
    """
    用法示例（按论文常见周频设置）：

    # 1) 周频、每期再平衡、前半 IS 网格搜索 δ、SR 年化到年（√52）
    python op_sr_min.py --excel /path/Your.xlsx \
        --window 52 --rebalance 1 --split_ratio 0.5 \
        --annualize --per_year 52

    # 2) 指定一个固定 δ（不做搜索）
    python op_sr_min.py --excel /path/Your.xlsx \
        --window 52 --rebalance 1 --split_ratio 0.5 \
        --annualize --per_year 52 --delta 0.97

    # 3) 月频（若你的表是月度收益），60个月窗口、每月再平衡、SR 年化（√12）
    python op_sr_min.py --excel /path/Your.xlsx \
        --window 60 --rebalance 1 --split_ratio 0.5 \
        --annualize --per_year 12
    """
    main()
