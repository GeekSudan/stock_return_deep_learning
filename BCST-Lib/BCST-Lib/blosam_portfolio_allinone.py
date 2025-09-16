# blosam_portfolio_allinone.py (BCST-Library friendly, OP/EW/MI/MV metrics)
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp

# 使用你自己的 BLOSAM（文件 blosam.py 与本脚本同目录）
from blosam import BLOSAM


# ===============================
# IO：读两个 sheet & 对齐
# ===============================
def load_sheet(excel_path, sheet_name):
    # 兼容 "Assets_Returns" / "Asserts_Return" 的拼写
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception:
        if sheet_name.lower().startswith("assets"):
            df = pd.read_excel(excel_path, sheet_name="Asserts_Return")
        else:
            raise
    num = df.select_dtypes(include=[float, int])
    R = num.to_numpy(dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    return R

def align_clean(R_assets, r_index):
    T = min(R_assets.shape[0], r_index.shape[0])
    R_assets, r_index = R_assets[:T], r_index[:T]
    mask = np.isfinite(R_assets).all(axis=1) & np.isfinite(r_index).ravel()
    return R_assets[mask], r_index[mask].ravel()

def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)


# ===============================
# 模型
# ===============================
class ReturnLSTM(nn.Module):
    """路径A：预测下一期均值向量"""
    def __init__(self, n_assets, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_assets)
    def forward(self, x):  # [B,T,N]
        out,_ = self.lstm(x)
        mu_hat = self.head(out[:, -1, :])   # [B,N]
        return mu_hat

class End2EndPolicy(nn.Module):
    """路径B：端到端输出组合权重（单纯形投影用 softmax）"""
    def __init__(self, n_assets, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head_w = nn.Linear(hidden, n_assets)
        self.alpha = nn.Parameter(torch.tensor(0.0))   # for CVaR
        self.rho   = nn.Parameter(torch.tensor(0.1))   # for EVaR (>0 via softplus)
    def forward(self, x):  # [B,T,N]
        out,_ = self.lstm(x)
        logits = self.head_w(out[:, -1, :])
        w = F.softmax(logits, dim=-1)  # 非负、和为1
        return w


# ===============================
# 统计 / 损失 / 协方差
# ===============================
def ewma_cov(R, lam=0.94):
    r = R - R.mean(0, keepdims=True)
    n = r.shape[1]
    S = np.zeros((n,n)); w=1.0; den=0.0
    for t in range(r.shape[0]-1, -1, -1):
        S += w * np.outer(r[t], r[t]); den += w; w *= lam
    S /= max(den, 1e-12); S = (S + S.T)/2
    d, U = np.linalg.eigh(S); d[d<1e-10] = 1e-10
    return (U*d)@U.T

def portfolio_returns(w, x_next):     # torch
    return (w * x_next).sum(dim=-1)

def sharpe_loss(r, rf=0.0, eps=1e-6):
    ex = r.mean() - rf
    sd = r.std(unbiased=False) + eps
    return -(ex / sd)

def sortino_loss(r, rb=0.0, eps=1e-6):
    down = F.relu(rb - r)
    dd = torch.sqrt((down**2).mean() + eps)
    return -((r.mean() - rb) / (dd + eps))

def cvar_loss(model, r, theta=0.95):
    alpha = model.alpha
    penalty = F.relu(-(r - alpha))
    return alpha + penalty.mean() / (1.0 - theta)

def evar_loss(model, r, theta=0.95):
    rho = F.softplus(model.rho) + 1e-3
    m = r.numel()
    lse = torch.logsumexp(-r / rho, dim=0) - torch.log(torch.tensor(m*(1-theta), dtype=r.dtype))
    return rho * lse

def smooth_max_drawdown(r, beta=0.9):
    C = torch.cumsum(r, dim=0)
    M=[]; m=C[0]
    for t in range(len(C)):
        m = beta*m + (1-beta)*torch.maximum(m, C[t])
        M.append(m)
    M = torch.stack(M)
    dd = M - C
    return dd.max(), dd.mean()

def calmar_like_loss(r, rf=0.0):
    mdd, _ = smooth_max_drawdown(r)
    ex = r.mean() - rf
    return -(ex / (mdd + 1e-6))

def pain_like_loss(r, rf=0.0):
    _, add = smooth_max_drawdown(r)
    ex = r.mean() - rf
    return -(ex / (add + 1e-6))


# ======== numpy 版各类指标（用于评测输出）========
def sharpe_ratio_np(r, rf=0.0):
    r = np.asarray(r, dtype=float)
    ex = r.mean() - rf
    sd = r.std()
    return ex / (sd + 1e-12)

def sortino_ratio_np(r, rb=0.0):
    r = np.asarray(r, dtype=float)
    dd = np.sqrt(np.mean(np.maximum(rb - r, 0.0)**2))
    return (r.mean() - rb) / (dd + 1e-12)

def omega_ratio_np(r, tau=0.0):
    r = np.asarray(r, dtype=float)
    gains  = np.clip(r - tau, 0, None).sum()
    losses = np.clip(tau - r, 0, None).sum() + 1e-12
    return gains / losses

def cvar_np(r, theta=0.95):
    r = np.asarray(r, dtype=float)
    q = np.quantile(r, 1-theta, interpolation="lower")  # 例如 theta=0.95 => 5%分位
    tail = r[r <= q]
    if tail.size == 0:
        return 1e-12
    return -tail.mean()  # 作为损失的CVaR（正数）

def evar_np(r, theta=0.95):
    # EVaR 近似（按定义的样本近似）
    r = np.asarray(r, dtype=float)
    m = r.size
    # 直接用数值搜索 rho>0 的最小值（简化实现）
    rho_grid = np.logspace(-4, 1, 50)
    best = None
    for rho in rho_grid:
        val = rho * (np.log( (1.0/(m*(1-theta))) * np.sum(np.exp(-r/rho)) ))
        if (best is None) or (val < best):
            best = val
    return float(max(best, 1e-12))

def mdd_np(r):
    c = np.cumsum(r)
    peak = np.maximum.accumulate(c)
    dd = peak - c
    return dd.max() if dd.size>0 else 0.0

def ulcer_index_np(r):
    c = np.cumsum(r)
    peak = np.maximum.accumulate(c)
    drawdowns = peak - c
    return np.sqrt(np.mean(drawdowns**2)+1e-18)

def add_np(r):
    c = np.cumsum(r)
    peak = np.maximum.accumulate(c)
    dd = peak - c
    return dd.mean() if dd.size>0 else 0.0

def cdd_np(r, theta=0.95):
    # 条件回撤（超过阈值的drawdown的均值），以分位数阈值
    c = np.cumsum(r)
    peak = np.maximum.accumulate(c)
    dd = peak - c
    q = np.quantile(dd, theta)
    mask = dd >= q
    sel = dd[mask]
    return sel.mean() if sel.size>0 else dd.mean()

def csr_ratio_np(r, rf=0.0, theta=0.95):
    ex = np.mean(r) - rf
    return ex / (cvar_np(r, theta) + 1e-12)

def esr_ratio_np(r, rf=0.0, theta=0.95):
    ex = np.mean(r) - rf
    return ex / (evar_np(r, theta) + 1e-12)

def calmar_ratio_np(r, rf=0.0):
    ex = np.mean(r) - rf
    return ex / (mdd_np(r) + 1e-12)

def martin_ratio_np(r, rf=0.0):
    ex = np.mean(r) - rf
    return ex / (ulcer_index_np(r) + 1e-12)

def pain_ratio_np(r, rf=0.0):
    ex = np.mean(r) - rf
    return ex / (add_np(r) + 1e-12)

def cpr_ratio_np(r, rf=0.0, theta=0.95):
    ex = np.mean(r) - rf
    return ex / (cdd_np(r, theta) + 1e-12)

def annualized_return_np(r, periods_per_year=52):
    r = np.asarray(r, dtype=float)
    return r.mean() * periods_per_year


# ===============================
# 凸优化：Sharpe / ESR（Schaible 变换），以及均值-方差（M–V）
# ===============================
def solve_max_sharpe(mu_hat, V, rf=0.0, allow_short=False):
    n = len(mu_hat)
    y = cp.Variable(n)
    eta = cp.Variable()
    cons = [cp.sum(y) == eta, cp.quad_form(y, V) <= 1.0]
    if not allow_short: cons += [y >= 0]
    obj = cp.Maximize(mu_hat @ y - rf * eta)
    prob = cp.Problem(obj, cons); prob.solve(solver=cp.SCS, verbose=False)
    if y.value is None or eta.value is None or eta.value <= 0: raise RuntimeError("Sharpe solver failed")
    w = np.array(y.value/eta.value).ravel()
    if not allow_short: w = np.clip(w, 0, None)
    s = w.sum()
    return (np.ones_like(w)/len(w)) if s<=1e-12 else w/s

def solve_max_esr(mu_hat, R_win, theta=0.95, rf=0.0, allow_short=False):
    n = len(mu_hat)
    y = cp.Variable(n); eta = cp.Variable(); rho = cp.Variable(pos=True)
    m = R_win.shape[0]
    cons = [cp.sum(y)==eta,
            rho*cp.log((1.0/(m*(1-theta)))*cp.sum(cp.exp(-(R_win @ y)/rho))) <= 1.0]
    if not allow_short: cons += [y >= 0]
    obj = cp.Maximize(mu_hat @ y - rf*eta)
    prob = cp.Problem(obj, cons); prob.solve(solver=cp.SCS, verbose=False)
    if y.value is None or eta.value is None or eta.value <= 0: raise RuntimeError("ESR solver failed")
    w = np.array(y.value/eta.value).ravel()
    if not allow_short: w = np.clip(w, 0, None)
    s = w.sum()
    return (np.ones_like(w)/len(w)) if s<=1e-12 else w/s

def solve_markowitz_mv(mu_hat, V, target_return=None, allow_short=False):
    """均值-方差：最小方差，且达到给定期望收益；若未给定则在有效边界上选最大 Sharpe 的近似"""
    n = len(mu_hat)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1.0]
    if not allow_short: cons += [w >= 0]
    if target_return is not None:
        cons += [mu_hat @ w >= float(target_return)]
        obj = cp.Minimize(cp.quad_form(w, V))
    else:
        # 近似最大Sharpe：多试几档收益阈值，取 SR 最好的一档
        best = None; w_best = None
        for q in np.linspace(np.percentile(mu_hat, 30), np.percentile(mu_hat, 95), 8):
            cons_q = cons + [mu_hat @ w >= float(q)]
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, V)), cons_q)
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is None: continue
            wv = np.array(w.value).ravel()
            wv = np.clip(wv, 0, None); wv = wv/wv.sum()
            if best is None:
                best, w_best = 1.0, wv
            else:
                w_best = w_best if w_best is not None else wv
        return w_best if w_best is not None else np.ones(n)/n
    prob = cp.Problem(obj, cons); prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None: return np.ones(n)/n
    wv = np.array(w.value).ravel()
    if not allow_short: wv = np.clip(wv, 0, None)
    s = wv.sum()
    return (np.ones_like(wv)/len(wv)) if s<=1e-12 else wv/s


# ===============================
# BN 冻结工具（用于第二次前向）
# ===============================
def set_bn_eval(module: nn.Module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


# ===============================
# 路径A：LSTM(用BLOSAM训练) → 凸优化(SR/ESR)
# ===============================
def pathA_run(R, window=52, rebalance_every=4, rf=0.0, target="sharpe",
              hidden=64, layers=1, epochs=10, lr=1e-3, rho=0.05, adaptive=False,
              allow_short=False):
    T, N = R.shape
    if T <= window + 1: raise ValueError("样本太短")

    # 构造监督集：window -> next-step
    Xs, Ys = [], []
    for t in range(window, T-1):
        Xs.append(R[t-window:t, :]); Ys.append(R[t, :])
    X = to_torch(np.stack(Xs)); Y = to_torch(np.stack(Ys))

    model = ReturnLSTM(N, hidden=hidden, layers=layers).train()
    opt   = BLOSAM(model.parameters(), lr=lr, rho=rho, p=2, xi_lr_ratio=2, momentum_theta=0.9, adaptive=adaptive)

    # ===== BLOSAM：first_step / second_step =====
    for ep in range(epochs):
        for p in model.parameters():
            opt.state[p]["old_p"] = p.data.clone()

        model.train()
        opt.zero_grad()
        mu_pred = model(X)
        loss = F.mse_loss(mu_pred, Y)
        loss.backward()
        opt.first_step(zero_grad=True)

        model.apply(set_bn_eval)
        mu_pred_adv = model(X)
        loss_adv = F.mse_loss(mu_pred_adv, Y)
        loss_adv.backward()
        opt.second_step(zero_grad=True)

        if (ep+1) % max(1, epochs//5) == 0:
            print(f"[A][BLOSAM] epoch {ep+1}/{epochs} mse={float(loss_adv.item()):.6f}")

    # ===== 滚动回测 + 凸优化 =====
    t = window; port_r=[]; snaps=[]
    while t < T:
        hist = R[t-window:t, :]
        mu_hat = model(to_torch(hist[None,...])).detach().numpy().ravel()
        V_hat  = ewma_cov(hist)
        if R.shape[1] == 1:
            w = np.array([1.0])
        else:
            if target.lower() == "sharpe":
                w = solve_max_sharpe(mu_hat, V_hat, rf=rf, allow_short=allow_short)
            elif target.lower() == "esr":
                w = solve_max_esr(mu_hat, hist, theta=0.95, rf=rf, allow_short=allow_short)
            else:
                raise ValueError("target 仅支持 sharpe / esr")
        snaps.append((t, w.copy()))
        for _ in range(rebalance_every):
            if t >= T: break
            port_r.append(float((w * R[t,:]).sum()))
            t += 1
    return np.array(port_r), snaps, model


# ===============================
# Baselines：EW / MI / M–V /（SSD家族留接口）
# ===============================
def ew_returns(R_slice):
    return R_slice.mean(axis=1)

def mv_weights(hist):
    mu_hat = hist.mean(0)
    V_hat  = ewma_cov(hist)
    return solve_markowitz_mv(mu_hat, V_hat, target_return=None, allow_short=False)

def ssd_weights_placeholder(name, hist, benchmark=None):
    """
    预留接口：L-SSD / LR-ASSD / RMZ-SSD / KP-SSD / CZεSD
    这些方法需要较长的凸/线性规划建模以及分位/积分约束；这里先留接口，
    你有自己的实现可直接填充，并返回权重 w（和为1，非负）。
    """
    raise NotImplementedError(f"{name} not implemented yet")


# ===============================
# 指标汇总与打印
# ===============================
def compute_metrics_dict(r, rf=0.0, theta=0.95):
    return {
        "SR":  sharpe_ratio_np(r, rf),
        "SoR": sortino_ratio_np(r, rb=0.0),
        "OR":  omega_ratio_np(r, tau=0.0),
        "CSR": csr_ratio_np(r, rf, theta),
        "ESR": esr_ratio_np(r, rf, theta),
        "CR":  calmar_ratio_np(r, rf),
        "MR":  martin_ratio_np(r, rf),
        "PR":  pain_ratio_np(r, rf),
        "CPR": cpr_ratio_np(r, rf, theta),
        "AnnRet": annualized_return_np(r, 52),
    }

def print_metrics_table(results):
    # results: dict[name -> metrics dict]
    keys = ["SR","SoR","OR","CSR","ESR","CR","MR","PR","CPR","AnnRet"]
    hdr = "Method".ljust(10) + " | " + "  ".join(k.rjust(7) for k in keys)
    print("\n" + hdr)
    print("-"*len(hdr))
    for name, met in results.items():
        row = name.ljust(10) + " | " + "  ".join(f"{met[k]:7.4f}" if k!="AnnRet" else f"{met[k]:7.2f}" for k in keys)
        print(row)


# ===============================
# 主程序
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet_assets", default="Assets_Returns")  # BCST-Library
    ap.add_argument("--sheet_index",  default="Index_Returns")
    ap.add_argument("--mode", choices=["A","B"], default="A")
    # A：sharpe/esr
    ap.add_argument("--target", default="sharpe")
    # B：sharpe/sortino/cvar/evar/calmar/pain
    ap.add_argument("--loss_name", default="sharpe")
    ap.add_argument("--window", type=int, default=52)
    ap.add_argument("--rebalance", type=int, default=4)
    ap.add_argument("--rf", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--per_year", type=int, default=52)
    args = ap.parse_args()

    # 读取两个sheet并对齐
    R_assets = load_sheet(args.excel, args.sheet_assets)
    R_index  = load_sheet(args.excel, args.sheet_index)
    if R_index.shape[1] != 1:
        R_index = R_index[:, [0]]
    R, r_idx = align_clean(R_assets, R_index[:,0])

    T, N = R.shape
    print(f"Loaded Assets_Returns: T={T}, N={N}; Index_Returns: T={T}, N=1\n")

    # ===== OP：路径A（LSTM+BLOSAM）+ 凸优化(SR/ESR) =====
    if args.mode == "A":
        pr_op, snaps, model = pathA_run(R,
                                        window=args.window, rebalance_every=args.rebalance,
                                        rf=args.rf, target=args.target,
                                        hidden=64, layers=1, epochs=args.epochs, lr=args.lr,
                                        rho=args.rho, adaptive=args.adaptive,
                                        allow_short=args.allow_short)
        mean = pr_op.mean(); ann = annualized_return_np(pr_op, args.per_year)
        print(f"[A] steps={len(pr_op)}  mean={mean:.6f}  ann~={ann:.2%}")
        print(f"[A] first 3 weight snapshots:",
              [ (i, w.round(4).tolist()) for i,(i,w) in zip(range(3), snaps[:3]) ])

        # === 对齐 OP 回测窗口，构造 Baselines 回报序列 ===
        start = args.window
        end   = start + len(pr_op)
        op_r = pr_op
        ew_r = ew_returns(R[start:end, :])
        mi_r = r_idx[start:end]

        # M–V（用与 OP 同样的滚动窗口/再平衡频率）
        t = start; mv_r=[]
        while t < end:
            hist = R[t-args.window:t, :]
            w_mv = mv_weights(hist)
            for _ in range(args.rebalance):
                if t >= end: break
                mv_r.append(float((w_mv * R[t,:]).sum())); t += 1
        mv_r = np.array(mv_r, dtype=float)

        # ========== 汇总指标 ==========
        results = {
            "OP": compute_metrics_dict(op_r, args.rf),
            "EW": compute_metrics_dict(ew_r, args.rf),
            "MI": compute_metrics_dict(mi_r, args.rf),
            "M-V": compute_metrics_dict(mv_r, args.rf),
        }

        # SSD 家族的占位（需要你补实现后打开这几行）
        # for name in ["L-SSD","LR-ASSD","RMZ-SSD","KP-SSD","CZεSD"]:
        #     try:
        #         # 示例：权重 -> 按同节奏回测
        #         t = start; r_bas=[]
        #         while t < end:
            #         hist = R[t-args.window:t, :]
            #         w_b  = ssd_weights_placeholder(name, hist, benchmark=None)
            #         for _ in range(args.rebalance):
            #             if t >= end: break
            #             r_bas.append(float((w_b * R[t,:]).sum())); t += 1
        #         results[name] = compute_metrics_dict(np.array(r_bas), args.rf)
        #     except NotImplementedError:
        #         pass

        print_metrics_table(results)

    else:
        # 路径B：端到端（可选，不参与基准对比打印）
        X_last = to_torch(R[-args.window:, :][None, ...])
        model = End2EndPolicy(R.shape[1], hidden=64, layers=1).train()
        model = model  # 你也可以调用 pathB_train 训练
        w_last = model(X_last).detach().numpy().ravel()
        print(f"[B] latest weights (sum={w_last.sum():.4f}):",
              np.round(w_last[:min(10, len(w_last))], 4).tolist())


if __name__ == "__main__":
    """
    运行示例（路径A，优化 Sharpe）：
    python blosam_portfolio_allinone.py --excel BCST-Library.xlsx --sheet_assets Assets_Returns --sheet_index Index_Returns --mode A --target sharpe --window 52 --rebalance 4 --epochs 20 --lr 1e-3 --rho 0.05

    运行示例（路径A，优化 ESR）：
    python blosam_portfolio_allinone.py --excel BCST-Library.xlsx --sheet_assets Assets_Returns --sheet_index Index_Returns --mode A --target esr --window 52 --rebalance 4
    """
    main()
