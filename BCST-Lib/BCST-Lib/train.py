# portfolio_demo.py
import argparse
import pandas as pd
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 数据读取与预处理
# =========================
def load_excel_returns(path, sheet=0, usecols=None):
    """
    读取 Excel，返回 numpy 数组：
    - 单列：shape [T, 1]
    - 多列：shape [T, N]
    """
    df = pd.read_excel(path, sheet_name=sheet)
    if usecols is not None:
        df = df[usecols]
    # 只保留数值列
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError("Excel 未检测到数值列，请检查文件/列名。")
    R = num_df.values.astype(float)
    if R.ndim == 1:
        R = R[:, None]
    return R  # [T, N]

def train_test_split_by_ratio(R, train_ratio=0.7):
    T = R.shape[0]
    split = int(T * train_ratio)
    return R[:split], R[split:]

def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)

# =========================
# LSTM 模型（两条路径通用）
# =========================
class ReturnLSTM(nn.Module):
    # 路径A：用于预测下一期均值向量（或作为因子）
    def __init__(self, n_assets, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_assets)
    def forward(self, x):  # x: [B, T, N]
        out,_ = self.lstm(x)
        mu_hat = self.head(out[:, -1, :])  # [B, N]
        return mu_hat

class End2EndPolicy(nn.Module):
    # 路径B：端到端，输出权重（softmax 投影到单纯形）
    def __init__(self, n_assets, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head_w = nn.Linear(hidden, n_assets)
        # CVaR/EVaR 辅助变量
        self.alpha = nn.Parameter(torch.tensor(0.0))   # for CVaR
        self.rho   = nn.Parameter(torch.tensor(0.1))   # for EVaR (>0 after softplus)
    def forward(self, x):  # x: [B, T, N]
        out,_ = self.lstm(x)
        logits = self.head_w(out[:, -1, :])
        w = F.softmax(logits, dim=-1)  # [B, N], 非负且和为1
        return w

# =========================
# 统计工具
# =========================
def ewma_cov(returns, lam=0.94):
    """
    returns: [T, N]，返回 EWMA 协方差矩阵 [N, N]
    """
    r = returns - returns.mean(0, keepdims=True)
    N = r.shape[1]
    S = np.zeros((N, N))
    w = 1.0
    denom = 0.0
    for t in range(r.shape[0]-1, -1, -1):
        S += w * np.outer(r[t], r[t])
        denom += w
        w *= lam
    if denom <= 1e-12:
        denom = 1.0
    S /= denom
    # PSD 修正（数值稳健）
    S = (S + S.T) / 2
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals[eigvals < 1e-10] = 1e-10
    S = (eigvecs * eigvals) @ eigvecs.T
    return S

# =========================
# 路径 A：预测 -> 凸优化
# 示例：Sharpe 的 Schaible 变换（式(23)）；ESR（式(27)）
# =========================
def solve_max_sharpe(mu_hat, V, rf=0.0, allow_short=False):
    """
    最大化 Sharpe 的凸化：
    max_y,eta  mu^T y - rf*eta
    s.t. 1^T y = eta, y^T V y <= 1, (可选) y >= 0
    返回 x* = y*/eta*
    """
    n = len(mu_hat)
    y = cp.Variable(n)
    eta = cp.Variable()

    constraints = [cp.sum(y) == eta, cp.quad_form(y, V) <= 1.0]
    if not allow_short:
        constraints.append(y >= 0)

    obj = cp.Maximize(mu_hat @ y - rf * eta)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if y.value is None or eta.value is None or eta.value <= 0:
        raise RuntimeError("Convex solver failed for Sharpe.")
    x_star = y.value / eta.value
    return x_star

def solve_max_esr(mu_hat, returns_win, theta=0.95, rf=0.0, allow_short=False):
    """
    最大化 ESR 的凸化（基于 EVaR 约束）：
    max_y,eta, rho  mu^T y - rf*eta
    s.t. 1^T y = eta,
         rho * log( (1/((1-theta)m)) * sum_j exp(-r_j^T y / rho) ) <= 1,
         (可选) y >= 0
    返回 x* = y*/eta*
    """
    n = len(mu_hat)
    y = cp.Variable(n)
    eta = cp.Variable()
    rho = cp.Variable(pos=True)

    R = returns_win  # [m, n]
    m = R.shape[0]

    constraints = [cp.sum(y) == eta,
                   rho * cp.log( (1.0/(m*(1-theta))) * cp.sum(cp.exp(-(R @ y) / rho)) ) <= 1.0]
    if not allow_short:
        constraints.append(y >= 0)

    obj = cp.Maximize(mu_hat @ y - rf * eta)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if y.value is None or eta.value is None or eta.value <= 0:
        raise RuntimeError("Convex solver failed for ESR.")
    return y.value / eta.value

def rolling_rebalance_pathA(R, window=52, rebalance_every=4, rf=0.0,
                            target="sharpe", allow_short=False,
                            lstm_hidden=64, lstm_layers=1, epochs=5, lr=1e-3):
    """
    R: [T, N] 收益序列
    过程：
      - 用 LSTM 预测下一期均值 mu_hat
      - 协方差用 EWMA
      - 凸优化求权重（Sharpe/ESR）
      - 每 rebalance_every 期再平衡一次
    """
    T, N = R.shape
    if T <= window + 1:
        raise ValueError("样本太短，增加数据或降低 window。")

    # 训练一个简易 LSTM 做均值预测（仅示例，可自行替换你的模型/特征）
    model = ReturnLSTM(N, hidden=lstm_hidden, layers=lstm_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 构造滑窗训练集：用 window 步预测下一步
    X_list, Y_list = [], []
    for t in range(window, T-1):
        X_list.append(R[t-window:t, :])
        Y_list.append(R[t, :])  # 用真实下一期收益作为近似“均值”标签
    X = to_torch(np.stack(X_list))   # [B, window, N]
    Y = to_torch(np.stack(Y_list))   # [B, N]

    # 训练若干 epoch（回归 MSE）
    for _ in range(epochs):
        mu_pred = model(X)
        loss = F.mse_loss(mu_pred, Y)
        opt.zero_grad(); loss.backward(); opt.step()

    weights = []
    port_returns = []

    # 滚动回测 + 再平衡
    t = window
    last_w = None
    while t < T:
        t0 = t - window
        hist = R[t0:t, :]                  # [window, N]
        # 预测下一期均值（取最后一个窗预测）
        mu_hat = model(to_torch(hist[None, ...])).detach().numpy().ravel()
        # 协方差
        V_hat = ewma_cov(hist)

        # 凸优化
        if N == 1:
            # 单资产：权重即 1
            w = np.array([1.0])
        else:
            if target.lower() == "sharpe":
                w = solve_max_sharpe(mu_hat, V_hat, rf=rf, allow_short=allow_short)
            elif target.lower() == "esr":
                w = solve_max_esr(mu_hat, hist, theta=0.95, rf=rf, allow_short=allow_short)
            else:
                raise ValueError("target 仅演示 'sharpe' 或 'esr'，你也可扩展到 SoR/CSR/回撤类。")
            # 归一化 & 非负修正（防数值偏差）
            if not allow_short:
                w = np.clip(w, 0, None)
            s = w.sum()
            if s <= 1e-12:
                w = np.ones_like(w) / len(w)
            else:
                w = w / s

        # 保存权重
        weights.append((t, w.copy()))
        # 执行 rebalance_every 期
        for k in range(rebalance_every):
            if t >= T: break
            r_t = R[t, :]      # 实际收益
            pr = float((w * r_t).sum())
            port_returns.append(pr)
            t += 1

    return np.array(port_returns), weights

# =========================
# 路径 B：端到端
# =========================
def portfolio_returns(w, x_next):
    # w: [B, N], x_next: [B, N] —— 组合收益 [B]
    return (w * x_next).sum(dim=-1)

def sharpe_loss(r, rf=0.0, eps=1e-6):
    ex = r.mean() - rf
    std = r.std(unbiased=False) + eps
    return -(ex / std)

def sortino_loss(r, rb=0.0, eps=1e-6):
    downside = F.relu(rb - r)
    dd = torch.sqrt((downside**2).mean() + eps)
    ex = r.mean() - rb
    return -(ex / (dd + eps))

def cvar_loss(model, r, theta=0.95):
    alpha = model.alpha
    penalty = F.relu(-(r - alpha))
    return alpha + penalty.mean() / (1.0 - theta)

def evar_loss(model, r, theta=0.95):
    rho = F.softplus(model.rho) + 1e-3  # >0
    m = r.numel()
    lse = torch.logsumexp(-r / rho, dim=0) - torch.log(torch.tensor(m*(1-theta), dtype=r.dtype))
    return rho * lse

def smooth_max_drawdown(r, beta=0.9):
    # EMA 平滑近似运行峰值，构造回撤
    C = torch.cumsum(r, dim=0)
    M = []
    m = C[0]
    for t in range(len(C)):
        m = beta*m + (1-beta)*torch.maximum(m, C[t])
        M.append(m)
    M = torch.stack(M)
    dd = M - C
    mdd = dd.max()
    add = dd.mean()
    return mdd, add

def calmar_like_loss(r, rf=0.0):
    mdd, _ = smooth_max_drawdown(r)
    ex = r.mean() - rf
    return -(ex / (mdd + 1e-6))

def pain_like_loss(r, rf=0.0):
    _, add = smooth_max_drawdown(r)
    ex = r.mean() - rf
    return -(ex / (add + 1e-6))

def make_batches_for_B(R, window=52):
    """
    组批：用前 window 期预测下一期权重并与下一期收益相乘作为组合收益
    返回 X:[B,T,N], Y:[B,N]
    """
    T, N = R.shape
    Xs, Ys = [], []
    for i in range(window, T):
        Xs.append(R[i-window:i, :])
        Ys.append(R[i, :])  # 下一期收益
    X = to_torch(np.stack(Xs))
    Y = to_torch(np.stack(Ys))
    return X, Y

def train_end2end_pathB(R, window=52, epochs=20, lr=1e-3,
                        loss_name="sharpe", rf=0.0, lam_turnover=0.0,
                        hidden=64, layers=1):
    """
    端到端训练：
      - LSTM -> 权重 -> 组合收益
      - 可选择 Sharpe/Sortino/CVaR/EVaR/Calmar/Pain 代理作为损失
    """
    T, N = R.shape
    if T <= window + 1:
        raise ValueError("样本太短，增加数据或降低 window。")

    X, Y = make_batches_for_B(R, window=window)  # X:[B,T,N], Y:[B,N]
    model = End2EndPolicy(N, hidden=hidden, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        w = model(X)                # [B,N]
        r = portfolio_returns(w, Y) # [B]

        if loss_name.lower() == "sharpe":
            base = sharpe_loss(r, rf=rf)
        elif loss_name.lower() == "sortino":
            base = sortino_loss(r, rb=rf)
        elif loss_name.lower() == "cvar":
            base = cvar_loss(model, r, theta=0.95)
        elif loss_name.lower() == "evar":
            base = evar_loss(model, r, theta=0.95)
        elif loss_name.lower() == "calmar":
            base = calmar_like_loss(r, rf=rf)
        elif loss_name.lower() == "pain":
            base = pain_like_loss(r, rf=rf)
        else:
            raise ValueError("loss_name 仅支持 sharpe/sortino/cvar/evar/calmar/pain")

        # 换手惩罚（降低频繁交易；这里简化：相邻样本的权重差）
        turn_pen = 0.0
        if lam_turnover > 0:
            turn_pen = (w[1:] - w[:-1]).abs().sum(dim=-1).mean() * lam_turnover

        loss = base + turn_pen
        opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % max(1, epochs//10) == 0:
            mean_r = float(r.mean().detach())
            print(f"[B][{ep+1}/{epochs}] loss={loss.item():.6f} mean_r={mean_r:.6f}")

    # 训练完返回最后一批的权重与组合收益（示例）
    return model, w.detach().numpy(), r.detach().numpy()

# =========================
# 命令行与主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, required=True, help="Excel 文件路径")
    ap.add_argument("--sheet", type=str, default="0", help="工作表名或索引（默认0）")
    ap.add_argument("--path", type=str, choices=["A","B"], default="A", help="选择路径：A=预测+凸优化；B=端到端")
    ap.add_argument("--window", type=int, default=52, help="滚动窗口长度")
    ap.add_argument("--rebalance", type=int, default=4, help="路径A的再平衡频率")
    ap.add_argument("--rf", type=float, default=0.0, help="风险自由利率（按同周期）")
    ap.add_argument("--target", type=str, default="sharpe", help="路径A目标：sharpe/esr")
    ap.add_argument("--allow_short", action="store_true", help="路径A允许卖空（默认不允许）")
    ap.add_argument("--epochs", type=int, default=10, help="训练 epoch（两条路径用）")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    ap.add_argument("--loss_name", type=str, default="sharpe", help="路径B损失：sharpe/sortino/cvar/evar/calmar/pain")
    ap.add_argument("--lam_turnover", type=float, default=0.0, help="路径B换手惩罚系数")
    args = ap.parse_args()

    # sheet 解析为 int 或 str
    try:
        sheet = int(args.sheet)
    except:
        sheet = args.sheet

    R = load_excel_returns(args.excel, sheet=sheet)  # [T, N]
    T, N = R.shape
    print(f"Loaded returns: T={T}, N={N}")

    if args.path == "A":
        pr, weights = rolling_rebalance_pathA(
            R, window=args.window, rebalance_every=args.rebalance, rf=args.rf,
            target=args.target, allow_short=args.allow_short,
            lstm_hidden=64, lstm_layers=1, epochs=args.epochs, lr=args.lr
        )
        ann_r = float(np.mean(pr) * 52)  # 如果是周收益，可按年化近似
        print(f"[A] Out-of-sample steps: {len(pr)}, mean={pr.mean():.6f}, ann~={ann_r:.2%}")
        print(f"[A] First 3 weights snapshots: {[ (i, w.round(4).tolist()) for i,(i,w) in zip(range(3), weights[:3]) ]}")

    else:
        model, w_last, r_last = train_end2end_pathB(
            R, window=args.window, epochs=args.epochs, lr=args.lr,
            loss_name=args.loss_name, rf=args.rf, lam_turnover=args.lam_turnover,
            hidden=64, layers=1
        )
        mean_r = float(np.mean(r_last))
        ann_r = float(mean_r * 52)  # 周→年化近似
        print(f"[B] mean={mean_r:.6f}, ann~={ann_r:.2%}")
        print(f"[B] example weights (last batch first 3 rows):\n{w_last[:3]}")

if __name__ == "__main__":
    main()
