# -*- coding: utf-8 -*-
import argparse, math, random, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- 全局 --------------------
WEEKS_PER_YEAR = 52

def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_div(a, b, eps=1e-12): return a / (b + eps)

def to_np(a):
    if isinstance(a, np.ndarray): return a
    return np.asarray(a)

# -------------------- T-bill 读取与周频化 --------------------
def load_tbill_series(tbill_path, T, yearly_tbill=True):
    """
    读取 T-bill 序列，并对齐到长度 T（资产收益的长度）。
    - 支持 txt/csv（单列或多列，取第一列）
    - 若为年化（yearly_tbill=True），转换为周频： (1+r_year)^(1/52)-1
    - 若长度不足，使用最后一个值延展；若过长，截断
    - 自动把百分数（>1）转为小数
    返回：np.array 长度 T 的周频 r_f 序列
    """
    # 读文件
    ext = os.path.splitext(tbill_path)[1].lower()
    if ext in [".csv"]:
        rf = pd.read_csv(tbill_path, header=None).iloc[:, 0].values
    else:
        # .txt 或其他当成 csv 无表头读取
        rf = pd.read_csv(tbill_path, header=None).iloc[:, 0].values

    rf = rf.astype(float)

    # 若给的是百分比（例如 5 表示 5%），转换为 0.05
    if np.nanmax(np.abs(rf)) > 1.0:
        rf = rf / 100.0

    # 年化 -> 周频
    if yearly_tbill:
        rf = (1.0 + rf) ** (1.0 / WEEKS_PER_YEAR) - 1.0

    # 对齐长度
    if len(rf) < T:
        # 用最后一个值延展
        if len(rf) == 0:
            rf = np.zeros(T, dtype=float)
        else:
            last = rf[-1]
            pad = np.full(T - len(rf), last, dtype=float)
            rf = np.concatenate([rf, pad], axis=0)
    elif len(rf) > T:
        rf = rf[:T]

    # 清理 NaN/Inf
    rf = np.where(np.isfinite(rf), rf, 0.0)
    return rf

# -------------------- 年化风险指标（支持时变 rf） --------------------
def _excess(returns, rf):
    r = to_np(returns).astype(float)
    if np.isscalar(rf):
        ex = r - rf
    else:
        rf = to_np(rf).astype(float)
        if len(rf) != len(r):  # 广播/截断保护
            m = min(len(rf), len(r))
            ex = r[:m] - rf[:m]
        else:
            ex = r - rf
    return ex

def equity_curve(returns):
    returns = to_np(returns).astype(float)
    returns = np.where(np.isfinite(returns), returns, 0.0)
    return np.cumprod(1.0 + returns)

def drawdown_series(returns):
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-12)
    return dd

def sharpe_ratio(returns, rf):
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    vol_ann  = np.std(ex, ddof=0) * math.sqrt(WEEKS_PER_YEAR)
    return safe_div(mean_ann, vol_ann)

def sortino_ratio(returns, rf):
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    downside = ex[ex < 0]
    dd_ann = np.std(downside, ddof=0) * math.sqrt(WEEKS_PER_YEAR)
    return safe_div(mean_ann, dd_ann)

def omega_ratio(returns, rf, threshold=0.0):
    ex = _excess(returns, rf)
    gains = ex[ex > threshold].sum()
    losses = -ex[ex < threshold].sum()
    return safe_div(gains, losses)

def conditional_sharpe_ratio(returns, rf, alpha=0.05):
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    q = np.quantile(ex, alpha)
    tail = ex[ex < q]
    tail_std_ann = np.std(tail, ddof=0) * math.sqrt(WEEKS_PER_YEAR)
    return safe_div(mean_ann, tail_std_ann)

def entropic_sharpe_ratio(returns, rf=None, lam=1.0):
    # ESR 通常直接基于原收益或超额收益，这里用超额收益更一致
    if rf is None:
        x = to_np(returns).astype(float)
    else:
        x = _excess(returns, rf)
    return (1.0 / lam) * math.log(np.mean(np.exp(lam * x)) + 1e-12)

def calmar_ratio(returns, rf):
    # 分子：超额收益年化；分母：组合回撤（基于组合净值）
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    dd = drawdown_series(returns)  # 用组合原始收益求净值与回撤
    mdd = np.max(dd)
    return safe_div(mean_ann, mdd)

def pain_ratio(returns, rf):
    # 分子：超额收益年化；分母：平均回撤
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    dd = drawdown_series(returns)
    pain = np.mean(dd)
    return safe_div(mean_ann, pain)

def martin_ratio(returns, rf):
    # 分子：超额收益年化；分母：Ulcer Index（回撤的 RMS）
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    dd = drawdown_series(returns)
    ulcer = math.sqrt(np.mean(dd**2))
    return safe_div(mean_ann, ulcer)

def conditional_pain_ratio(returns, rf, alpha=0.05):
    # 分子：超额收益年化；分母：回撤分位数
    ex = _excess(returns, rf)
    mean_ann = np.mean(ex) * WEEKS_PER_YEAR
    dd = drawdown_series(returns)
    q = np.quantile(dd, 1.0 - alpha)
    return safe_div(mean_ann, q)

# -------------------- 数据构造（滑动窗口） --------------------
def make_dataset(data, window=20):
    X, Y = [], []
    for t in range(len(data) - window):
        X.append(data[t:t+window])            # 窗口内资产收益
        Y.append(data[t+1:t+window+1])        # 下一期对齐
    return np.array(X), np.array(Y)

# -------------------- LSTM（增强版） --------------------
class LSTMPortfolio(nn.Module):
    """
    - LayerNorm 预处理
    - 多层/双向 LSTM
    - 轻量时间注意力聚合（比只取最后一步更稳）
    - Softmax(温度) 投出到单纯形
    """
    def __init__(self, n_assets, hidden=64, num_layers=2, bidirectional=True,
                 dropout=0.1, temp=6.0):
        super().__init__()
        self.n_assets = n_assets
        self.temp = temp

        self.in_norm = nn.LayerNorm(n_assets)

        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        feat_dim = hidden * (2 if bidirectional else 1)

        self.attn_v = nn.Parameter(torch.randn(feat_dim, 1) * 0.01)
        self.attn_dropout = nn.Dropout(dropout)

        self.head = nn.Linear(feat_dim, n_assets)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x):
        # x: (B, T, N)
        x = self.in_norm(x)
        out, _ = self.lstm(x)                             # (B, T, F)
        scores = torch.matmul(out, self.attn_v).squeeze(-1)  # (B, T)
        alpha = torch.softmax(scores, dim=1)                 # (B, T)
        alpha = self.attn_dropout(alpha)
        context = torch.sum(out * alpha.unsqueeze(-1), dim=1)  # (B, F)
        logits = self.head(context)                           # (B, N)
        w = torch.softmax(logits * self.temp, dim=-1)         # (B, N)
        return w

# -------------------- 主流程 --------------------
def main(args):
    set_seed(args.seed)

    # 读取资产与指数（指数当前未使用，保留接口）
    assets = pd.read_excel(args.excel_path, sheet_name=args.assets_sheet).values
    _index  = pd.read_excel(args.excel_path, sheet_name=args.index_sheet).values.squeeze()

    T, N = assets.shape

    # 读取并周频化无风险利率
    rf_weekly_full = load_tbill_series(args.tbill_path, T=T, yearly_tbill=args.yearly_tbill)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 50/50 划分
    split = T // 2
    train_assets = assets[:split]
    test_assets  = assets[split:]
    rf_train_full = rf_weekly_full[:split]
    rf_test_full  = rf_weekly_full[split:]

    # 数据集（滑动窗口）
    X_train_np, Y_train_np = make_dataset(train_assets, window=args.window)
    X_test_np,  Y_test_np  = make_dataset(test_assets,  window=args.window)

    # 与窗口对齐的 rf（取每个样本窗口的“下一步” rf 作为超额收益的对齐项）
    # 例如 port_ret = sum(Y[:, -1, :] * w) 与 rf_aligned 同步
    rf_train_aligned = rf_train_full[args.window:]   # 长度 = len(train_assets) - window
    rf_test_aligned  = rf_test_full[args.window:]

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train_np,  dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test_np,   dtype=torch.float32).to(device)

    rf_train_t = torch.tensor(rf_train_aligned, dtype=torch.float32).to(device)
    rf_test_t  = torch.tensor(rf_test_aligned,  dtype=torch.float32).to(device)

    # 模型与优化器（SGDM）
    model = LSTMPortfolio(
        n_assets=N,
        hidden=args.hidden,
        num_layers=args.num_layers,
        bidirectional=not args.unidirectional,
        dropout=args.dropout,
        temp=args.temp
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # ---------------- 训练：基于超额收益的 Sharpe-like 损失 ----------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        w = model(X_train)  # (B, N)
        port_ret = torch.sum(Y_train[:, -1, :] * w, dim=-1)  # (B,)
        excess = port_ret - rf_train_t                       # 对齐的周频无风险序列
        loss = -torch.mean(excess) / (torch.std(excess, unbiased=False) + 1e-8)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            if args.clip_norm and args.clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
            optimizer.step()
        else:
            print(f"[Warn] epoch {epoch} loss is NaN/Inf. Skip step.")

        if epoch % max(1, args.log_interval) == 0:
            print(f"[{epoch:03d}] train loss={loss.item():.6f}")

    # ---------------- 评估（Train/Test，全部年化指标） ----------------
    model.eval()
    with torch.no_grad():
        w_train = model(X_train).cpu().numpy()
        w_test  = model(X_test).cpu().numpy()

    # 将每个样本窗口的权重，与其对应“下一步真实收益”做点积 → 形成一条时间序列
    port_ret_train = (train_assets[args.window:] @ w_train.T).diagonal()  # (B_train,)
    port_ret_test  = (test_assets[args.window:]  @ w_test.T ).diagonal()  # (B_test,)

    def report(name, r, rf_series):
        print(f"\n=== {name} (Annualized, Excess over T-bill) ===")
        print(f"Sharpe Ratio       : {sharpe_ratio(r, rf_series):.6f}")
        print(f"Sortino Ratio      : {sortino_ratio(r, rf_series):.6f}")
        print(f"Omega Ratio        : {omega_ratio(r, rf_series):.6f}")
        print(f"CSR                : {conditional_sharpe_ratio(r, rf_series):.6f}")
        print(f"ESR                : {entropic_sharpe_ratio(r, rf_series):.6f}")
        print(f"Calmar Ratio       : {calmar_ratio(r, rf_series):.6f}")
        print(f"Martin Ratio       : {martin_ratio(r, rf_series):.6f}")
        print(f"Pain Ratio         : {pain_ratio(r, rf_series):.6f}")
        print(f"Conditional Pain R.: {conditional_pain_ratio(r, rf_series):.6f}")

    report("Train Set", port_ret_train, rf_train_aligned)
    report("Test  Set", port_ret_test,  rf_test_aligned)

# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument("--excel_path", type=str, required=True)
    p.add_argument("--assets_sheet", type=str, default="Assets_Returns")
    p.add_argument("--index_sheet",  type=str, default="Index_Returns")
    p.add_argument("--tbill_path",   type=str, required=True)
    p.add_argument("--yearly_tbill", action="store_true",
                   help="若提供的是年化 T-bill（常见的3M年化），加此开关将自动换算到周频。")

    # 模型/训练
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--unidirectional", action="store_true", help="默认双向；加此参数改为单向 LSTM")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temp", type=float, default=6.0, help="softmax 温度（越大越尖锐）")

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)

    # 其他
    p.add_argument("--seed", type=int, default=2025)

    args = p.parse_args()
    main(args)
