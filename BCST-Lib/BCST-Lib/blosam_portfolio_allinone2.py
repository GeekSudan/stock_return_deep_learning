# 目标：只比较 Sharpe Ratio (SR)：LSTM / EW / MI
# 论文口径对齐：
#  - 训练用“递归（扩张）窗口”：每个样本是从开头到 t-1 的整段历史 (--expanding_train)
#  - 推断/调仓同样用递归窗口 (--expanding_infer)，可设长度上限 (--max_infer_len)
#  - 样本间使用 δ 衰减权重 (--train_delta)，思想与 OP 一致
#  - 可选在线微调，亦可用递归窗口 (--expanding_refit)
#  - 模型：LSTM(+可选双向) + 时间注意力池化 + LayerNorm + MLP + Softmax(·/τ)
#
# 依赖：numpy, pandas, torch, 以及 blosam 优化器：from blosam import BLOSAM

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from blosam import BLOSAM


# ============== IO ==============
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

def to_torch(x):
    return torch.tensor(x, dtype=torch.float32)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============== 模型（带注意力 + 可选双向） ==============
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H, mask=None):  # H: [B,T,H], mask: [B,T] (True=valid)
        score = torch.tanh(self.W(H))              # [B,T,H]
        score = self.v(score).squeeze(-1)          # [B,T]
        if mask is not None:
            score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=1)        # [B,T]
        ctx = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # [B,H]
        return ctx, alpha

class End2EndPolicy(nn.Module):
    def __init__(self, n_assets, hidden=128, layers=2, dropout=0.1,
                 bidirectional=True, tau=1.0):
        super().__init__()
        self.tau = float(max(tau, 1e-4))
        self.lstm = nn.LSTM(
            input_size=n_assets, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=(dropout if layers>1 else 0.0),
            bidirectional=bool(bidirectional)
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(out_dim)
        self.attn = TemporalAttention(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, n_assets),
        )

    def forward(self, x, mask=None):  # x: [B,T,N], mask: [B,T]
        H, _ = self.lstm(x)           # [B,T,H']
        H = self.ln(H)
        ctx, _ = self.attn(H, mask=mask)
        logits = self.mlp(ctx)        # [B,N]
        w = torch.softmax(logits / self.tau, dim=-1)
        return w


# ============== 指标 & 损失 ==============
def portfolio_returns(w, x_next):
    return (w * x_next).sum(dim=-1)

def sharpe_ratio_np(r, rf=0.0, per_year=52, annualize=False):
    r = np.asarray(r, dtype=float)
    ex = r.mean() - rf
    sd = r.std() + 1e-12
    sr = ex / sd
    return sr * (np.sqrt(per_year) if annualize else 1.0)

def _weighted_stats_torch(r, w, rf=0.0, eps=1e-8):
    w = w / (w.sum() + eps)
    mu = (w * r).sum()
    var = (w * (r - mu)**2).sum()
    sd = torch.sqrt(var + eps)
    ex = mu - rf
    return ex, sd

def sharpe_loss_unweighted(r, rf=0.0, eps=1e-6):
    ex = r.mean() - rf
    sd = r.std(unbiased=False) + eps
    return -(ex / sd)

def sharpe_loss_weighted(r, w, rf=0.0):
    ex, sd = _weighted_stats_torch(r, w, rf)
    return -(ex / (sd + 1e-12))


# ============== 标准化 & padding（支持变长序列） ==============
def standardize_seq_timewise(seq, eps=1e-8):
    mu = seq.mean(axis=0, keepdims=True)
    sd = seq.std(axis=0, keepdims=True) + eps
    return (seq - mu) / sd

def make_batches_expanding(R_train, standardize=False):
    """
    训练样本：X_t = R_train[:t, :],  Y_t = R_train[t, :]
    t=1..(T-1)。返回：X_pad [B,T_max,N], mask [B,T_max], Y [B,N]
    """
    T, N = R_train.shape
    seqs = []
    for t in range(1, T):
        hist = R_train[:t, :]
        if standardize:
            hist = standardize_seq_timewise(hist)
        seqs.append(hist)
    # pad
    T_max = max(s.shape[0] for s in seqs)
    B = len(seqs)
    X_pad = np.zeros((B, T_max, N), dtype=np.float32)
    mask = np.zeros((B, T_max), dtype=bool)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        X_pad[i, :L, :] = s
        mask[i, :L] = True
    Y = R_train[1:, :]  # 对应下期收益
    return to_torch(X_pad), torch.tensor(mask, dtype=torch.bool), to_torch(Y)

def hist_to_model_input(model, hist, standardize=False):
    if hist.shape[0] == 0:
        # 没有历史（极早期），退化为等权
        n = hist.shape[1]
        return np.ones(n) / n
    if standardize:
        hist = standardize_seq_timewise(hist)
    X = to_torch(hist[None, ...])      # [1,T,N]
    mask = torch.ones((1, hist.shape[0]), dtype=torch.bool)
    with torch.no_grad():
        w = model(X, mask=mask).cpu().numpy().ravel()
    w = np.maximum(w, 0.0); s = w.sum()
    return (w / s) if s > 1e-12 else np.ones(hist.shape[1]) / hist.shape[1]


# ============== 训练（BLOSAM，两步法） ==============
def train_lstm_policy_expanding(R_train, epochs=60, lr=1e-3, rho=0.05, rf=0.0,
                                hidden=128, layers=2, dropout=0.1, bidirectional=True, tau=1.0,
                                adaptive=False, seed=42, train_delta=0.995, standardize=False):
    """
    训练样本覆盖 IS 的每一个时间点：X_t = [0..t-1], 目标是当期收益 R[t]
    样本间权重：w_t ∝ δ^(m-1 - t)，越新越大（与 OP 的 δ 思想一致）
    """
    set_seed(seed)
    X, mask, Y = make_batches_expanding(R_train, standardize=standardize)
    N = R_train.shape[1]
    model = End2EndPolicy(n_assets=N, hidden=hidden, layers=layers,
                          dropout=dropout, bidirectional=bidirectional, tau=tau).train()
    opt   = BLOSAM(model.parameters(), lr=lr, rho=rho, adaptive=adaptive)

    # 样本权重（按目标期 t 的新旧程度）
    W = None
    m = X.shape[1]  # T_max（仅用于构造指数序列的长度参考）
    B = X.shape[0]  # 样本数 = T_train-1
    if (train_delta is not None) and (0.0 < train_delta < 1.0):
        # 权重向量与样本索引对齐：样本 i 对应目标期 t=i+1
        idx = np.arange(1, B+1, dtype=float)
        ww = (train_delta ** (B - idx))      # 较新的样本 idx 越大，权重越大
        W = to_torch(ww / ww.sum())          # [B]

    for ep in range(epochs):
        # for p in model.parameters():
        #     opt.state[p]["old_p"] = p.data.clone()

        # Step 1
        opt.zero_grad()
        w_hat = model(X, mask=mask)          # [B,N]
        r = portfolio_returns(w_hat, Y)      # [B]
        loss = sharpe_loss_weighted(r, W, rf) if W is not None else sharpe_loss_unweighted(r, rf)
        loss.backward()
        opt.first_step(zero_grad=True)

        # Step 2
        w_hat2 = model(X, mask=mask)
        r2 = portfolio_returns(w_hat2, Y)
        loss2 = sharpe_loss_weighted(r2, W, rf) if W is not None else sharpe_loss_unweighted(r2, rf)
        loss2.backward()
        opt.second_step(zero_grad=True)

        if (ep+1) % max(1, epochs//10) == 0:
            print(f"[Train(expanding)][epoch {ep+1}/{epochs}] loss={float(loss2.item()):.6f}")

    return model


# ============== OOS（支持递归推断 + 递归微调） ==============
def finetune_on_recent(model, R_hist, epochs, lr, rho, rf, adaptive, train_delta, standardize):
    model.train()
    if R_hist.shape[0] < 2:  # 至少要能形成一个样本
        model.eval(); return
    X, mask, Y = make_batches_expanding(R_hist, standardize=standardize)
    opt  = BLOSAM(model.parameters(), lr=lr, rho=rho, adaptive=adaptive)

    W = None
    B = X.shape[0]
    if (train_delta is not None) and (0.0 < train_delta < 1.0):
        idx = np.arange(1, B+1, dtype=float)
        ww = (train_delta ** (B - idx))
        W = to_torch(ww / ww.sum())

    for _ in range(epochs):
        for p in model.parameters():
            opt.state[p]["old_p"] = p.data.clone()
        opt.zero_grad()
        r = portfolio_returns(model(X, mask=mask), Y)
        loss = sharpe_loss_weighted(r, W, rf) if W is not None else sharpe_loss_unweighted(r, rf)
        loss.backward()
        opt.first_step(zero_grad=True)

        r2 = portfolio_returns(model(X, mask=mask), Y)
        loss2 = sharpe_loss_weighted(r2, W, rf) if W is not None else sharpe_loss_unweighted(r2, rf)
        loss2.backward()
        opt.second_step(zero_grad=True)

    model.eval()

def oos_run(model, R, start, rebalance=1, standardize=False,
            refit_every=13, refit_epochs=3, refit_horizon=260,
            refit_lr=5e-4, refit_rho=0.05, rf=0.0, adaptive=False, train_delta=0.995,
            expanding_infer=True, max_infer_len=0, expanding_refit=False):
    T, N = R.shape
    t = start
    out = []
    step_since_refit = 0
    model.eval()
    while t < T:
        # 在线微调（可选递归窗口）
        if refit_every and refit_epochs and step_since_refit >= refit_every:
            if expanding_refit:
                R_hist = R[:t, :]
            else:
                beg = max(0, t - refit_horizon)
                R_hist = R[beg:t, :]
            finetune_on_recent(model, R_hist, refit_epochs, refit_lr, refit_rho,
                               rf, adaptive, train_delta, standardize)
            step_since_refit = 0
            model.eval()

        # 递归推断：用 [0..t-1] 全历史（可设上限）
        if expanding_infer:
            hist = R[:t, :]
            if max_infer_len and hist.shape[0] > max_infer_len:
                hist = hist[-max_infer_len:, :]
        else:
            # 理论上不推荐固定窗口，但保留选项
            win = min(260, t)  # 兜底窗口
            hist = R[t-win:t, :]

        w = hist_to_model_input(model, hist, standardize=standardize)

        # 持有 rebalance 期
        for _ in range(rebalance):
            if t >= T: break
            out.append(float((w * R[t, :]).sum()))
            t += 1
            step_since_refit += 1
    return np.array(out, dtype=float)


# ============== 主程序 ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet_assets", default="Assets_Returns")
    ap.add_argument("--sheet_index",  default="Index_Returns")

    # 数据与切分
    ap.add_argument("--split_ratio", type=float, default=0.5,
                    help="前多少比例样本作为 IS；剩余 OOS")
    ap.add_argument("--rebalance", type=int, default=1,
                    help="OOS 每多少期调仓（论文常用 1）")
    ap.add_argument("--standardize", action="store_true",
                    help="对每个序列（按时间维）做 z-score 标准化")

    # 模型结构
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--tau", type=float, default=1.0)

    # 训练
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_delta", type=float, default=0.995,
                    help="样本间 δ 衰减（0<δ<1）；None/<=0/>=1 则不加权")

    # 递归窗口开关（训练/推断/微调）
    ap.add_argument("--expanding_train", action="store_true",
                    help="训练用递归（扩张）窗口（默认开启）")
    ap.add_argument("--expanding_infer", action="store_true",
                    help="推断用递归（扩张）窗口（默认开启）")
    ap.add_argument("--expanding_refit", action="store_true",
                    help="在线微调用递归（扩张）窗口")
    ap.add_argument("--max_infer_len", type=int, default=0,
                    help="推断时递归窗口长度上限（0=不限制）")
    # Sharpe 口径
    ap.add_argument("--rf", type=float, default=0.0,
                    help="每期无风险利率（周频可用 (1+ry)^(1/52)-1）")
    ap.add_argument("--per_year", type=int, default=52)
    ap.add_argument("--annualize", action="store_true")

    args = ap.parse_args()

    # 默认开启递归训练/推断（更贴论文）
    if not args.expanding_train:
        args.expanding_train = True
    if not args.expanding_infer:
        args.expanding_infer = True

    # 读并对齐
    R_assets = load_sheet(args.excel, args.sheet_assets)
    R_index  = load_sheet(args.excel, args.sheet_index)
    if R_index.shape[1] != 1:
        R_index = R_index[:, [0]]
    R, r_idx = align_two(R_assets, R_index[:,0])
    T, N = R.shape
    print(f"Loaded Assets_Returns: T={T}, N={N}; Index_Returns: T={T}, N=1\n")

    # 切分
    split = int(T * args.split_ratio)
    if split < 2:
        raise ValueError("IS 样本太短，无法形成递归训练样本。")
    R_is = R[:split, :]
    R_oos = R[split:, :]
    start = split

    # 训练（递归）
    model = train_lstm_policy_expanding(
        R_train=R_is,
        epochs=args.epochs, lr=args.lr, rho=args.rho, rf=args.rf,
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        bidirectional=args.bidirectional, tau=args.tau,
        adaptive=args.adaptive, seed=args.seed,
        train_delta=(args.train_delta if (0.0 < args.train_delta < 1.0) else None),
        standardize=args.standardize
    )

    # OOS（递归推断 + 可选递归微调）
    lstm_r = oos_run(
        model, R, start=start, rebalance=args.rebalance,
        standardize=args.standardize,
        refit_every=0, refit_epochs=0,   # 如需在线微调，把这两个设成 >0，并打开 expanding_refit
        refit_horizon=260, refit_lr=5e-4, refit_rho=0.05,
        rf=args.rf, adaptive=args.adaptive,
        train_delta=(args.train_delta if (0.0 < args.train_delta < 1.0) else None),
        expanding_infer=args.expanding_infer, max_infer_len=args.max_infer_len,
        expanding_refit=args.expanding_refit
    )

    # 基线（对齐 OOS 窗口）
    end = start + len(lstm_r)
    ew_r = R[split:end, :].mean(axis=1)
    mi_r = r_idx[split:end]

    # Sharpe（可年化）
    sr_lstm = sharpe_ratio_np(lstm_r, rf=args.rf, per_year=args.per_year, annualize=args.annualize)
    sr_ew   = sharpe_ratio_np(ew_r,   rf=args.rf, per_year=args.per_year, annualize=args.annualize)
    sr_mi   = sharpe_ratio_np(mi_r,   rf=args.rf, per_year=args.per_year, annualize=args.annualize)

    title = f"SR (annualized={args.annualize}, per_year={args.per_year})  OOS=[{start},{end})"
    print("\n=== Sharpe Ratio Comparison (LSTM: expanding train/infer) ===")
    print(title)
    print("-" * len(title))
    print(f"LSTM   : {sr_lstm:.4f}")
    print(f"EW     : {sr_ew:.4f}")
    print(f"MI     : {sr_mi:.4f}")


if __name__ == "__main__":
    """
    推荐命令（论文口径：周频、每周再平衡、年化 SR；训练/推断均用递归窗口）：

    python blosam_sr_min_plus_v3.py \
      --excel /Users/sudan/Desktop/Code/BCST-Lib/Datasets/DowJones/DowJones.xlsx \
      --split_ratio 0.5 --rebalance 1 \
      --annualize --per_year 52 --rf 0.00038 \
      --standardize \
      --train_delta 0.995 \
      --epochs 80 --hidden 192 --rho 0.05 \
      --bidirectional --layers 2 --dropout 0.1 --tau 0.8 \
      --max_infer_len 1040
    """
    main()
