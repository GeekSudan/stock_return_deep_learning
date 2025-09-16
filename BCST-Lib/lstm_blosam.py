import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from blosam import BLOSAM  

WEEKS_PER_YEAR = 52


# =========================
# 工具：健壮性 & 复现
# =========================
def _to_np(a):
    return np.asarray(a, dtype=np.float64)

def _sanitize(a):
    a = _to_np(a)
    a[~np.isfinite(a)] = 0.0
    return a

def set_seed(seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 指标（基于超额或总收益）
# =========================
def sharpe_ratio(excess, annualize=False):
    ex = _sanitize(excess)
    sr = ex.mean() / (ex.std(ddof=1) + 1e-12)
    return float(sr * (WEEKS_PER_YEAR ** 0.5) if annualize else sr)

def sortino_ratio(excess, annualize=False):
    ex = _sanitize(excess)
    downside = np.clip(-ex, 0.0, None)  # 以0为门槛；对超额收益等价于以rf为门槛
    dstd = np.sqrt((downside ** 2).mean() + 1e-18)
    sor = ex.mean() / (dstd + 1e-12)
    return float(sor * (WEEKS_PER_YEAR ** 0.5) if annualize else sor)

def omega_ratio(excess):
    ex = _sanitize(excess)
    gains = np.clip(ex, 0.0, None)
    losses = np.clip(-ex, 0.0, None)
    num = gains.mean()
    den = losses.mean()
    if den <= 1e-18:
        return float('inf') if num > 0 else 0.0
    return float(num / den)


# ---------- CVaR (两种口径) ----------
def _var_tail(losses, alpha=0.95):
    L = _sanitize(losses)
    var = np.quantile(L, alpha)
    mask = (L >= var)
    return float(var), mask

def cvar_tailavg(excess, alpha=0.95):
    ex = _sanitize(excess)
    L = -ex
    var, mask = _var_tail(L, alpha)
    if not np.any(mask):
        return float(var)
    return float(L[mask].mean())

def cvar_ru(excess, alpha=0.95, z_grid=301):
    ex = _sanitize(excess)
    L = -ex
    m = L.size
    if m == 0: return 0.0

    lo_q = max(0.0, alpha - 0.1)
    hi_q = min(1.0, alpha)
    z_lo = np.quantile(L, lo_q) if m > 5 else L.min()
    z_hi = np.quantile(L, hi_q) if m > 5 else L.max()
    if z_hi <= z_lo: z_hi = z_lo + 1e-8

    Z = np.linspace(z_lo, z_hi, z_grid)
    denom = max(1e-12, (1.0 - alpha) * m)
    vals = Z + (np.maximum(L[:, None] - Z[None, :], 0.0).sum(axis=0) / denom)
    idx = int(np.argmin(vals))
    z_star = float(Z[idx])

    for _ in range(3):
        hinge = np.maximum(L - z_star, 0.0)
        cnt = float((L > z_star).sum())
        grad = 1.0 - cnt / max(1e-12, (1.0 - alpha) * m)
        step = 0.1 * max(1.0, abs(z_star))
        z_new = z_star - step * grad
        if not np.isfinite(z_new):
            break
        z_star = float(z_new)

    hinge = np.maximum(L - z_star, 0.0)
    cvar = z_star + hinge.sum() / denom
    return float(cvar)


# ---------- EVaR ----------
def evar_einf(excess, alpha=0.95):
    ex = _sanitize(excess)
    L = -ex
    if L.size == 0:
        return 0.0

    def _objective(t):
        if t <= 0: return np.inf
        z = L * t
        z_max = np.max(z)
        log_mgf = z_max + np.log(np.exp(z - z_max).mean() + 1e-300)
        return (log_mgf - math.log(max(1.0 - alpha, 1e-12))) / t

    ts = np.logspace(-6, 1, 200)
    vals = np.array([_objective(t) for t in ts])
    k = int(np.argmin(vals))
    t0 = ts[k]

    a = ts[max(0, k - 5)]
    c = ts[min(len(ts) - 1, k + 5)]
    if c <= a:
        a, c = max(1e-6, t0/10), min(10.0, t0*10)

    left, right = a, c
    for _ in range(30):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        f1, f2 = _objective(m1), _objective(m2)
        if f1 < f2:
            right = m2
        else:
            left = m1
        if right - left < 1e-6:
            break
    t_star = 0.5 * (left + right)
    return float(_objective(t_star))

def csrp(excess, alpha=0.95, method="ru"):
    mean_ex = _sanitize(excess).mean()
    cvar = cvar_ru(excess, alpha=alpha) if method == "ru" else cvar_tailavg(excess, alpha=alpha)
    return float(mean_ex / (cvar + 1e-12))

def esrp(excess, alpha=0.95):
    ev = evar_einf(excess, alpha=alpha)
    mean_ex = _sanitize(excess).mean()
    return float(mean_ex / (ev + 1e-12))


# =========================
# 模型：BetterPortfolioNet（均值-波动双头 + TCN + LSTM + MHA）
# =========================
class BetterPortfolioNet(nn.Module):
    """
    输入: (1, T, N) -> 输出: (1, T, N) 的 (mu, log_sigma)
    结构: 资产嵌入 + TemporalConv -> LSTM -> MHA -> 双头
    """
    def __init__(self, n_assets, hidden=256, num_layers=2, attn_heads=4, dropout=0.1, emb_dim=32, tcn_k=5):
        super().__init__()
        self.n_assets = n_assets
        self.emb = nn.Embedding(n_assets, emb_dim)
        self.asset_linear = nn.Linear(1, emb_dim)

        self.tcn = nn.Conv1d(in_channels=n_assets*emb_dim, out_channels=n_assets*emb_dim,
                             kernel_size=tcn_k, padding=tcn_k//2, groups=n_assets*emb_dim)

        self.proj_in = nn.Linear(n_assets*emb_dim, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=attn_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden)

        self.mean_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, n_assets)
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, n_assets)
        )

    def forward(self, x):  # x: (1, T, N)
        B, T, N = x.shape
        val_emb = self.asset_linear(x.unsqueeze(-1))  # (B,T,N,emb)
        idx = torch.arange(N, device=x.device).view(1,1,N).expand(B,T,N)
        asset_emb = self.emb(idx)                     # (B,T,N,emb)
        h = val_emb + asset_emb                       # (B,T,N,emb)
        h = h.reshape(B, T, N * asset_emb.size(-1))   # (B,T,N*emb)

        h_tcn = self.tcn(h.transpose(1,2)).transpose(1,2)  # (B,T,N*emb)
        h = h + h_tcn

        h = self.proj_in(h)           # (B,T,H)
        self.lstm.flatten_parameters()
        h, _ = self.lstm(h)           # (B,T,H)
        h = self.ln1(h)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln2(h + attn_out)

        mu = self.mean_head(h)             # (B,T,N)
        log_sigma = self.vol_head(h)       # (B,T,N)
        return mu, log_sigma


# =========================
# 训练目标：Gaussian NLL + Sharpe 代理
# =========================
def gaussian_nll_loss(pred_mu, pred_logsigma, target, min_logsig=-6.0, max_logsig=2.0):
    logsig = torch.clamp(pred_logsigma, min_logsig, max_logsig)
    inv_var = torch.exp(-2.0 * logsig)
    nll = 0.5 * ((target - pred_mu)**2 * inv_var + 2.0*logsig)  # 常数项忽略
    return nll.mean()

def sharpe_surrogate_loss_dist(mu_next, true_next, mode="softmax", temperature=6.0, eps=1e-8):
    if mode == "softmax":
        w = torch.softmax(temperature * mu_next, dim=-1)
    elif mode == "pos_norm":
        w = torch.clamp(mu_next, min=0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + eps)
    else:
        w = torch.ones_like(mu_next) / mu_next.size(-1)

    port_ret = (w * true_next).sum(dim=-1)  # (B,T-1)
    mean = port_ret.mean()
    std = port_ret.std(unbiased=True) + eps
    sr = mean / std
    return -sr  # 最小化 -SR == 最大化 SR


# # =========================
# # 预训练 & 扩展式微调（BLOSAM 两步）
# # =========================
def train_full_sequence(model, seq_np_std, device, optimizer, epochs=200, alpha_mse=0.7, temp=6.0, clip_norm=1.0, verbose=True):
    model.train()
    x = torch.from_numpy(seq_np_std[None, ...]).float().to(device)  # (1,T,N)

    for ep in range(1, epochs + 1):
        # first step
        mu_pred, logsig_pred = model(x)
        mu_pred  = mu_pred[:, :-1, :]
        logsig_pred = logsig_pred[:, :-1, :]
        target   = x[:, 1:, :]

        loss_nll = gaussian_nll_loss(mu_pred, logsig_pred, target)
        loss_sr  = sharpe_surrogate_loss_dist(mu_pred, target, mode="softmax", temperature=temp)
        loss = alpha_mse * loss_nll + (1.0 - alpha_mse) * loss_sr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.first_step(zero_grad=True)

        # second step
        mu_pred2, logsig_pred2 = model(x)
        mu_pred2  = mu_pred2[:, :-1, :]
        logsig_pred2 = logsig_pred2[:, :-1, :]
        loss_nll2 = gaussian_nll_loss(mu_pred2, logsig_pred2, target)
        loss_sr2  = sharpe_surrogate_loss_dist(mu_pred2, target, mode="softmax", temperature=temp)
        loss2 = alpha_mse * loss_nll2 + (1.0 - alpha_mse) * loss_sr2

        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.second_step(zero_grad=True)

        if verbose and (ep % max(1, epochs // 10) == 0):
            print(f"[Pretrain {ep:03d}] NLL={loss_nll.item():.6f}  SRloss={loss_sr.item():.6f}  (SR≈{-loss_sr.item():.6f})")

# def train_full_sequence(model, seq_np_std, device, optimizer,
#                         epochs=200, alpha_mse=0.7, temp=6.0,
#                         clip_norm=1.0, verbose=True):
#     """
#     SGDM 一步训练：整段序列（训练半段）喂入。
#     loss = alpha * Gaussian-NLL + (1-alpha) * (-Sharpe)
#     """
#     model.train()
#     x = torch.from_numpy(seq_np_std[None, ...]).float().to(device)  # (1,T,N)

#     for ep in range(1, epochs + 1):
#         mu_pred, logsig_pred = model(x)
#         mu_pred      = mu_pred[:, :-1, :]
#         logsig_pred  = logsig_pred[:, :-1, :]
#         target       = x[:, 1:, :]

#         loss_nll = gaussian_nll_loss(mu_pred, logsig_pred, target)
#         loss_sr  = sharpe_surrogate_loss_dist(mu_pred, target, mode="softmax", temperature=temp)
#         loss = alpha_mse * loss_nll + (1.0 - alpha_mse) * loss_sr

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
#         optimizer.step()

#         if verbose and (ep % max(1, epochs // 10) == 0):
#             print(f"[Pretrain {ep:03d}] NLL={loss_nll.item():.6f}  SRloss={loss_sr.item():.6f}  (SR≈{-loss_sr.item():.6f})")

# def finetune_expand_until(model, full_assets_std_np, end_t, device, optimizer,
#                           finetune_epochs=2, alpha_mse=0.7, temp=6.0,
#                           lr_override=None, clip_norm=1.0, verbose=False):
#     """
#     SGDM 一步微调：用 [0, end_t) 的全部历史做少量 epoch 微调
#     """
#     if end_t < 2 or finetune_epochs <= 0:
#         return
#     model.train()

#     # 暂时性更小 lr（可选）
#     old_lrs = [g.get('lr', None) for g in optimizer.param_groups]
#     if lr_override is not None:
#         for g in optimizer.param_groups:
#             g['lr'] = lr_override

#     x = torch.from_numpy(full_assets_std_np[:end_t][None, ...]).float().to(device)

#     for ep in range(finetune_epochs):
#         mu_pred, logsig_pred = model(x)
#         mu_pred      = mu_pred[:, :-1, :]
#         logsig_pred  = logsig_pred[:, :-1, :]
#         target       = x[:, 1:, :]

#         loss_nll = gaussian_nll_loss(mu_pred, logsig_pred, target)
#         loss_sr  = sharpe_surrogate_loss_dist(mu_pred, target, mode="softmax", temperature=temp)
#         loss = alpha_mse * loss_nll + (1.0 - alpha_mse) * loss_sr

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
#         optimizer.step()

#         if verbose:
#             print(f"  [Finetune@{end_t} ep{ep+1}] NLL={loss_nll.item():.6f}  SRloss={loss_sr.item():.6f}  (SR≈{-loss_sr.item():.6f})")

#     # 恢复 lr
#     if lr_override is not None:
#         for g, old in zip(optimizer.param_groups, old_lrs):
#             if old is not None:
#                 g['lr'] = old


def finetune_expand_until(model, full_assets_std_np, end_t, device, optimizer, finetune_epochs=2, alpha_mse=0.7, temp=6.0, lr_override=None, clip_norm=1.0, verbose=False):
    if end_t < 2 or finetune_epochs <= 0:
        return
    model.train()

    old_lrs = [g.get('lr', None) for g in optimizer.param_groups]
    if lr_override is not None:
        for g in optimizer.param_groups:
            g['lr'] = lr_override

    x = torch.from_numpy(full_assets_std_np[:end_t][None, ...]).float().to(device)
    for ep in range(finetune_epochs):
        mu_pred, logsig_pred = model(x)
        mu_pred  = mu_pred[:, :-1, :]
        logsig_pred = logsig_pred[:, :-1, :]
        target   = x[:, 1:, :]

        loss_nll = gaussian_nll_loss(mu_pred, logsig_pred, target)
        loss_sr  = sharpe_surrogate_loss_dist(mu_pred, target, mode="softmax", temperature=temp)
        loss = alpha_mse * loss_nll + (1.0 - alpha_mse) * loss_sr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.first_step(zero_grad=True)

        mu_pred2, logsig_pred2 = model(x)
        mu_pred2  = mu_pred2[:, :-1, :]
        logsig_pred2 = logsig_pred2[:, :-1, :]
        loss_nll2 = gaussian_nll_loss(mu_pred2, logsig_pred2, target)
        loss_sr2  = sharpe_surrogate_loss_dist(mu_pred2, target, mode="softmax", temperature=temp)
        loss2 = alpha_mse * loss_nll2 + (1.0 - alpha_mse) * loss_sr2

        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.second_step(zero_grad=True)

        if verbose:
            print(f"  [Finetune@{end_t} ep{ep+1}] NLL={loss_nll.item():.6f}  SRloss={loss_sr.item():.6f}  (SR≈{-loss_sr.item():.6f})")

    if lr_override is not None:
        for g, old in zip(optimizer.param_groups, old_lrs):
            if old is not None:
                g['lr'] = old


# =========================
# 配权：EWMA 协方差 + 波动头自适应缩放 + 单纯形投影
# =========================
def ewma_cov(X, span=20, eps=1e-6):
    X = _sanitize(X)
    lam = math.exp(-1.0 / max(span, 1))
    mu = np.zeros(X.shape[1], dtype=np.float64)
    cov = np.eye(X.shape[1], dtype=np.float64) * 1e-6
    for t in range(X.shape[0]):
        x = X[t]
        mu = lam * mu + (1 - lam) * x
        d = (x - mu)[..., None]
        cov = lam * cov + (1 - lam) * (d @ d.T)
    cov += np.eye(X.shape[1]) * eps
    return cov

def project_to_simplex(v):
    v = np.maximum(_sanitize(v), 0.0)
    s = v.sum()
    if s <= 1e-12:
        return None
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0]
    if rho.size == 0:
        return None
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w

def mv_weight(mu_hat, hist_returns, span=20, sigma_hat=None):
    Sigma = ewma_cov(hist_returns, span=span)
    if sigma_hat is not None:
        sigma_hat = np.clip(_sanitize(sigma_hat), 1e-6, 10.0)
        D = np.diag(sigma_hat)
        Sigma = D @ Sigma @ D
    try:
        inv = np.linalg.pinv(Sigma)
        raw = inv @ mu_hat
    except Exception:
        raw = mu_hat.copy()
    w = project_to_simplex(raw)
    return w


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_path", type=str, required=True)
    ap.add_argument("--assets_sheet", type=str, default="assets")
    ap.add_argument("--index_sheet", type=str, default="index")
    ap.add_argument("--tbill_path", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adaptive", action="store_true")

    ap.add_argument("--alpha_mse", type=float, default=0.7, help="NLL vs SR surrogate mixing: alpha*NLL + (1-alpha)*SR")
    ap.add_argument("--temp", type=float, default=6.0, help="softmax temperature for SR surrogate")
    ap.add_argument("--clip_norm", type=float, default=1.0)

    ap.add_argument("--yearly_tbill", action="store_true", help="若 txt 中是年化 3M T-bill，则转换为周频")
    ap.add_argument("--alpha", type=float, default=0.95, help="CVaR/EVaR 置信度")

    # 论文口径/诊断
    ap.add_argument("--annualize_sr_sor", action="store_true", help="年化 SR/SoR（×sqrt(52)）")
    ap.add_argument("--annualize_csr_esr", action="store_true", help="年化 CSR/ESR（一般不年化）")
    ap.add_argument("--scale_pct", action="store_true", help="指标计算前收益×100（百分比点口径）")
    ap.add_argument("--risk_on_total", action="store_true", help="风险分母用总收益R（而非超额R-rf）")
    ap.add_argument("--cvar_method", type=str, default="ru", choices=["ru","tail"], help="CVaR 计算方法")

    ap.add_argument("--finetune_epochs", type=int, default=2, help="测试期每周扩展式微调轮数（0 关闭）")
    ap.add_argument("--finetune_lr", type=float, default=1e-4, help="微调时临时使用的更小学习率")
    ap.add_argument("--cov_window", type=int, default=200, help="估计协方差的历史窗口长度")
    ap.add_argument("--cov_span", type=int, default=20, help="EWMA 衰减跨度")

    ap.add_argument("--drop_last_if_odd", action="store_true", help="奇数周丢最后一周，严格 half-half")
    ap.add_argument("--standardize", action="store_true", help="训练/微调使用训练期统计量做标准化")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 读取数据
    df_assets = pd.read_excel(args.excel_path, sheet_name=args.assets_sheet, engine="openpyxl")
    df_index  = pd.read_excel(args.excel_path, sheet_name=args.index_sheet, engine="openpyxl")
    assets = df_assets.values.astype(np.float64)                 # (T, N)
    index_ret = df_index.values.squeeze().astype(np.float64)     # (T,)

    with open(args.tbill_path, "r", encoding="utf-8") as f:
        rf_vals = np.array([float(x) for x in f.read().strip().split(",")], dtype=np.float64)  # (T,)

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

    # 切分（绩效计算用原始收益）
    seq_train = assets[:T_half, :]
    seq_test  = assets[T_half:, :]
    rf_test   = rf_vals[T_half:]
    mi_test   = index_ret[T_half:]

    # 标准化（训练/微调输入）
    if args.standardize:
        mu_train = seq_train.mean(axis=0, keepdims=True)
        std_train = seq_train.std(axis=0, ddof=1, keepdims=True)
        std_train = np.where(std_train < 1e-6, 1.0, std_train)
        assets_std = (assets - mu_train) / std_train
        seq_train_std = assets_std[:T_half, :]
        seq_test_std  = assets_std[T_half:, :]
    else:
        mu_train = np.zeros((1, N))
        std_train = np.ones((1, N))
        seq_train_std = seq_train.copy()
        seq_test_std  = seq_test.copy()
        assets_std    = np.vstack([seq_train_std, seq_test_std])

    # 模型 + BLOSAM
    model = BetterPortfolioNet(
        n_assets=N,
        hidden=max(128, args.hidden),
        num_layers=max(2, args.layers),
        attn_heads=args.attn_heads,
        dropout=max(0.05, args.dropout),
        emb_dim=32,
        tcn_k=5
    ).to(device)

    optimizer = BLOSAM(
        model.parameters(),
        lr=args.lr,
        rho=args.rho,
        p=2,
        xi_lr_ratio=3,
        momentum_theta=0.9,
        weight_decay=0.0005,
        adaptive=args.adaptive
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.lr,
    #     momentum=0.9,
    #     weight_decay=0.0005
    # )
    # 预训练（完整训练半段，整段序列喂入）
    train_full_sequence(
        model=model,
        seq_np_std=seq_train_std,
        device=device,
        optimizer=optimizer,
        epochs=args.epochs,
        alpha_mse=args.alpha_mse,
        temp=args.temp,
        clip_norm=args.clip_norm,
        verbose=True
    )

    # ===== 测试期：逐周（t-1 预测 -> t 权重） + 可选扩展式微调 =====
    ew_w = np.ones(N) / N
    lstm_port, ew_port, mi_port, rf_used = [], [], [], []

    # 无微调初值：整段测试跑一次
    model.eval()
    with torch.no_grad():
        x_test_std = torch.from_numpy(seq_test_std[None, ...]).float().to(device)
        mu_test_std, logsig_test_std = model(x_test_std)
        mu_test_std = mu_test_std.squeeze(0).cpu().numpy()         # (T_test, N)
        logsig_test_std = logsig_test_std.squeeze(0).cpu().numpy() # (T_test, N)
    mu_test = mu_test_std * std_train + mu_train
    sigma_test = np.exp(logsig_test_std) * std_train  # 标准化逆变换的近似（只用于尺度参考）

    for t in range(1, seq_test.shape[0]):
        # 扩展式微调：用截止 t-1 的历史
        if args.finetune_epochs > 0:
            finetune_expand_until(
                model=model,
                full_assets_std_np=assets_std,
                end_t=T_half + t,
                device=device,
                optimizer=optimizer,
                finetune_epochs=args.finetune_epochs,
                alpha_mse=args.alpha_mse,
                temp=args.temp,
                lr_override=args.finetune_lr,
                clip_norm=args.clip_norm,
                verbose=False
            )
            model.eval()
            with torch.no_grad():
                hist_std = assets_std[:T_half + t]
                x_hist = torch.from_numpy(hist_std[None, ...]).float().to(device)
                mu_hist_std, logsig_hist_std = model(x_hist)
                mu_prev = (mu_hist_std.squeeze(0)[-1].cpu().numpy()) * std_train + mu_train  # (N,)
                sigma_prev = np.exp(logsig_hist_std.squeeze(0)[-1].cpu().numpy()) * std_train
                mu_prev = mu_prev.squeeze()
                sigma_prev = sigma_prev.squeeze()
        else:
            mu_prev = mu_test[t-1].squeeze()
            sigma_prev = sigma_test[t-1].squeeze()

        # 均值-方差权重（协方差用最近窗口；用波动头做 D Σ D 校正）
        start_hist = max(0, T_half + t - args.cov_window)
        hist_win = assets[start_hist: T_half + t]
        w = mv_weight(mu_hat=mu_prev, hist_returns=hist_win, span=args.cov_span, sigma_hat=sigma_prev)
        if w is None:
            w = ew_w

        # 实现收益
        realized = seq_test[t]
        lstm_port.append(float(np.dot(w, realized)))
        ew_port.append(float(np.dot(ew_w, realized)))
        mi_port.append(float(mi_test[t]))
        rf_used.append(float(rf_test[t]))

    lstm_port = _sanitize(lstm_port)
    ew_port   = _sanitize(ew_port)
    mi_port   = _sanitize(mi_port)
    rf_used   = _sanitize(rf_used)

    # === 口径：超额 or 总收益 ===
    if args.risk_on_total:
        base_lstm = lstm_port
        base_ew   = ew_port
        base_mi   = mi_port
    else:
        base_lstm = lstm_port - rf_used
        base_ew   = ew_port   - rf_used
        base_mi   = mi_port   - rf_used

    # === 百分比点口径 ===
    scale = 100.0 if args.scale_pct else 1.0
    ex_lstm = base_lstm * scale
    ex_ew   = base_ew   * scale
    ex_mi   = base_mi   * scale

    # === 指标 ===
    print("\n=== Ex-post risk-adjusted metrics on TEST ===")
    for name, ex in [("LSTM+BLOSAM", ex_lstm), ("EW", ex_ew), ("MI", ex_mi)]:
        sr  = sharpe_ratio(ex, annualize=args.annualize_sr_sor)
        sor = sortino_ratio(ex, annualize=args.annualize_sr_sor)
        orr = omega_ratio(ex)
        csr = csrp(ex, alpha=args.alpha, method=args.cvar_method)
        esr = esrp(ex, alpha=args.alpha)

        csr_out = csr
        esr_out = esr
        if args.annualize_csr_esr:
            csr_out *= math.sqrt(WEEKS_PER_YEAR)
            esr_out *= math.sqrt(WEEKS_PER_YEAR)

        cvar_tail_val = cvar_tailavg(ex, alpha=args.alpha)
        cvar_ru_val   = cvar_ru(ex, alpha=args.alpha)
        evar_val      = evar_einf(ex, alpha=args.alpha)

        print(f"\n{name}:")
        print(f"  mean(base) = {np.mean(ex):.6g}   std = {np.std(ex, ddof=1):.6g}")
        print(f"  SRp  = {sr:.6g}{' (annualized)' if args.annualize_sr_sor else ''}")
        print(f"  SoRp = {sor:.6g}{' (annualized)' if args.annualize_sr_sor else ''}")
        print(f"  OR   = {orr:.6g}")
        print(f"  CSRp = {csr_out:.6g}{' (annualized)' if args.annualize_csr_esr else ''}   "
              f"[CVaR_tail={cvar_tail_val:.6g}, CVaR_RU={cvar_ru_val:.6g}]")
        print(f"  ESRp = {esr_out:.6g}{' (annualized)' if args.annualize_csr_esr else ''}   "
              f"[EVaR={evar_val:.6g}]")

    # 保存结果
    out = pd.DataFrame({
        "lstm_port": lstm_port,
        "ew_port": ew_port,
        "mi_port": mi_port,
        "rf": rf_used,
        "base_lstm": base_lstm,
        "base_ew": base_ew,
        "base_mi": base_mi,
        "excess_lstm_final": ex_lstm,
        "excess_ew_final": ex_ew,
        "excess_mi_final": ex_mi
    })
    out.to_csv("test_results_lstm_blosam.csv", index=False)
    print("\nSaved: test_results_lstm_blosam.csv")


if __name__ == "__main__":
    main()
