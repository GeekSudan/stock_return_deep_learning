# -*- coding: utf-8 -*-
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re

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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# 与 ND 脚本完全一致的后验指标器
# =========================
def ex_post_metrics_generic(returns, rf=None, mode="weekly",
                            threshold="zero", periods_per_year=52, theta=0.95):
    """
    和 ND 脚本对齐：
      - threshold='zero'   : 传入的 returns 被视为“超额”序列，直接评估
      - threshold='rf'     : 传入的 returns 被视为“总收益”序列，内部减 rf 再评估
      - mode='annualized'  : 对 SR/SoR/Ω/CSR/ESR 的处理方式与 ND 一致（结果=周比值×sqrt(52)）
    """
    r = _sanitize(returns)
    if threshold == "rf":
        if rf is None:
            raise ValueError("eval_threshold=rf 需要传入 rf 序列")
        exc = r - _sanitize(rf)
    else:
        exc = r

    mu_w = exc.mean()
    std_w = exc.std(ddof=1)
    sr_w = mu_w / (std_w + 1e-12)

    dn = np.maximum(-exc, 0.0)
    sdn_w = math.sqrt((dn**2).mean() + 1e-12)
    sor_w = mu_w / (sdn_w + 1e-12)

    lpm1_w = dn.mean() + 1e-12
    omega_w = mu_w / lpm1_w + 1.0

    m = len(exc)
    losses = -exc
    q = int(math.ceil((1.0 - theta) * m))
    if q > 0:
        worst = np.partition(losses, -q)[-q:]
        cvar_w = worst.mean() + 1e-12
    else:
        cvar_w = max(losses.max(), 0.0) + 1e-12
    csr_w = mu_w / cvar_w

    # EVaR via log-sum-exp（与 ND 相同）
    best = None
    for rho in np.logspace(-3, 0, 40):  # 1e-3..1
        logits = -exc / max(rho, 1e-6)
        mlog = np.max(logits)
        lse = mlog + math.log(np.exp(logits - mlog).sum() + 1e-12)
        logS = lse - math.log(m*(1.0-theta) + 1e-12)
        val = rho * logS
        best = val if (best is None or val < best) else best
    evar_w = (best if best is not None else 0.0) + 1e-12
    esr_w = mu_w / evar_w

    if mode == "weekly":
        return dict(SRp=sr_w, SoRp=sor_w, ORp=omega_w, CSRp=csr_w, ESRp=esr_w)

    # annualized：与 ND 一致（等价于周频比值×sqrt(periods_per_year)）
    mu_a  = mu_w * periods_per_year
    std_a = std_w * math.sqrt(periods_per_year)
    sdn_a = sdn_w * math.sqrt(periods_per_year)
    lpm1_a = lpm1_w * math.sqrt(periods_per_year)
    cvar_a = cvar_w * math.sqrt(periods_per_year)
    evar_a = evar_w * math.sqrt(periods_per_year)
    return dict(
        SRp = mu_a/(std_a+1e-12),
        SoRp= mu_a/(sdn_a+1e-12),
        ORp = mu_a/(lpm1_a+1e-12) + 1.0,
        CSRp= mu_a/(cvar_a+1e-12),
        ESRp= mu_a/(evar_a+1e-12),
    )

# =========================
# 模型：BetterPortfolioNet
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
# 训练目标（代理）与辅助
# =========================
def gaussian_nll_loss(pred_mu, pred_logsigma, target, min_logsig=-6.0, max_logsig=2.0):
    logsig = torch.clamp(pred_logsigma, min_logsig, max_logsig)
    inv_var = torch.exp(-2.0 * logsig)
    nll = 0.5 * ((target - pred_mu)**2 * inv_var + 2.0*logsig)
    return nll.mean()

def _portfolio_returns_from_mu(mu_next, true_next, mode="softmax", temperature=6.0, eps=1e-8):
    if mode == "softmax":
        w = torch.softmax(temperature * mu_next, dim=-1)
    elif mode == "pos_norm":
        w = torch.clamp(mu_next, min=0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + eps)
    else:
        w = torch.ones_like(mu_next) / mu_next.size(-1)
    port_ret = (w * true_next).sum(dim=-1)  # (B,T-1)
    return port_ret

def sr_proxy_loss(r_port, eps=1e-8):
    mean = r_port.mean()
    std  = r_port.std(unbiased=True) + eps
    return -(mean / std)

def sortino_proxy_loss(r_port, eps=1e-8):
    downside = F.relu(-r_port)
    dstd = torch.sqrt((downside**2).mean() + eps)
    return -(r_port.mean() / (dstd + eps))

def omega_proxy_loss(r_port, eps=1e-8):
    gains  = F.relu(r_port)
    losses = F.relu(-r_port)
    num = gains.mean()
    den = losses.mean() + eps
    return -(num / den)

def cvar_ru_torch(r_port, alpha=0.95, iters=10, lr=0.1, smooth=True):
    L = -r_port
    if L.numel() == 0:
        return torch.tensor(0.0, device=L.device)
    with torch.no_grad():
        z0 = torch.quantile(L.flatten(), alpha)
    z = torch.tensor(float(z0.item()), device=L.device, requires_grad=True)
    for _ in range(iters):
        hinge = F.softplus(L - z, beta=5.0) if smooth else F.relu(L - z)
        obj = z + hinge.mean() / max(1e-12, (1.0 - alpha))
        g = torch.autograd.grad(obj, z, retain_graph=True)[0]
        with torch.no_grad():
            z -= lr * g
    hinge = F.softplus(L - z, beta=5.0) if smooth else F.relu(L - z)
    cvar = z + hinge.mean() / max(1e-12, (1.0 - alpha))
    return cvar

def evar_torch(r_port, alpha=0.95, iters=20, lr=0.1):
    L = -r_port
    s = torch.tensor(-2.0, device=L.device, requires_grad=True)  # t = exp(s) > 0
    for _ in range(iters):
        t = torch.exp(s)
        z = t * L
        z_max = torch.max(z.detach())
        log_mgf = z_max + torch.log(torch.mean(torch.exp(z - z_max)) + 1e-12)
        obj = (log_mgf - math.log(max(1.0 - alpha, 1e-12))) / (t + 1e-12)
        g = torch.autograd.grad(obj, s, retain_graph=True)[0]
        with torch.no_grad():
            s -= lr * g
    t = torch.exp(s)
    z = t * L
    z_max = torch.max(z.detach())
    log_mgf = z_max + torch.log(torch.mean(torch.exp(z - z_max)) + 1e-12)
    evar = (log_mgf - math.log(max(1.0 - alpha, 1e-12))) / (t + 1e-12)
    return evar

def csr_proxy_loss(r_port, alpha=0.95, iters=10, lr=0.1):
    cvar = cvar_ru_torch(r_port, alpha=alpha, iters=iters, lr=lr, smooth=True)
    mean = r_port.mean()
    return -(mean / (cvar + 1e-12))

def esr_proxy_loss(r_port, alpha=0.95, iters=20, lr=0.1):
    ev = evar_torch(r_port, alpha=alpha, iters=iters, lr=lr)
    mean = r_port.mean()
    return -(mean / (ev + 1e-12))

def multiobjective_surrogate_loss(
    mu_next, true_next,
    mode="softmax", temperature=6.0, alpha_cvar_evar=0.95,
    w_sr=1.0, w_sor=0.0, w_omega=0.0, w_csr=0.0, w_esr=0.0,
    iters_cvar=10, iters_evar=20, lr_cvar=0.1, lr_evar=0.1,
    eps=1e-8
):
    r_port = _portfolio_returns_from_mu(mu_next, true_next, mode=mode, temperature=temperature, eps=eps)
    loss = torch.tensor(0.0, device=r_port.device)
    if w_sr   > 0: loss = loss + w_sr   * sr_proxy_loss(r_port, eps=eps)
    if w_sor  > 0: loss = loss + w_sor  * sortino_proxy_loss(r_port, eps=eps)
    if w_omega> 0: loss = loss + w_omega* omega_proxy_loss(r_port, eps=eps)
    if w_csr  > 0: loss = loss + w_csr  * csr_proxy_loss(r_port, alpha=alpha_cvar_evar, iters=iters_cvar, lr=lr_cvar)
    if w_esr  > 0: loss = loss + w_esr  * esr_proxy_loss(r_port, alpha=alpha_cvar_evar, iters=iters_evar, lr=lr_evar)
    return loss

def _one_epoch(model, x, optimizer, args, loss_weights):
    """返回 (loss_nll, loss_proxy, total_loss)。"""
    mu_pred, logsig_pred = model(x)
    mu_pred     = mu_pred[:, :-1, :]
    logsig_pred = logsig_pred[:, :-1, :]
    target      = x[:, 1:, :]

    loss_nll = gaussian_nll_loss(mu_pred, logsig_pred, target)
    loss_proxy = multiobjective_surrogate_loss(
        mu_next=mu_pred, true_next=target,
        mode="softmax", temperature=args.temp, alpha_cvar_evar=args.alpha,
        w_sr=loss_weights["w_sr"], w_sor=loss_weights["w_sor"], w_omega=loss_weights["w_omega"],
        w_csr=loss_weights["w_csr"], w_esr=loss_weights["w_esr"],
        iters_cvar=10, iters_evar=20, lr_cvar=0.1, lr_evar=0.1
    )
    total = args.alpha_mse * loss_nll + (1.0 - args.alpha_mse) * loss_proxy
    return mu_pred, target, loss_nll, loss_proxy, total

# =========================
# 训练 & 微调
# =========================
def train_full_sequence(model, seq_np_std, device, optimizer, args, loss_weights, verbose=True):
    model.train()
    x = torch.from_numpy(seq_np_std[None, ...]).float().to(device)  # (1,T,N)
    use_blosam = (args.optim == "blosam")
    for ep in range(1, args.epochs + 1):
        if use_blosam:
            from blosam import BLOSAM  # 确保已安装
            mu1, target, loss_nll, loss_proxy, total = _one_epoch(model, x, optimizer, args, loss_weights)
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.first_step(zero_grad=True)
            mu2, target2, loss_nll2, loss_proxy2, total2 = _one_epoch(model, x, optimizer, args, loss_weights)
            total2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.second_step(zero_grad=True)
            mu_for_sr = mu2; show_proxy = loss_proxy
        else:
            mu1, target, loss_nll, loss_proxy, total = _one_epoch(model, x, optimizer, args, loss_weights)
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            mu_for_sr = mu1; show_proxy = loss_proxy

        if verbose and (ep % max(1, args.epochs // 10) == 0):
            approx_sr = -sr_proxy_loss(_portfolio_returns_from_mu(mu_for_sr, target, temperature=args.temp)).item()
            tag = "BLOSAM" if use_blosam else "SGDM"
            print(f"[Pretrain {ep:03d} {tag}] NLL={loss_nll.item():.6f}  Proxy={show_proxy.item():.6f}  (SR≈{approx_sr:.6f})")

def finetune_expand_until(model, full_assets_std_np, end_t, device, optimizer, args, loss_weights, verbose=False):
    if end_t < 2 or args.finetune_epochs <= 0:
        return
    model.train()
    old_lrs = [g.get('lr', None) for g in optimizer.param_groups]
    if args.finetune_lr is not None:
        for g in optimizer.param_groups: g['lr'] = args.finetune_lr
    x = torch.from_numpy(full_assets_std_np[:end_t][None, ...]).float().to(device)
    use_blosam = (args.optim == "blosam")
    for ep in range(args.finetune_epochs):
        if use_blosam:
            from blosam import BLOSAM
            mu1, target, loss_nll, loss_proxy, total = _one_epoch(model, x, optimizer, args, loss_weights)
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.first_step(zero_grad=True)
            mu2, target2, loss_nll2, loss_proxy2, total2 = _one_epoch(model, x, optimizer, args, loss_weights)
            total2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.second_step(zero_grad=True)
            mu_for_sr = mu2
        else:
            mu1, target, loss_nll, loss_proxy, total = _one_epoch(model, x, optimizer, args, loss_weights)
            optimizer.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            mu_for_sr = mu1
        if verbose:
            approx_sr = -sr_proxy_loss(_portfolio_returns_from_mu(mu_for_sr, target, temperature=args.temp)).item()
            tag = "BLOSAM" if use_blosam else "SGDM"
            print(f"  [Finetune@{end_t} ep{ep+1} {tag}] NLL={loss_nll.item():.6f}  Proxy={loss_proxy.item():.6f}  (SR≈{approx_sr:.6f})")
    if args.finetune_lr is not None:
        for g, old in zip(optimizer.param_groups, old_lrs):
            if old is not None: g['lr'] = old

# =========================
# 配权：EWMA 协方差 + 预白化 + 单纯形投影 (+ 可选上限)
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

def cap_and_project(w, cap=0.0):
    if cap is None or cap <= 0:
        return w
    w = np.clip(w, 0.0, cap)
    s = w.sum()
    if s <= 1e-12:
        return np.ones_like(w) / w.size
    return w / s

def mv_weight(mu_hat, hist_returns, span=20, sigma_hat=None, ridge=1e-2, cap=0.0):
    Sigma = ewma_cov(hist_returns, span=span, eps=ridge)
    mu = _sanitize(mu_hat).copy()
    if sigma_hat is not None:
        s = np.clip(_sanitize(sigma_hat), 1e-6, 1e6)
        Dinv = np.diag(1.0 / s)
        Sigma = Dinv @ Sigma @ Dinv
        mu    = (mu / s)
    if ridge > 0:
        Sigma = Sigma + ridge * np.eye(Sigma.shape[0])
    try:
        w_raw = np.linalg.pinv(Sigma) @ mu
    except Exception:
        w_raw = mu.copy()
    w = project_to_simplex(w_raw)
    if w is None:
        w = np.ones_like(mu) / mu.size
    w = cap_and_project(w, cap=cap)
    return w

# =========================
# 逐个指标：训练 + 测试（核心封装）
# =========================
def run_one_metric(metric_name, data_pack, args):
    """
    metric_name: 'SR' / 'SoR' / 'OR' / 'CSR' / 'ESR'
    data_pack: (assets, index_ret, rf_vals, seq_train, seq_test, rf_test, mi_test, seq_train_std, seq_test_std,
                assets_std, mu_train, std_train, N, T_half)
    """
    (assets, index_ret, rf_vals,
     seq_train, seq_test, rf_test, mi_test,
     seq_train_std, seq_test_std, assets_std, mu_train, std_train, N, T_half) = data_pack

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 为当前 metric 设定代理损失权重（只开对应一项）
    wset = dict(w_sr=0.0, w_sor=0.0, w_omega=0.0, w_csr=0.0, w_esr=0.0)
    if   metric_name == "SR":  wset["w_sr"]  = 1.0
    elif metric_name == "SoR": wset["w_sor"] = 1.0
    elif metric_name == "OR":  wset["w_omega"]=1.0
    elif metric_name == "CSR": wset["w_csr"] = 1.0
    elif metric_name == "ESR": wset["w_esr"] = 1.0
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    print(f"\n========== Training for target metric: {metric_name} ==========")

    # 2) 模型 & 优化器
    model = BetterPortfolioNet(
        n_assets=N,
        hidden=max(128, args.hidden),
        num_layers=max(2, args.layers),
        attn_heads=args.attn_heads,
        dropout=max(0.05, args.dropout),
        emb_dim=32,
        tcn_k=5
    ).to(device)

    if args.optim == "blosam":
        try:
            from blosam import BLOSAM
        except Exception as e:
            raise ImportError(
                f"选择了 --optim blosam，但未能导入 BLOSAM：{e}\n"
                f"请安装/确保 blosam 可用，或改用 --optim sgdm。"
            )
        optimizer = BLOSAM(
            model.parameters(),
            lr=args.lr, rho=args.rho, p=2, xi_lr_ratio=3,
            momentum_theta=args.momentum, weight_decay=args.weight_decay,
            adaptive=args.adaptive
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
        )

    # 3) 预训练（完整训练半段，整段序列喂入）
    train_full_sequence(
        model=model,
        seq_np_std=seq_train_std,
        device=device,
        optimizer=optimizer,
        args=args,
        loss_weights=wset,
        verbose=True
    )

    # 4) 测试期：逐周（t-1 预测 -> t 权重） + 可选扩展式微调（按同一 metric 权重）
    ew_w = np.ones(N) / N
    lstm_port, lstm_soft_port, ew_port, mi_port, rf_used = [], [], [], [], []

    # 先跑一次整段测试，便于无微调时拿到预测
    model.eval()
    with torch.no_grad():
        x_test_std = torch.from_numpy(seq_test_std[None, ...]).float().to(device)
        mu_test_std, logsig_test_std = model(x_test_std)
        mu_test_std = mu_test_std.squeeze(0).cpu().numpy()         # (T_test, N)
        logsig_test_std = logsig_test_std.squeeze(0).cpu().numpy() # (T_test, N)
    mu_test = mu_test_std * std_train + mu_train
    sigma_test = np.exp(logsig_test_std) * std_train

    for t in range(1, seq_test.shape[0]):
        # 扩展式微调：用截止 t-1 的历史
        if args.finetune_epochs > 0:
            finetune_expand_until(
                model=model,
                full_assets_std_np=assets_std,
                end_t=T_half + t,
                device=device,
                optimizer=optimizer,
                args=args,
                loss_weights=wset,
                verbose=False
            )
            model.eval()
            with torch.no_grad():
                hist_std = assets_std[:T_half + t]
                x_hist = torch.from_numpy(hist_std[None, ...]).float().to(device)
                mu_hist_std, logsig_hist_std = model(x_hist)
                mu_prev = (mu_hist_std.squeeze(0)[-1].cpu().numpy()) * std_train + mu_train
                sigma_prev = np.exp(logsig_hist_std.squeeze(0)[-1].cpu().numpy()) * std_train
                mu_prev = mu_prev.squeeze()
                sigma_prev = sigma_prev.squeeze()
        else:
            mu_prev = mu_test[t-1].squeeze()
            sigma_prev = sigma_test[t-1].squeeze()

        # Softmax 权重（稳定）
        tau_mu = args.temp * mu_prev
        tau_mu = tau_mu - np.max(tau_mu)
        w_soft = np.exp(tau_mu); w_soft = w_soft / (w_soft.sum() + 1e-12)

        # MV 权重（预白化 + ridge + cap）
        start_hist = max(0, T_half + t - args.cov_window)
        hist_win = assets[start_hist: T_half + t]
        w_mv = mv_weight(
            mu_hat=mu_prev,
            hist_returns=hist_win,
            span=args.cov_span,
            sigma_hat=sigma_prev,
            ridge=args.mv_ridge,
            cap=args.mv_cap
        )
        if w_mv is None: w_mv = ew_w

        realized = seq_test[t]
        lstm_port.append(float(np.dot(w_mv, realized)))            # LSTM+MV（你主用）
        lstm_soft_port.append(float(np.dot(w_soft, realized)))     # LSTM+SOFTMAX（对照）
        ew_port.append(float(np.dot(ew_w, realized)))
        mi_port.append(float(mi_test[t]))
        rf_used.append(float(rf_test[t]))

    lstm_port      = _sanitize(lstm_port)
    lstm_soft_port = _sanitize(lstm_soft_port)
    ew_port        = _sanitize(ew_port)
    mi_port        = _sanitize(mi_port)
    rf_used        = _sanitize(rf_used)

    # === 和 ND 完全一致的评估入口 ===
    # 如果 eval_threshold=='zero'：这里就传“超额”序列；如果 'rf'：这里就传“总收益”序列
    if args.eval_threshold == "rf":
        base_lstm      = lstm_port
        base_lstm_soft = lstm_soft_port
        base_ew        = ew_port
        base_mi        = mi_port
    else:  # 'zero'
        base_lstm      = lstm_port      - rf_used
        base_lstm_soft = lstm_soft_port - rf_used
        base_ew        = ew_port        - rf_used
        base_mi        = mi_port        - rf_used

    # （可选）百分比点缩放，对比 ND 不会改变比值本身；如无需要可保持默认 False
    scale = 100.0 if args.scale_pct else 1.0
    ex_lstm      = base_lstm      * scale
    ex_lstm_soft = base_lstm_soft * scale
    ex_ew        = base_ew        * scale
    ex_mi        = base_mi        * scale

    # === 输出：只用 ND 指标器 ===
    def report_with_nd_meter(name, base_series, rf_series):
        met = ex_post_metrics_generic(
            returns=base_series,
            rf=(rf_series if args.eval_threshold=='rf' else None),
            mode=args.eval_mode,
            threshold=args.eval_threshold,
            theta=args.alpha,
            periods_per_year=WEEKS_PER_YEAR
        )
        print(f"\n{name}:")
        print(f"  mean(base) = {np.mean(base_series):.6g}   std = {np.std(base_series, ddof=1):.6g}")
        for k, v in met.items():
            print(f"  {k} = {v:.6f}{' (annualized)' if args.eval_mode=='annualized' else ''}")

    print(f"\n=== Ex-post metrics on TEST (target={metric_name}, mode={args.eval_mode}, thr={args.eval_threshold}) ===")
    tag_mv   = f"{metric_name}::LSTM+MV+" + ("BLOSAM" if args.optim == "blosam" else "SGDM")
    tag_smax = f"{metric_name}::LSTM+SOFTMAX+" + ("BLOSAM" if args.optim == "blosam" else "SGDM")
    report_with_nd_meter(tag_mv,   ex_lstm,      rf_used)
    report_with_nd_meter(tag_smax, ex_lstm_soft, rf_used)
    report_with_nd_meter("EW",     ex_ew,        rf_used)
    report_with_nd_meter("MI",     ex_mi,        rf_used)

    # 保存结果（每个 metric 独立一个 CSV）
    out_name = f"results_{metric_name}_{'blosam' if args.optim=='blosam' else 'sgdm'}.csv"
    out = pd.DataFrame({
        "lstm_mv_port": lstm_port,
        "lstm_soft_port": lstm_soft_port,
        "ew_port": ew_port,
        "mi_port": mi_port,
        "rf": rf_used,
        "base_lstm_mv": base_lstm,
        "base_lstm_soft": base_lstm_soft,
        "base_ew": base_ew,
        "base_mi": base_mi,
        "excess_lstm_mv_final": ex_lstm,
        "excess_lstm_soft_final": ex_lstm_soft,
        "excess_ew_final": ex_ew,
        "excess_mi_final": ex_mi
    })
    out.to_csv(out_name, index=False)
    print(f"Saved: {out_name}")

# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_path", type=str, required=True)
    ap.add_argument("--assets_sheet", type=str, default="assets")
    ap.add_argument("--index_sheet", type=str, default="index")
    ap.add_argument("--tbill_path", type=str, required=True)

    # 逐个指标
    ap.add_argument("--metrics", nargs="*", default=["SR","SoR","OR","CSR","ESR"],
                    help="逐个优化的指标列表")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--attn_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    # 优化器
    ap.add_argument("--optim", type=str, default="blosam", choices=["blosam", "sgdm"],
                    help="选择优化器：blosam（两步）或 sgdm（单步 SGD+Momentum）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--rho", type=float, default=0.05)     # BLOSAM
    ap.add_argument("--adaptive", action="store_true")     # BLOSAM

    ap.add_argument("--alpha_mse", type=float, default=0.7, help="NLL vs proxy mixing: alpha*NLL + (1-alpha)*Proxy")
    ap.add_argument("--temp", type=float, default=6.0, help="softmax temperature for proxy")
    ap.add_argument("--clip_norm", type=float, default=1.0)

    ap.add_argument("--yearly_tbill", action="store_true", help="若 txt 中是年化 3M T-bill，则转换为周频")
    ap.add_argument("--alpha", type=float, default=0.95, help="CVaR/EVaR 置信度")

    # === 与 ND 对齐的评估开关 ===
    ap.add_argument("--eval_mode", choices=["weekly","annualized"], default="weekly")
    ap.add_argument("--eval_threshold", choices=["zero","rf"], default="zero")

    # 其余诊断选项（不再直接影响指标计算，避免口径冲突）
    ap.add_argument("--scale_pct", action="store_true", help="评估前收益×100（百分比点口径）")

    # 微调 & 协方差估计
    ap.add_argument("--finetune_epochs", type=int, default=2, help="测试期每周扩展式微调轮数（0 关闭）")
    ap.add_argument("--finetune_lr", type=float, default=1e-4, help="微调时临时使用的更小学习率")
    ap.add_argument("--cov_window", type=int, default=200, help="估计协方差的历史窗口长度")
    ap.add_argument("--cov_span", type=int, default=26, help="EWMA 衰减跨度（默认 26 周）")

    ap.add_argument("--drop_last_if_odd", action="store_true", help="奇数周丢最后一周，严格 half-half")
    ap.add_argument("--standardize", action="store_true", help="训练/微调使用训练期统计量做标准化")
    ap.add_argument("--seed", type=int, default=123)

    # 这些权重由 run_one_metric 内部覆盖为单一指标=1 的设置（命令行不再联动）
    ap.add_argument("--w_sr", type=float, default=1.0)
    ap.add_argument("--w_sor", type=float, default=0.0)
    ap.add_argument("--w_omega", type=float, default=0.0)
    ap.add_argument("--w_csr", type=float, default=0.0)
    ap.add_argument("--w_esr", type=float, default=0.0)

    # MV 稳健化
    ap.add_argument("--mv_ridge", type=float, default=1e-2)
    ap.add_argument("--mv_cap", type=float, default=0.2)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| Optim:", args.optim)

    # 读取数据
    df_assets = pd.read_excel(args.excel_path, sheet_name=args.assets_sheet, engine="openpyxl")
    df_index  = pd.read_excel(args.excel_path, sheet_name=args.index_sheet, engine="openpyxl")
    assets = df_assets.values.astype(np.float64)                 # (T, N)
    index_ret = df_index.values.squeeze().astype(np.float64)     # (T,)

    # T-bill
    with open(args.tbill_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        parts = [p for p in re.split(r"[,\s]+", txt) if p != ""]
        rf_vals = np.array([float(x) for x in parts], dtype=np.float64)

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

    # 打包共用数据，供每个 metric 复用
    data_pack = (assets, index_ret, rf_vals,
                 seq_train, seq_test, rf_test, mi_test,
                 seq_train_std, seq_test_std, assets_std, mu_train, std_train, N, T_half)

    # 逐个指标跑
    for m in args.metrics:
        run_one_metric(m, data_pack, args)

if __name__ == "__main__":
    main()
