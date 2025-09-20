#!/usr/bin/env python3
# nd_repro_wang_gan_torch.py
# GPU-enabled (PyTorch) reproduction of Wang & Gan (2023) neurodynamic portfolio optimization.

import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Callable, Tuple

# ------------------------
# Global knobs
# ------------------------
EPS = 1e-12
WEEKS_PER_YEAR = 52

def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Torch helpers (GPU aware)
# ------------------------
def t_sanitize(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    x = x.clone()
    mask = ~torch.isfinite(x)
    if mask.any():
        x[mask] = fill
    return x

def t_safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # elementwise safe division
    # handle both scalar and vector cases
    b_safe = torch.where(b.abs() > eps, b, torch.sign(b + 0.0) * eps)
    return a / b_safe

def t_clip_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    nrm = torch.linalg.vector_norm(v)
    if torch.isfinite(nrm) and (nrm > max_norm) and (nrm > 0):
        v = v * (max_norm / nrm)
    return v

def to_torch_np(a, device, dtype=torch.float64):
    return torch.as_tensor(a, dtype=dtype, device=device)

def to_numpy_cpu(x: torch.Tensor):
    return x.detach().cpu().numpy()

# ------------------------
# Post-hoc metrics (run on CPU or GPU; final to CPU floats)
# ------------------------
def ex_post_metrics_generic(returns, rf=None, mode="weekly",
                            threshold="zero", periods_per_year=52, theta=0.95):
    """
    returns: 1D np.array or torch.Tensor
    rf:      same length when threshold='rf'
    """
    # Use torch on CPU for simplicity/stability; convert inputs
    if isinstance(returns, torch.Tensor):
        r = to_numpy_cpu(returns)
    else:
        r = np.asarray(returns, dtype=np.float64)
    if rf is not None and isinstance(rf, torch.Tensor):
        rf_np = to_numpy_cpu(rf)
    else:
        rf_np = None if rf is None else np.asarray(rf, dtype=np.float64)

    r = np.asarray(r, dtype=np.float64)
    if threshold == "rf":
        if rf_np is None:
            raise ValueError("eval_threshold=rf 需要传入 rf 序列")
        exc = r - rf_np
    else:
        exc = r

    mu_w = float(np.nanmean(exc))
    std_w = float(np.nanstd(exc, ddof=1))
    sr_w = mu_w / (std_w + EPS) if std_w > 0 else 0.0

    dn = np.maximum(-exc, 0.0)
    sdn_w = math.sqrt(float(np.mean(dn**2)) + EPS)
    sor_w = mu_w / (sdn_w + EPS)

    lpm1_w = float(np.mean(dn)) + EPS
    omega_w = mu_w / lpm1_w + 1.0

    m = len(exc)
    losses = -exc
    q = int(math.ceil((1.0 - theta) * m))
    if q > 0:
        worst = np.partition(losses, -q)[-q:]
        cvar_w = float(np.mean(worst)) + EPS
    else:
        cvar_w = max(float(np.max(losses)), 0.0) + EPS
    csr_w = mu_w / (cvar_w + EPS)

    # EVaR via log-sum-exp (CPU)
    def evar_weekly(x, theta=0.95):
        x = np.asarray(x, dtype=np.float64)
        best = None
        for rho in np.logspace(-3, 0, 40):  # 1e-3..1
            logits = -x / max(rho, 1e-6)
            mlog = np.max(logits)
            lse = mlog + math.log(np.exp(logits - mlog).sum() + EPS)
            logS = lse - math.log(m * (1.0 - theta) + EPS)
            val = rho * logS
            best = val if (best is None or val < best) else best
        return float(best if best is not None else 0.0)

    evar_w = evar_weekly(exc, theta=theta) + EPS
    esr_w = mu_w / (evar_w + EPS)

    if mode == "weekly":
        return dict(SRp=sr_w, SoRp=sor_w, ORp=omega_w, CSRp=csr_w, ESRp=esr_w)

    mu_a  = mu_w * periods_per_year
    std_a = std_w * math.sqrt(periods_per_year)
    sdn_a = sdn_w * math.sqrt(periods_per_year)
    lpm1_a = lpm1_w * math.sqrt(periods_per_year)
    cvar_a = cvar_w * math.sqrt(periods_per_year)
    evar_a = evar_w * math.sqrt(periods_per_year)

    sr_a  = mu_a / (std_a + EPS) if std_a > 0 else 0.0
    sor_a = mu_a / (sdn_a + EPS)
    omega_a = mu_a / (lpm1_a + EPS) + 1.0
    csr_a = mu_a / (cvar_a + EPS)
    esr_a = mu_a / (evar_a + EPS)
    return dict(SRp=sr_a, SoRp=sor_a, ORp=omega_a, CSRp=csr_a, ESRp=esr_a)

# ------------------------
# Estimators
# ------------------------
def moving_estimates_torch(R_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # R_hist: (m, n)
    R = t_sanitize(R_hist)
    mu = R.mean(dim=0)  # (n,)
    m = R.shape[0]
    if m > 1:
        # sample covariance with ddof=1
        R_dm = R - mu
        V = (R_dm.T @ R_dm) / (m - 1)
    else:
        n = R.shape[1]
        V = torch.zeros((n, n), dtype=R.dtype, device=R.device)
    V = t_sanitize(V) + 1e-5 * torch.eye(V.shape[0], dtype=V.dtype, device=V.device)
    return mu, V

def semideviation_sq_torch(z, R, rb):
    rz = R @ z  # (m,)
    s = torch.clamp(rb - rz, min=0.0)
    return (s*s).mean()

def lpm1_fun_torch(z, R, rb):
    rz = R @ z
    return torch.clamp(rb - rz, min=0.0).mean()

def evar_terms_stable_torch(z, R, rho, theta):
    """
    Return (EVaR(z), alpha, logS); torch (GPU) with logsumexp.
    alpha sums to 1.
    """
    m = R.shape[0]
    rz = R @ z  # (m,)
    rho = torch.clamp(torch.as_tensor(rho, dtype=R.dtype, device=R.device), min=1e-6)
    logits = -rz / rho
    lse = torch.logsumexp(logits, dim=0)  # log sum exp
    logS = lse - math.log(m * (1.0 - theta) + EPS)
    evar = rho * logS
    # softmax for alpha
    alpha = torch.softmax(logits, dim=0)  # (m,)
    return evar, alpha, logS

# ------------------------
# PNN Solver (Torch)
# ------------------------
class PNNSolverTorch:
    def __init__(self, n, device,
                 eps=1e-3, step=0.05, max_steps=2000, tol=1e-7,
                 max_grad_norm=10.0, max_rhs_norm=10.0, max_u_abs=50.0,
                 verbose=False, dtype=torch.float64):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.eps = float(eps)
        self.h = float(step)
        self.max_steps = int(max_steps)
        self.tol = float(tol)
        self.max_grad_norm = float(max_grad_norm)
        self.max_rhs_norm = float(max_rhs_norm)
        self.max_u_abs = float(max_u_abs)
        self.verbose = verbose

        e = torch.ones((n, 1), dtype=dtype, device=device)
        self.P = (e @ e.T) / float(n)
        self.ImP = torch.eye(n, dtype=dtype, device=device) - self.P
        self.e_over_n = (e.flatten() / float(n))

    def _project_u(self, u: torch.Tensor) -> torch.Tensor:
        u = t_sanitize(u)
        u = torch.clamp(u, -self.max_u_abs, self.max_u_abs)
        return u

    def _project_x_budget(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0)
        s = x.sum()
        if s <= EPS:
            return torch.ones_like(x) / x.numel()
        return x / s

    def solve(self, grad_f: Callable[[torch.Tensor], torch.Tensor], x0=None) -> torch.Tensor:
        n = self.n
        if x0 is None:
            u = torch.ones(n, dtype=self.dtype, device=self.device) / n
        else:
            u = x0.clone().to(self.device, dtype=self.dtype)
        u = t_sanitize(u)

        for _ in range(self.max_steps):
            x = torch.clamp(u, min=0.0)
            s = x.sum()
            if s <= EPS:
                x = torch.ones(n, dtype=self.dtype, device=self.device) / n
            else:
                x = x / s

            z = (self.ImP @ x) + self.e_over_n
            g = grad_f(z)
            g = t_sanitize(g)
            g = t_clip_norm(g, self.max_grad_norm)

            RHS = - (self.P @ x) + self.e_over_n - (self.ImP @ (u - x + g))
            RHS = t_sanitize(RHS)
            RHS = t_clip_norm(RHS, self.max_rhs_norm)

            du = (self.h / max(self.eps, 1e-12)) * RHS
            u_new = self._project_u(u + du)

            if torch.linalg.vector_norm(u_new - u) < self.tol:
                u = u_new
                break
            u = u_new

        x = torch.clamp(u, min=0.0)
        x = x / max(x.sum(), EPS)
        return x

# ------------------------
# One-scale solvers (Torch)
# ------------------------
def solve_SR(mu, V, R_hist, rf, pnn: PNNSolverTorch, max_outer=15, tol_outer=1e-6, x_init=None):
    n = mu.numel()
    x = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    for _ in range(max_outer):
        num = (mu @ x) - rf
        den = (x @ V @ x) + EPS
        gamma = torch.clamp(num / den, min=0.0)

        def grad_f(z):
            return t_clip_norm((gamma**2) * (V @ z) - gamma * mu, 50.0)

        x_new = pnn.solve(grad_f, x0=x)
        if torch.linalg.vector_norm(x_new - x) < tol_outer:
            x = x_new; break
        x = x_new
    return x

def solve_SoR(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch, max_outer=15, tol_outer=1e-6, x_init=None):
    R = t_sanitize(R_hist); m = R.shape[0]; rb = rf
    n = mu.numel()
    x = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    for _ in range(max_outer):
        sd2 = semideviation_sq_torch(x, R, rb) + EPS
        gamma = torch.clamp((mu @ x - rf) / sd2, min=0.0)

        def grad_f(z):
            rz = R @ z
            s = torch.clamp(rb - rz, min=0.0)
            grad_sd2 = -(2.0/m) * (R.T @ s)
            return t_clip_norm(0.5*(gamma**2)*grad_sd2 - gamma*mu, 50.0)

        x_new = pnn.solve(grad_f, x0=x)
        if torch.linalg.vector_norm(x_new - x) < tol_outer:
            x = x_new; break
        x = x_new
    return x

def solve_OR(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch, max_outer=18, tol_outer=1e-6, x_init=None):
    R = t_sanitize(R_hist); m = R.shape[0]; rb = rf
    n = mu.numel()
    x = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    for _ in range(max_outer):
        lpm = lpm1_fun_torch(x, R, rb) + EPS
        gamma = torch.clamp((mu @ x - rf) / (lpm**2), min=0.0)

        def grad_f(z):
            rz = R @ z
            mask = (rb - rz) > 0.0
            if mask.any():
                grad_lpm1 = -(R[mask].T @ torch.ones(mask.sum(), dtype=mu.dtype, device=mu.device)) / m
            else:
                grad_lpm1 = torch.zeros(n, dtype=mu.dtype, device=mu.device)
            lpm_z = lpm1_fun_torch(z, R, rb) + EPS
            return t_clip_norm((gamma**2) * lpm_z * grad_lpm1 - gamma * mu, 50.0)

        x_new = pnn.solve(grad_f, x0=x)
        if torch.linalg.vector_norm(x_new - x) < tol_outer:
            x = x_new; break
        x = x_new
    return x

def solve_CSR(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch, theta=0.95, max_outer=18, tol_outer=1e-6, x_init=None, rho_init=1e-2):
    R = t_sanitize(R_hist); m = R.shape[0]
    n = mu.numel()
    x = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    rho = torch.tensor(max(rho_init, 1e-5), dtype=mu.dtype, device=mu.device)
    for _ in range(max_outer):
        rz = R @ x
        s = torch.clamp(-rz - rho, min=0.0)
        denom = rho + s.sum() / (m*(1.0 - theta)) + EPS
        lam = torch.clamp((mu @ x - rf) / denom, min=0.0)

        def grad_f_z(z):
            rz_ = R @ z
            mask = (-rz_ - rho) > 0.0
            if mask.any():
                grad = lam * (-(R[mask].T @ torch.ones(mask.sum(), dtype=mu.dtype, device=mu.device)) / (m*(1.0-theta))) - mu
            else:
                grad = -mu
            return t_clip_norm(grad, 50.0)

        x_new = pnn.solve(grad_f_z, x0=x)

        rz_new = R @ x_new
        mask = (-rz_new - rho) > 0.0
        subgrad_rho = lam * (1.0 - (mask.sum() / (m*(1.0-theta) + EPS)))
        rho_new = torch.clamp(rho - 0.05 * subgrad_rho, min=1e-5)

        if (torch.linalg.vector_norm(x_new - x) < tol_outer) and (torch.abs(rho_new - rho) < 1e-6):
            x, rho = x_new, rho_new; break
        x, rho = x_new, rho_new
    return x

def solve_ESR(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch, theta=0.95, max_outer=20, tol_outer=1e-6, x_init=None, rho_init=5e-2):
    R = t_sanitize(R_hist)
    n = mu.numel()
    x = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    rho = torch.tensor(max(rho_init, 1e-5), dtype=mu.dtype, device=mu.device)
    for _ in range(max_outer):
        evar, alpha, _ = evar_terms_stable_torch(x, R, rho, theta)
        denom = torch.clamp(evar, min=EPS)
        lam = torch.clamp((mu @ x - rf) / denom, min=0.0)

        def grad_f_z(z):
            evar_z, alpha_z, _ = evar_terms_stable_torch(z, R, rho, theta)
            grad = - lam * (R.T @ alpha_z) - mu
            return t_clip_norm(grad, 50.0)

        x_new = pnn.solve(grad_f_z, x0=x)

        rz_new = R @ x_new
        logits = -rz_new / torch.clamp(rho, min=1e-6)
        lse = torch.logsumexp(logits, dim=0)
        S = torch.exp(lse - math.log(R.shape[0]*(1.0-theta) + EPS))
        alpha_new = torch.softmax(logits, dim=0)
        grad_rho = lam * (torch.log(torch.clamp(S, min=EPS)) + (rz_new @ alpha_new) / torch.clamp(rho, min=1e-6))
        rho_new = torch.clamp(rho - 0.02 * grad_rho, min=1e-5)

        if (torch.linalg.vector_norm(x_new - x) < tol_outer) and (torch.abs(rho_new - rho) < 1e-6):
            x, rho = x_new, rho_new; break
        x, rho = x_new, rho_new
    return x

# ------------------------
# Two-scale helpers/solvers (Torch)
# ------------------------
def _pnn_step(pnn: PNNSolverTorch, u, grad_f_at_z):
    x = torch.clamp(u, min=0.0)
    z = (pnn.ImP @ x) + pnn.e_over_n
    g = grad_f_at_z(z)
    g = t_sanitize(g)
    g = t_clip_norm(g, pnn.max_grad_norm)
    RHS = - (pnn.P @ x) + pnn.e_over_n - (pnn.ImP @ (u - x + g))
    RHS = t_sanitize(RHS)
    RHS = t_clip_norm(RHS, pnn.max_rhs_norm)
    du = (pnn.h / max(pnn.eps, 1e-12)) * RHS
    u_new = pnn._project_u(u + du)
    return u_new

def solve_SR_two_scale(mu, V, R_hist, rf, pnn: PNNSolverTorch,
                       iters=4000, eta=5e-4, tol=1e-7, x_init=None, gamma_init=0.0):
    n = mu.numel()
    u = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    gamma = torch.tensor(max(gamma_init, 0.0), dtype=mu.dtype, device=mu.device)
    prev_u, prev_gamma = u.clone(), gamma.clone()
    for _ in range(iters):
        def grad_f(z):
            return t_clip_norm((gamma**2) * (V @ z) - gamma * mu, 50.0)
        u = _pnn_step(pnn, u, grad_f)
        x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
        den = (x @ V @ x) + EPS
        target = torch.clamp((mu @ x - rf) / den, min=0.0)
        gamma = (1.0 - eta) * gamma + eta * target
        gamma = torch.clamp(gamma, 0.0, 1e6)
        if (torch.linalg.vector_norm(u - prev_u) < tol) and (torch.abs(gamma - prev_gamma) < tol):
            break
        prev_u, prev_gamma = u.clone(), gamma.clone()
    x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
    return x

def solve_SoR_two_scale(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch,
                        iters=5000, eta=5e-4, tol=1e-7, x_init=None, gamma_init=0.0):
    R = t_sanitize(R_hist); m = R.shape[0]; rb = rf
    n = mu.numel()
    u = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    gamma = torch.tensor(max(gamma_init, 0.0), dtype=mu.dtype, device=mu.device)
    prev_u, prev_gamma = u.clone(), gamma.clone()
    for _ in range(iters):
        def grad_f(z):
            rz = R @ z
            s = torch.clamp(rb - rz, min=0.0)
            grad_sd2 = -(2.0/m) * (R.T @ s)
            return t_clip_norm(0.5*(gamma**2)*grad_sd2 - gamma*mu, 50.0)
        u = _pnn_step(pnn, u, grad_f)
        x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
        sd2 = semideviation_sq_torch(x, R, rb) + EPS
        target = torch.clamp((mu @ x - rf) / sd2, min=0.0)
        gamma = (1.0 - eta) * gamma + eta * target
        gamma = torch.clamp(gamma, 0.0, 1e6)
        if (torch.linalg.vector_norm(u - prev_u) < tol) and (torch.abs(gamma - prev_gamma) < tol):
            break
        prev_u, prev_gamma = u.clone(), gamma.clone()
    x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
    return x

def solve_OR_two_scale(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch,
                       iters=6000, eta=5e-4, tol=1e-7, x_init=None, gamma_init=0.0):
    R = t_sanitize(R_hist); m = R.shape[0]; rb = rf
    n = mu.numel()
    u = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    gamma = torch.tensor(max(gamma_init, 0.0), dtype=mu.dtype, device=mu.device)
    prev_u, prev_gamma = u.clone(), gamma.clone()
    for _ in range(iters):
        def grad_f(z):
            rz = R @ z
            mask = (rb - rz) > 0.0
            if mask.any():
                grad_lpm1 = -(R[mask].T @ torch.ones(mask.sum(), dtype=mu.dtype, device=mu.device)) / m
            else:
                grad_lpm1 = torch.zeros(n, dtype=mu.dtype, device=mu.device)
            lpm_z = lpm1_fun_torch(z, R, rb) + EPS
            return t_clip_norm((gamma**2) * lpm_z * grad_lpm1 - gamma * mu, 50.0)
        u = _pnn_step(pnn, u, grad_f)
        x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
        lpm = lpm1_fun_torch(x, R, rb) + EPS
        target = torch.clamp((mu @ x - rf) / (lpm**2), min=0.0)
        gamma = (1.0 - eta) * gamma + eta * target
        gamma = torch.clamp(gamma, 0.0, 1e6)
        if (torch.linalg.vector_norm(u - prev_u) < tol) and (torch.abs(gamma - prev_gamma) < tol):
            break
        prev_u, prev_gamma = u.clone(), gamma.clone()
    x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
    return x

def solve_CSR_two_scale(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch,
                        theta=0.95, iters=6000, eta=5e-4, eta_rho=5e-4, tol=1e-7,
                        x_init=None, lam_init=0.0, rho_init=1e-2):
    R = t_sanitize(R_hist); m = R.shape[0]
    n = mu.numel()
    u = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    lam = torch.tensor(max(lam_init, 0.0), dtype=mu.dtype, device=mu.device)
    rho = torch.tensor(max(rho_init, 1e-5), dtype=mu.dtype, device=mu.device)
    prev_u, prev_lam, prev_rho = u.clone(), lam.clone(), rho.clone()
    for _ in range(iters):
        def grad_f_z(z):
            rz = R @ z
            mask = (-rz - rho) > 0.0
            if mask.any():
                grad = lam * (-(R[mask].T @ torch.ones(mask.sum(), dtype=mu.dtype, device=mu.device)) / (m*(1.0-theta))) - mu
            else:
                grad = -mu
            return t_clip_norm(grad, 50.0)
        u = _pnn_step(pnn, u, grad_f_z)
        x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)

        rz = R @ x
        s = torch.clamp(-rz - rho, min=0.0)
        denom = rho + s.sum() / (m*(1.0-theta)) + EPS
        lam_target = torch.clamp((mu @ x - rf) / denom, min=0.0)
        lam = (1.0 - eta) * lam + eta * lam_target
        lam = torch.clamp(lam, 0.0, 1e6)

        mask = (-rz - rho) > 0.0
        subgrad_rho = lam * (1.0 - (mask.sum() / (m*(1.0-theta) + EPS)))
        rho = torch.clamp(rho - eta_rho * subgrad_rho, min=1e-5, max=1e3)

        if (torch.linalg.vector_norm(u - prev_u) < tol and
            torch.abs(lam - prev_lam) < tol and
            torch.abs(rho - prev_rho) < tol):
            break
        prev_u, prev_lam, prev_rho = u.clone(), lam.clone(), rho.clone()
    x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
    return x

def solve_ESR_two_scale(mu, V_unused, R_hist, rf, pnn: PNNSolverTorch,
                        theta=0.95, iters=7000, eta=5e-4, eta_rho=3e-4, tol=1e-7,
                        x_init=None, lam_init=0.0, rho_init=5e-2):
    R = t_sanitize(R_hist); m = R.shape[0]
    n = mu.numel()
    u = torch.ones(n, dtype=mu.dtype, device=mu.device) / n if x_init is None else x_init.clone()
    lam = torch.tensor(max(lam_init, 0.0), dtype=mu.dtype, device=mu.device)
    rho = torch.tensor(max(rho_init, 1e-5), dtype=mu.dtype, device=mu.device)
    prev_u, prev_lam, prev_rho = u.clone(), lam.clone(), rho.clone()
    for _ in range(iters):
        def grad_f_z(z):
            evar_z, alpha_z, _ = evar_terms_stable_torch(z, R, rho, theta)
            grad = - lam * (R.T @ alpha_z) - mu
            return t_clip_norm(grad, 50.0)
        u = _pnn_step(pnn, u, grad_f_z)
        x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)

        evar_x, alpha_x, logS_x = evar_terms_stable_torch(x, R, rho, theta)
        denom = torch.clamp(evar_x, min=EPS)
        lam_target = torch.clamp((mu @ x - rf) / denom, min=0.0)
        lam = (1.0 - eta) * lam + eta * lam_target
        lam = torch.clamp(lam, 0.0, 1e6)

        rz = R @ x
        logits = -rz / torch.clamp(rho, min=1e-6)
        lse = torch.logsumexp(logits, dim=0)
        S = torch.exp(lse - math.log(m*(1.0-theta) + EPS))
        alpha = torch.softmax(logits, dim=0)
        grad_rho = lam * (torch.log(torch.clamp(S, min=EPS)) + (rz @ alpha) / torch.clamp(rho, min=1e-6))
        rho = torch.clamp(rho - eta_rho * grad_rho, min=1e-5, max=1e3)

        if (torch.linalg.vector_norm(u - prev_u) < tol and
            torch.abs(lam - prev_lam) < tol and
            torch.abs(rho - prev_rho) < tol):
            break
        prev_u, prev_lam, prev_rho = u.clone(), lam.clone(), rho.clone()
    x = torch.clamp(u, min=0.0); x = x / max(x.sum(), EPS)
    return x

# ------------------------
# IO utilities
# ------------------------
def read_assets_index_from_excel(path, assets_sheet="Assets_Returns", index_sheet="Index_Returns"):
    df_assets = pd.read_excel(path, sheet_name=assets_sheet, engine="openpyxl")
    df_index  = pd.read_excel(path, sheet_name=index_sheet, engine="openpyxl")
    assets = df_assets.values.astype(np.float64)
    index_ret = df_index.values.squeeze().astype(np.float64)
    return assets, index_ret

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_path", required=True)
    ap.add_argument("--assets_sheet", default="Assets_Returns")
    ap.add_argument("--index_sheet", default="Index_Returns")
    ap.add_argument("--tbill_path", required=True)
    ap.add_argument("--yearly_tbill", action="store_true")
    ap.add_argument("--drop_last_if_odd", action="store_true")
    ap.add_argument("--rebalance", type=int, default=1)
    ap.add_argument("--metrics", nargs="*", default=["SR","SoR","OR","CSR","ESR"])
    ap.add_argument("--theta", type=float, default=0.95)
    # PNN params
    ap.add_argument("--pnn_eps", type=float, default=3e-3)
    ap.add_argument("--pnn_step", type=float, default=0.02)
    ap.add_argument("--pnn_max", type=int, default=600)
    ap.add_argument("--pnn_tol", type=float, default=1e-7)
    ap.add_argument("--pnn_max_grad", type=float, default=8.0)
    ap.add_argument("--pnn_max_rhs", type=float, default=8.0)
    ap.add_argument("--pnn_max_u_abs", type=float, default=30.0)
    # TWO-SCALE
    ap.add_argument("--two_scale", action="store_true")
    ap.add_argument("--ts_iters", type=int, default=2500)
    ap.add_argument("--ts_eta", type=float, default=3e-4)
    ap.add_argument("--ts_eta_rho", type=float, default=2e-4)
    # Evaluation
    ap.add_argument("--eval_mode", choices=["weekly","annualized"], default="weekly")
    ap.add_argument("--eval_sample", choices=["oos","full"], default="oos")
    ap.add_argument("--eval_threshold", choices=["zero","rf"], default="zero")
    # Device
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; default auto")
    # Saving
    ap.add_argument("--save_weights_dir", default=None)
    ap.add_argument("--save_excess_csv", default=None)
    ap.add_argument("--save_metrics_csv", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    dtype = torch.float64
    print(f"[INFO] Using device: {device}")

    # Read data (CPU -> Torch on device)
    assets_np, index_np = read_assets_index_from_excel(args.excel_path, args.assets_sheet, args.index_sheet)
    with open(args.tbill_path, "r", encoding="utf-8") as f:
        rf_vals_np = np.array([float(x) for x in re.split(r"[,\s]+", f.read().strip()) if x != ""], dtype=np.float64)

    if args.yearly_tbill:
        rf_vals_np = (1.0 + rf_vals_np) ** (1.0 / WEEKS_PER_YEAR) - 1.0

    T_assets, N = assets_np.shape
    T_index = index_np.shape[0]
    T_rf    = rf_vals_np.shape[0]
    T_all   = min(T_assets, T_index, T_rf)
    if (T_assets != T_all) or (T_index != T_all) or (T_rf != T_all):
        print(f"[WARN] Length mismatch: assets={T_assets}, index={T_index}, tbill={T_rf}. Truncate to {T_all}.")
        assets_np = assets_np[:T_all, :]
        index_np  = index_np[:T_all]
        rf_vals_np = rf_vals_np[:T_all]

    if args.drop_last_if_odd and (T_all % 2 == 1):
        print("[INFO] Odd number of weeks detected; drop last week to make even.")
        assets_np = assets_np[:-1, :]
        index_np  = index_np[:-1]
        rf_vals_np = rf_vals_np[:-1]

    T = assets_np.shape[0]
    T_half = T // 2

    # torch tensors on device
    assets = to_torch_np(assets_np, device, dtype)            # (T, N)
    index_ret = to_torch_np(index_np, device, dtype)          # (T,)
    rf_vals = to_torch_np(rf_vals_np, device, dtype)          # (T,)

    seq_test  = assets[T_half:, :]
    rf_test   = rf_vals[T_half:]
    mi_test   = index_ret[T_half:]

    # Baselines (EW/MI) eval (OOS or full) -> do on CPU via numpy
    if args.eval_sample == "full":
        ew_base = to_numpy_cpu(assets.mean(dim=1))
        mi_base = to_numpy_cpu(index_ret)
        rf_eval = to_numpy_cpu(rf_vals)
    else:
        ew_base = to_numpy_cpu(seq_test.mean(dim=1))
        mi_base = to_numpy_cpu(mi_test)
        rf_eval = to_numpy_cpu(rf_test)

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

    print(f"T(all)={T}, N(assets)={N}, OOS weeks={seq_test.shape[0]}")
    print(f"\n== EW ({args.eval_sample}, {args.eval_mode}, thr={args.eval_threshold}) ==")
    for k, v in ew_metrics.items(): print(f"  {k}: {v:.6f}")
    print(f"\n== MI ({args.eval_sample}, {args.eval_mode}, thr={args.eval_threshold}) ==")
    for k, v in mi_metrics.items(): print(f"  {k}: {v:.6f}")

    # PNN solver (Torch)
    pnn = PNNSolverTorch(
        N, device,
        eps=args.pnn_eps, step=args.pnn_step, max_steps=args.pnn_max, tol=args.pnn_tol,
        verbose=False,
        max_grad_norm=args.pnn_max_grad, max_rhs_norm=args.pnn_max_rhs, max_u_abs=args.pnn_max_u_abs,
        dtype=dtype
    )

    metrics = tuple(args.metrics)
    x_state = {m: torch.ones(N, dtype=dtype, device=device)/N for m in metrics}
    weights_hist = {m: [] for m in metrics}
    excess_hist  = {m: [] for m in metrics}

    # Rolling OOS on GPU
    for i in range(T - T_half):
        t_global = T_half + i
        R_est = assets[:t_global, :]
        mu, V = moving_estimates_torch(R_est)
        rf_t = rf_vals[t_global]

        if (i % args.rebalance) == 0:
            for m in metrics:
                x0 = x_state[m]
                if not args.two_scale:
                    if m == "SR":
                        x_state[m] = solve_SR(mu, V, R_est, rf_t, pnn, x_init=x0)
                    elif m == "SoR":
                        x_state[m] = solve_SoR(mu, V, R_est, rf_t, pnn, x_init=x0)
                    elif m == "OR":
                        x_state[m] = solve_OR(mu, V, R_est, rf_t, pnn, x_init=x0)
                    elif m == "CSR":
                        x_state[m] = solve_CSR(mu, V, R_est, rf_t, pnn, theta=args.theta, x_init=x0)
                    elif m == "ESR":
                        x_state[m] = solve_ESR(mu, V, R_est, rf_t, pnn, theta=args.theta, x_init=x0)
                    else:
                        raise ValueError(f"Unknown metric {m}")
                else:
                    if m == "SR":
                        x_state[m] = solve_SR_two_scale(mu, V, R_est, rf_t, pnn,
                                                        iters=args.ts_iters, eta=args.ts_eta, x_init=x0)
                    elif m == "SoR":
                        x_state[m] = solve_SoR_two_scale(mu, V, R_est, rf_t, pnn,
                                                         iters=args.ts_iters, eta=args.ts_eta, x_init=x0)
                    elif m == "OR":
                        x_state[m] = solve_OR_two_scale(mu, V, R_est, rf_t, pnn,
                                                        iters=args.ts_iters, eta=args.ts_eta, x_init=x0)
                    elif m == "CSR":
                        x_state[m] = solve_CSR_two_scale(mu, V, R_est, rf_t, pnn, theta=args.theta,
                                                         iters=args.ts_iters, eta=args.ts_eta, eta_rho=args.ts_eta_rho,
                                                         x_init=x0)
                    elif m == "ESR":
                        x_state[m] = solve_ESR_two_scale(mu, V, R_est, rf_t, pnn, theta=args.theta,
                                                         iters=args.ts_iters, eta=args.ts_eta, eta_rho=args.ts_eta_rho,
                                                         x_init=x0)
                    else:
                        raise ValueError(f"Unknown metric {m}")

        # record weights & compute OOS excess for week t_global
        r_next = assets[t_global, :]  # (N,)
        for m in metrics:
            w = x_state[m]
            weights_hist[m].append(w.unsqueeze(0))  # store on GPU; stack later
            excess = (r_next @ w) - rf_t
            excess_hist[m].append(excess)

    # Stack histories and move to CPU numpy for saving/eval
    for m in metrics:
        weights_hist[m] = torch.vstack(weights_hist[m])  # (T_half, N)
        excess_hist[m]  = torch.stack(excess_hist[m])    # (T_half,)

    # Summaries (OOS only)
    nd_summary = {}
    for m in metrics:
        if args.eval_threshold == "rf":
            r_total = excess_hist[m] + rf_test
            nd_summary[m] = ex_post_metrics_generic(r_total, rf=rf_test,
                                mode=args.eval_mode, threshold="rf",
                                theta=args.theta, periods_per_year=WEEKS_PER_YEAR)
        else:
            nd_summary[m] = ex_post_metrics_generic(excess_hist[m], rf=None,
                                mode=args.eval_mode, threshold="zero",
                                theta=args.theta, periods_per_year=WEEKS_PER_YEAR)

    print(f"\n== Neurodynamics (NO, OOS, {args.eval_mode}, thr={args.eval_threshold}) ==")
    for m in metrics:
        print(f"--- {m} ---")
        for k, v in nd_summary[m].items():
            print(f"  {k}: {v:.6f}")

    # Save
    if args.save_weights_dir:
        os.makedirs(args.save_weights_dir, exist_ok=True)
        asset_names = np.array([f"A{i}" for i in range(N)])
        for m in metrics:
            W = to_numpy_cpu(weights_hist[m])
            np.savez(os.path.join(args.save_weights_dir, f"weights_{m}.npz"),
                     W=W, assets=asset_names)
        print(f"\nSaved weights to: {args.save_weights_dir}")

    if args.save_excess_csv:
        out = {
            "EW_excess": ew_base - rf_eval if args.eval_sample=="oos" else to_numpy_cpu(assets.mean(dim=1)) - to_numpy_cpu(rf_vals),
            "MI_excess": mi_base - rf_eval if args.eval_sample=="oos" else to_numpy_cpu(index_ret) - to_numpy_cpu(rf_vals),
        }
        for m in metrics:
            out[f"NO_{m}_excess"] = to_numpy_cpu(excess_hist[m])
        pd.DataFrame(out).to_csv(args.save_excess_csv, index=False)
        print(f"Saved excess to: {args.save_excess_csv}")

    if args.save_metrics_csv:
        rows = []
        rows.append({"Strategy": "EW", "Sample": args.eval_sample, "Mode": args.eval_mode,
                     "Threshold": args.eval_threshold, **ew_metrics})
        rows.append({"Strategy": "MI", "Sample": args.eval_sample, "Mode": args.eval_mode,
                     "Threshold": args.eval_threshold, **mi_metrics})
        for m in metrics:
            rows.append({"Strategy": f"NO_{m}", "Sample": "oos", "Mode": args.eval_mode,
                         "Threshold": args.eval_threshold, **nd_summary[m]})
        pd.DataFrame(rows).to_csv(args.save_metrics_csv, index=False)
        print(f"Saved metrics to: {args.save_metrics_csv}")

if __name__ == "__main__":
    main()
