# LSTM Portfolio Selection (Optimizer Choice) 

This script trains an **LSTM** to predict the **next holding-window mean returns per asset**, then performs a **rolling backtest** that:
1) predicts μ with the LSTM,  
2) estimates Σ (EWMA or sample),  
3) solves a **Markowitz** portfolio (long-only, simplex; optional cap & turnover penalty),  
4) holds the weights for **H** weeks, and  
5) reports **gross, annualized ex-post** metrics: **SRp, SoRp, ORp, CSRp, ESRp**.

---

## Data format

- **Excel** (headerless)
  - `asset_return`: rows = weeks, columns = assets; weekly simple returns (e.g., 0.01 = 1%).
  - `index_return`: single column; the index’s weekly return (used only for logging/sanity, not in the LSTM loss).
- **TXT** (headerless)
  - Weekly **risk-free** returns; either comma- or whitespace-separated; one row or one column.

All series are aligned to the shortest length. Weeks with **all-NaN** assets are dropped.  
Assets are **winsorized cross-sectionally** (per week, 1%/99% by default) and NaNs are filled (`zero` or `ffill`).

---

## Rolling protocol

- **In-sample window**: `lookback_L` weeks (default **52**).
- **Hold window**: `hold_H` weeks (default **12**).
- **Rebalance** every `hold_H` weeks: at each rebalance `t`, estimate on `[t-L, t-1]`, trade and hold for `[t, t+H-1]`.

The LSTM is trained on windows built from the same rebalance points:
- **Input** \(X): last `L` weeks (shape L×N)
- **Target** \(y): next `H`-week **mean** return per asset (shape N)

A fixed **train/val** split in *rebalance windows* is created via `train_ratio` (default 0.7). The split is persisted to
`split_rebals_L{L}_H{H}_{tag}.npz` unless `--no_persist_split`.

---

## Optimizers

Choose with `--opt {sgdm, blosam}`

- **sgdm** — PyTorch SGD with momentum (no external deps)
- **blosam** — The proposed optimizer (requires your local `optimizers.blosam` package).  
  If you don’t have it, simply use `--opt sgdm`.

A cosine or step scheduler is **not** used; the script uses **ReduceLROnPlateau** by default (disable with `--no_scheduler`).

---

## Markowitz allocation (at each rebalance)

- Objective:  
  `mu^T w - lam * w^T Σ w - gamma_turnover * ||w - last_w||_1`  
- Constraints: `w ≥ 0`, `sum(w) = 1`, optional per-asset cap `w_i ≤ weight_cap`.

Σ can be **EWMA** (`--cov_method ewma`, `--cov_halflife`) or **sample**; optional diagonal shrink (`--shrink`).

---

## Metrics (printed, **gross & annualized**)

- **SRp** — Sharpe: `mean(r_p - rf) / std(r_p - rf)` → × √`annualize_factor` (default 52)
- **SoRp** — Sortino (downside vs **rf**): `mean(r_p - rf) / sqrt(mean(max(rf - r_p, 0)^2))` → × √AF
- **CSRp** — Conditional Sharpe: `mean(excess) / CVaR_θ(loss)`, loss = `-r_p`, θ = `--theta` (default 0.95) → × √AF
- **ESRp** — Entropic Sharpe: `mean(excess) / EVaR_θ(loss)` → × √AF
- **ORp** — Omega: `mean(excess) / mean(max(rf - r_p, 0)) + 1`  
  - `--omega_ann block` (recommended for annualized reporting): compound weekly `r_p` and `rf` into **annual blocks** (length = `annualize_factor`) and compute Omega on those blocks.  
  - `--omega_ann none`: keep **weekly** Omega (not annualized).

Only these five metrics are printed. CSVs are saved for weights, weekly portfolio returns (gross), and training history.

---

## Install

```bash
# venv or conda recommended
pip install --upgrade pip
pip install torch pandas numpy openpyxl
# If you plan to use --opt blosam, ensure your local BLOSAM package is importable
```

## Quick run

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_lstm.py \
  --excel_path /path/to/data.xlsx \
  --txt_path /path/to/rf.txt \
  --asset_sheet asset_return \
  --index_sheet index_return \
  --opt sgdm --lookback_L 52 --hold_H 12 --train_ratio 0.7 \
  --lam 8.0 --cov_method ewma --cov_halflife 26 --shrink 0.2 \
  --theta 0.95 --verbose --lr 1e-3
```

# Traditional Baselines

## What it does
- Reads **headerless** weekly data: Excel (`asset_return` = assets, `index_return` = index) + TXT (weekly risk-free).
- Runs a rolling backtest with **52-week in-sample** + **12-week hold**, **rebalancing every 12 weeks**.
- Outputs **out-of-sample** results and **gross, annualized** metrics: **SRp, SoRp, ORp, CSRp, ESRp**.
- Baselines (paper abbreviations): **EW, MI, M–V, L-SSD, LR-ASSD, RMZ-SSD, PK-SSD, CZeSD**.

## Quick run

```bash
pip install pandas numpy cvxpy openpyxl

CUDA_VISIBLE_DEVICES=0 python3 classic_opt.py \
  --excel_path /path/to/data.xlsx \
  --txt_path /path/to/rf.txt \
  --asset_sheet asset_return \
  --index_sheet index_return \
  --methods EW MI M–V L-SSD LR-ASSD RMZ-SSD PK-SSD CZeSD \
  --lookback_L 52 --hold_H 12 \
  --annualize_factor 52 --theta 0.95 \
  --cov_method ewma --cov_halflife 26 --shrink 0.2 \
  --mv_lambda 8.0 --weight_cap 0.2 \
  --omega_ann block \
  --out_dir results_baselines
```

