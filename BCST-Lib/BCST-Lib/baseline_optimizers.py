# baseline_optimizers.py
import numpy as np
import cvxpy as cp

def _ensure_2d(R):
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    return R

# -----------------------------
# 工具：构造参考组合时间序列
# -----------------------------
def reference_series(R, ref="EW", ref_weights=None):
    """
    R: [T, N] 资产收益
    ref:
      - "EW": 等权
      - "weights": 使用 ref_weights（长度 N）
      - "asset:k": 第 k 列作为参考（字符串，如 "asset:0"）
    返回：参考组合的收益序列 r_ref: [T]
    """
    R = _ensure_2d(R)
    T, N = R.shape
    if ref == "EW":
        w = np.ones(N)/N
        return R @ w
    elif isinstance(ref, str) and ref.startswith("asset:"):
        k = int(ref.split(":")[1])
        e = np.zeros(N); e[k] = 1.0
        return R @ e
    elif ref == "weights":
        assert ref_weights is not None and len(ref_weights) == N
        w = np.asarray(ref_weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        return R @ w
    else:
        raise ValueError("ref 仅支持 'EW' / 'asset:k' / 'weights'.")

# -----------------------------
# Mean-Variance (两种用法)
# -----------------------------
def meanvar_max_utility(mu, V, gamma=1.0, allow_short=False):
    """
    最大化 U = mu^T x - (gamma/2) x^T V x
    约束：1^T x = 1，(非负可选)
    """
    mu = np.asarray(mu, dtype=float).ravel()
    V = np.asarray(V, dtype=float)
    n = len(mu)
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1.0]
    if not allow_short:
        cons.append(x >= 0)
    obj = cp.Maximize(mu @ x - 0.5 * gamma * cp.quad_form(x, V))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)
    if x.value is None:
        raise RuntimeError("MeanVar solver failed.")
    return np.array(x.value).ravel()

def meanvar_min_var_given_ret(mu, V, r_target, allow_short=False):
    """
    给定目标收益 r_target，最小方差
    min x^T V x
    s.t. mu^T x >= r_target, 1^T x = 1, (非负可选)
    """
    mu = np.asarray(mu, dtype=float).ravel()
    V = np.asarray(V, dtype=float)
    n = len(mu)
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1.0, mu @ x >= float(r_target)]
    if not allow_short:
        cons.append(x >= 0)
    obj = cp.Minimize(cp.quad_form(x, V))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)
    if x.value is None:
        raise RuntimeError("MeanVar(target return) solver failed.")
    return np.array(x.value).ravel()

# -----------------------------
# L-SSD / KP-SSD / CZ-εSD
# （LPM(1) 离散阈值近似：线性规划）
# -----------------------------
def _build_tau_grid(series, q=50):
    # series: 参考与候选组合的收益样本，用于确定阈值网格 tau
    # 采用分位数网格，覆盖尾部
    qs = np.linspace(0.0, 1.0, q)
    taus = np.quantile(series, qs)
    # 去重+平滑
    taus = np.unique(np.round(taus, 10))
    return taus

def _lpm1(y, tau):
    # 返回向量 y 的 LPM1: mean(max(tau - y, 0))
    return cp.sum(cp.pos(tau - y)) / y.shape[0]

def ssd_lp_max_return(R, ref="EW", ref_weights=None, q=50,
                      allow_short=False, epsilon=0.0):
    """
    线性 SSD（LPM(1) 约束离散化）：
      max mu^T x
      s.t. 对所有 tau∈TauGrid:  LPM1(Rx, tau) <= LPM1(R_ref, tau) + epsilon
           1^T x = 1, (x>=0 可选)
    参数：
      R: [T, N] 资产收益
      ref: "EW" / "asset:k" / "weights"
      epsilon: CZ-εSD 中的 ε（默认0表示严格SSD）
    返回：x*
    """
    R = _ensure_2d(R)
    T, N = R.shape
    r_ref = reference_series(R, ref=ref, ref_weights=ref_weights)  # [T]
    mu = np.mean(R, axis=0)

    x = cp.Variable(N)
    cons = [cp.sum(x) == 1.0]
    if not allow_short:
        cons.append(x >= 0)

    Rx = R @ x          # [T] 组合收益
    taus = _build_tau_grid(np.concatenate([r_ref, R.flatten()]), q=q)

    for tau in taus:
        cons.append(_lpm1(Rx, tau) <= np.mean(np.maximum(tau - r_ref, 0.0)) + float(epsilon))

    obj = cp.Maximize(mu @ x)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)
    if x.value is None:
        raise RuntimeError("SSD LP solver failed.")
    return np.array(x.value).ravel()

def L_SSD(R, **kwargs):
    """
    Linear SSD: epsilon=0
    """
    return ssd_lp_max_return(R, epsilon=0.0, **kwargs)

def KP_SSD(R, **kwargs):
    """
    Kuosmanen–Post SSD：这里用同样的 LPM(1) 离散近似框架实现一个实用版本。
    你可通过调 q（阈值密度）与 ref 选择，让它更贴近KP论文的设置。
    """
    return ssd_lp_max_return(R, **kwargs)

def CZ_eSD(R, epsilon=0.001, **kwargs):
    """
    CZ-εSD：在 L-SSD 约束右端加入 ε 松弛
    """
    return ssd_lp_max_return(R, epsilon=float(epsilon), **kwargs)

# -----------------------------
# 指标计算（便于对比）
# -----------------------------
def sharpe_ratio(r, rf=0.0, ddof=0):
    r = np.asarray(r, dtype=float)
    ex = r.mean() - rf
    sd = r.std(ddof=ddof)
    return ex / (sd + 1e-12)

def sortino_ratio(r, rb=0.0):
    r = np.asarray(r, dtype=float)
    dd = np.sqrt(np.mean(np.maximum(rb - r, 0.0)**2))
    return (r.mean() - rb) / (dd + 1e-12)

def annualized_return(r, periods_per_year=52):
    r = np.asarray(r, dtype=float)
    return r.mean() * periods_per_year  # 简化；也可用乘积年化
