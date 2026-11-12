"""趋势检验：OLS斜率、普通t检验、Newey-West稳健检验和BH多重检验。

这些是本项目的方法选择，不是老师指定的软件或唯一统计方法。
HAC使用Bartlett权重和n/(n-2)小样本协方差修正，p值采用渐近正态分布。
公式与statsmodels cov_hac实现对应；置信区间是近似95%区间。
"""
import numpy as np
from scipy.stats import linregress, norm


def trend_fit(years, values, lags=None):
    """完整连续年序列的线性趋势；返回每十年斜率和两种p值。"""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 3 or y.shape != x.shape or not np.isfinite(y).all():
        raise ValueError('Trend requires at least three complete annual observations.')
    if not np.all(np.diff(x) == 1):
        raise ValueError('Years must be consecutive and unique.')
    n = len(x)
    lag = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else int(lags)
    if not 0 <= lag < n:
        raise ValueError('HAC lag must be between zero and n-1.')
    xc = x - x.mean()
    design = np.column_stack([np.ones(n), xc])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    ordinary = linregress(x, y) if np.ptp(y) else None
    # 三明治协方差：bread @ meat @ bread；时间相邻残差的协方差也进入meat。
    xu = design * residual[:, None]
    meat = xu.T @ xu
    for k in range(1, lag + 1):
        cross = xu[k:].T @ xu[:-k]
        meat += (1 - k / (lag + 1)) * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    cov = bread @ meat @ bread * n / (n - 2)
    se = float(np.sqrt(max(cov[1, 1], 0)))
    # 恰为常数或恰为直线的人工数据需明确处理零残差，防止0/0。
    if np.max(np.abs(residual)) <= 1e-10:
        p_hac = 1.0 if abs(beta[1]) < 1e-10 else 0.0
    else:
        p_hac = float(2 * norm.sf(abs(beta[1] / se))) if se else 0.0
    rho = float(np.corrcoef(residual[1:], residual[:-1])[0, 1]) if np.std(residual) > 1e-10 and np.std(residual[1:]) > 0 and np.std(residual[:-1]) > 0 else 0.0
    return {'slope_per_decade': float(beta[1] * 10),
            'intercept_at_mean_year': float(beta[0]),
            'p_ols': float(ordinary.pvalue) if ordinary else 1.0,
            'p_hac': p_hac, 'hac_se_per_decade': se * 10,
            'ci95_low': float((beta[1] - 1.959963984540054 * se) * 10),
            'ci95_high': float((beta[1] + 1.959963984540054 * se) * 10),
            'residual_lag1_correlation': rho, 'hac_lags': lag}


def bh_qvalues(pvalues):
    """Benjamini-Hochberg q值；用于格点多重比较的补充结果，不代替单点p值。"""
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('p values must be finite and between 0 and 1.')
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    corrected = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    q = np.empty_like(p)
    q[order] = corrected
    return q
