"""复杂抽样设计下的方差估计（Taylor 线性化）。

加权的点估计（均值、构成比）不难，难的是标准误 —— 分层多阶段抽样下
不能把观测当成独立同分布，同一个初级抽样单元（PSU）里的人是相关的，
忽略聚类会系统性低估方差，把不显著的结论算成显著。

用的是标准的分层刀切/线性化估计量：先把统计量线性化成每个观测的残差 z，
再按「层内 PSU 之间的离散度」求和：

    V = Σ_h  n_h/(n_h-1)  Σ_i (z_hi· - z̄_h·)²

其中 z_hi· 是第 h 层第 i 个 PSU 内 z 的合计。设计自由度 = PSU 总数 - 层数。
NHANES 2017–2018 是 15 层 × 2 PSU，自由度 15 —— 这也是 NCHS 官方用的自由度。

域（子人群）估计不能先把数据切出来再算 —— 那样某些层/PSU 会整个消失，
方差就错了。正确做法是保留全部 PSU，用指示变量把域外的观测的 z 置零。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class Design:
    """抽样设计：权重 + 分层 + 初级抽样单元。"""

    weight: np.ndarray
    stratum: np.ndarray          # 层编码
    psu: np.ndarray              # PSU 编码（层内唯一即可）
    #: 只有权重、没有分层/PSU 时为 True。此时按「单层、每人一个 PSU」处理，
    #: 等价于有放回抽样的方差估计 —— 点估计仍对，但标准误会偏小。
    approximate: bool = False

    @property
    def n_strata(self) -> int:
        return len(np.unique(self.stratum))

    @property
    def n_psu(self) -> int:
        pairs = np.stack([self.stratum, self.psu], axis=1)
        return len(np.unique(pairs, axis=0))

    @property
    def df(self) -> int:
        """设计自由度 = PSU 总数 − 层数。"""
        return max(self.n_psu - self.n_strata, 0)

    def singleton_strata(self) -> int:
        """只有一个 PSU 的层。它们对方差贡献不了信息，多了就说明队列切得太碎。"""
        count = 0
        for h in np.unique(self.stratum):
            mask = self.stratum == h
            if len(np.unique(self.psu[mask])) < 2:
                count += 1
        return count


def build_design(
    weight: np.ndarray,
    stratum: np.ndarray | None,
    psu: np.ndarray | None,
) -> Design:
    """从宽表的三列构造 Design。缺分层/PSU 时退化为近似估计。"""
    n = weight.size
    if stratum is None or psu is None:
        return Design(
            weight=weight,
            stratum=np.zeros(n, dtype=np.int64),
            psu=np.arange(n, dtype=np.int64),
            approximate=True,
        )

    def encode(values: np.ndarray) -> np.ndarray:
        return np.unique(values.astype(str), return_inverse=True)[1]

    return Design(weight=weight, stratum=encode(stratum), psu=encode(psu))


def _psu_totals(z: np.ndarray, design: Design) -> list[np.ndarray]:
    """按层归集每个 PSU 内的 z 合计。z 可以是 (n,) 或 (n, m)。"""
    if z.ndim == 1:
        z = z[:, None]
    blocks: list[np.ndarray] = []
    for h in np.unique(design.stratum):
        rows = design.stratum == h
        psus = np.unique(design.psu[rows])
        totals = np.stack([z[rows & (design.psu == i)].sum(axis=0) for i in psus])
        blocks.append(totals)
    return blocks


def covariance(z: np.ndarray, design: Design) -> np.ndarray:
    """线性化残差的设计方差（协方差矩阵）。z 为 (n,) 或 (n, m)。"""
    single = z.ndim == 1
    matrix = z[:, None] if single else z
    m = matrix.shape[1]
    total = np.zeros((m, m))

    for totals in _psu_totals(matrix, design):
        n_h = totals.shape[0]
        if n_h < 2:
            # 单 PSU 层贡献不了层内离散度，跳过（数量由 singleton_strata 报出）
            continue
        deviations = totals - totals.mean(axis=0)
        total += (n_h / (n_h - 1)) * (deviations.T @ deviations)

    return total[0, 0] if single else total


@dataclass
class Estimate:
    value: float
    se: float
    df: int
    ci_low: float
    ci_high: float
    #: 设计效应：相对简单随机抽样，方差被聚类与不等权重放大了多少倍
    deff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": round(self.value, 6),
            "se": round(self.se, 6),
            "df": self.df,
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "deff": round(self.deff, 3) if self.deff is not None else None,
        }


def _ci(value: float, se: float, df: int, level: float) -> tuple[float, float]:
    if df <= 0 or not np.isfinite(se):
        return float("nan"), float("nan")
    t = float(stats.t.ppf(0.5 + level / 2, df))
    return value - t * se, value + t * se


def domain_mean(
    y: np.ndarray, design: Design, domain: np.ndarray | None = None,
    conf_level: float = 0.95,
) -> Estimate:
    """域内加权均值及其设计标准误。

    域外的观测保留在样本里但 z 置零 —— 直接把数据切掉会让某些 PSU 消失，
    方差就不再是这个设计下的方差了。
    """
    w = design.weight
    inside = np.ones(y.size, dtype=bool) if domain is None else domain
    usable = inside & np.isfinite(y) & np.isfinite(w) & (w > 0)

    weight_total = float(w[usable].sum())
    if weight_total == 0:
        return Estimate(float("nan"), float("nan"), design.df, float("nan"), float("nan"))

    value = float((w[usable] * y[usable]).sum() / weight_total)

    z = np.zeros(y.size)
    z[usable] = w[usable] * (y[usable] - value) / weight_total
    variance = float(covariance(z, design))
    se = float(np.sqrt(max(variance, 0.0)))

    # 设计效应：与「按简单随机抽样、同样的加权点估计」相比方差放大了多少
    n = int(usable.sum())
    srs_variance = float("nan")
    if n > 1:
        spread = np.average((y[usable] - value) ** 2, weights=w[usable])
        srs_variance = spread / n
    deff = float(variance / srs_variance) if srs_variance and srs_variance > 0 else None

    low, high = _ci(value, se, design.df, conf_level)
    return Estimate(value, se, design.df, low, high, deff)


def domain_proportion(
    indicator: np.ndarray, design: Design, domain: np.ndarray | None = None,
    conf_level: float = 0.95,
) -> Estimate:
    """域内加权构成比。构成比就是 0/1 变量的均值。"""
    return domain_mean(indicator.astype("float64"), design, domain, conf_level)


def _contrast_residuals(
    y: np.ndarray, design: Design, domains: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """各域均值相对第一个域的差，以及对应的线性化残差矩阵。"""
    w = design.weight
    values: list[float] = []
    residuals: list[np.ndarray] = []

    for domain in domains:
        usable = domain & np.isfinite(y) & np.isfinite(w) & (w > 0)
        weight_total = float(w[usable].sum())
        if weight_total == 0:
            values.append(float("nan"))
            residuals.append(np.zeros(y.size))
            continue
        value = float((w[usable] * y[usable]).sum() / weight_total)
        z = np.zeros(y.size)
        z[usable] = w[usable] * (y[usable] - value) / weight_total
        values.append(value)
        residuals.append(z)

    contrasts = np.array([values[i] - values[0] for i in range(1, len(values))])
    Z = np.stack([residuals[i] - residuals[0] for i in range(1, len(values))], axis=1)
    return contrasts, Z


def wald_test(
    y: np.ndarray, design: Design, domains: list[np.ndarray], conf_level: float = 0.95
) -> dict[str, Any]:
    """各域均值（或构成比）是否相等的设计校正 Wald 检验。

    对比向量取「各域 − 第一个域」，共 m = 域数 − 1 个。
    Wald = d' V⁻¹ d，再按 F 近似换算：F = W (df − m + 1) / (df m)。
    """
    m = len(domains) - 1
    if m < 1:
        return {"p": None, "test": None, "detail": "只有一个组，无从比较"}

    df = design.df
    if df <= 0:
        return {"p": None, "test": None, "detail": "抽样设计自由度为 0，无法检验"}
    if m > df:
        return {"p": None, "test": None,
                "detail": f"对比数 {m} 超过设计自由度 {df}，该检验不可估计"}

    contrasts, Z = _contrast_residuals(y, design, domains)
    if not np.all(np.isfinite(contrasts)):
        return {"p": None, "test": None, "detail": "有分组在该变量上没有可用观测"}

    V = np.atleast_2d(covariance(Z, design))
    try:
        solved = np.linalg.solve(V, contrasts)
    except np.linalg.LinAlgError:
        return {"p": None, "test": None, "detail": "对比的协方差矩阵奇异，无法检验"}

    wald = float(contrasts @ solved)
    if not np.isfinite(wald) or wald < 0:
        return {"p": None, "test": None, "detail": "Wald 统计量不可用"}

    f_stat = wald * (df - m + 1) / (df * m)
    p = float(stats.f.sf(f_stat, m, df - m + 1))
    return {
        "p": p,
        "test": "设计校正 Wald 检验",
        "detail": None,
        "wald": round(wald, 4),
        "f": round(float(f_stat), 4),
        "df_num": m,
        "df_den": df - m + 1,
    }


def categorical_wald_test(
    categories: np.ndarray, levels: list[str], design: Design,
    domains: list[np.ndarray],
) -> dict[str, Any]:
    """分类变量在各域间构成比是否相同的设计校正 Wald 检验。

    对比是 (域数−1)×(水平数−1) 维 —— 最后一个水平由其余水平决定，不独立。
    """
    g, k = len(domains), len(levels)
    m = (g - 1) * (k - 1)
    if m < 1:
        return {"p": None, "test": None, "detail": "组数或水平数不足，无从比较"}

    df = design.df
    if df <= 0:
        return {"p": None, "test": None, "detail": "抽样设计自由度为 0，无法检验"}
    if m > df:
        return {"p": None, "test": None,
                "detail": f"对比数 {m} 超过设计自由度 {df}，该检验不可估计"
                          f"（{g} 组 × {k} 个水平）"}

    contrast_parts, Z_parts = [], []
    for level in levels[:-1]:
        indicator = (categories == level).astype("float64")
        contrasts, Z = _contrast_residuals(indicator, design, domains)
        if not np.all(np.isfinite(contrasts)):
            return {"p": None, "test": None, "detail": "有分组在该变量上没有可用观测"}
        contrast_parts.append(contrasts)
        Z_parts.append(Z)

    contrasts = np.concatenate(contrast_parts)
    Z = np.concatenate(Z_parts, axis=1)
    V = np.atleast_2d(covariance(Z, design))

    try:
        solved = np.linalg.solve(V, contrasts)
    except np.linalg.LinAlgError:
        return {"p": None, "test": None, "detail": "对比的协方差矩阵奇异，无法检验"}

    wald = float(contrasts @ solved)
    if not np.isfinite(wald) or wald < 0:
        return {"p": None, "test": None, "detail": "Wald 统计量不可用"}

    f_stat = wald * (df - m + 1) / (df * m)
    return {
        "p": float(stats.f.sf(f_stat, m, df - m + 1)),
        "test": "设计校正 Wald 检验",
        "detail": None,
        "wald": round(wald, 4),
        "f": round(float(f_stat), 4),
        "df_num": m,
        "df_den": df - m + 1,
    }
