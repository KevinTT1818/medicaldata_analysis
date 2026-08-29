"""复杂抽样设计下的加权统计。

NHANES 这类调查按不同概率抽样，直接求均值得到的是样本均值而不是人群估计。
这里实现加权的点估计。

**不实现加权的方差与 p 值。** 复杂抽样下的标准误需要 Taylor 线性化并考虑
分层与初级抽样单元，不是给方差加个权重就完事。做不对就不做 —— 算子在加权模式下
一律不出 p 值，并说明原因；想要 p 值就得切到非加权模式，那时的结论只适用于样本本身。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..cohort.builder import WEIGHT_COLUMN

#: 算子不支持加权时给用户的提示。宁可显眼地说明，也不要静默给出非人群估计。
UNWEIGHTED_WARNING = (
    "该数据集带复杂抽样权重，但本算子未做加权。结果只描述样本本身，"
    "不能外推到人群。"
)


def sample_weights(frame: pl.DataFrame) -> np.ndarray | None:
    """从宽表取出抽样权重列。没有这一列的数据集返回 None。"""
    if WEIGHT_COLUMN not in frame.columns:
        return None
    return frame[WEIGHT_COLUMN].cast(pl.Float64, strict=False).to_numpy().astype("float64")


def warn_if_survey_data(frame: pl.DataFrame) -> list[str]:
    """数据集带抽样权重而算子不支持加权时，返回一条警告。"""
    return [UNWEIGHTED_WARNING] if has_weights(sample_weights(frame)) else []


def has_weights(w: np.ndarray | None) -> bool:
    """必须返回 Python bool —— numpy.bool_ 过不了 JSON 序列化，
    payload 要经 FastAPI 下发给前端，也要写进缓存文件。"""
    if w is None:
        return False
    return bool(np.isfinite(w).any() and float(np.nansum(w)) > 0)


def _clean(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """去掉缺失和非正权重。权重为 0 的人（如只参加访谈未参加体检）不参与估计。"""
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[ok], w[ok]


def mean(x: np.ndarray, w: np.ndarray) -> float:
    x, w = _clean(x, w)
    if x.size == 0:
        return float("nan")
    return float(np.sum(w * x) / np.sum(w))


def sd(x: np.ndarray, w: np.ndarray) -> float:
    """加权标准差，估计的是人群离散度，不是均值的标准误。"""
    x, w = _clean(x, w)
    if x.size < 2:
        return float("nan")
    mu = np.sum(w * x) / np.sum(w)
    variance = np.sum(w * (x - mu) ** 2) / np.sum(w)
    return float(np.sqrt(variance))


def quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """加权分位数。

    按取值排序后累计权重，取累计比例首次达到 q 的那个值，
    并在相邻两点间做线性插值（与 numpy.percentile 的 linear 方法思路一致）。
    """
    x, w = _clean(x, w)
    if x.size == 0:
        return float("nan")

    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]

    # 每个观测的代表位置取其权重区间的中点，避免端点偏倚
    positions = (cumulative - 0.5 * w) / total
    if q <= positions[0]:
        return float(x[0])
    if q >= positions[-1]:
        return float(x[-1])
    return float(np.interp(q, positions, x))


def proportion(mask: np.ndarray, w: np.ndarray) -> float:
    """满足条件者在人群中的占比（0–1）。"""
    ok = np.isfinite(w) & (w > 0)
    denominator = float(np.sum(w[ok]))
    if denominator == 0:
        return float("nan")
    return float(np.sum(w[ok & mask]) / denominator)


def effective_n(w: np.ndarray) -> float:
    """Kish 有效样本量：加权后等价于多少个等权观测。

    权重差异越大，有效样本量越小 —— 这是加权的代价，值得在结果里让人看见。
    """
    ok = np.isfinite(w) & (w > 0)
    if not ok.any():
        return 0.0
    weights = w[ok]
    return float(np.sum(weights) ** 2 / np.sum(weights**2))
