"""宽表取值的统一入口。

CDM 反向透视出来的宽表里，测量列是 VARCHAR（数值型和分类型共用一条路径），
诊断和结局是 0/1 整数。各处都要把它们转回可计算的形态，
这里集中一份，避免每个算子各抄一遍转换逻辑而慢慢跑偏。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..cohort.builder import Variable


def as_numeric(frame: pl.DataFrame, variable_id: str) -> np.ndarray:
    """取一列并转成 float64，缺失为 nan。"""
    s = frame[variable_id]
    if s.dtype != pl.Float64:
        s = s.cast(pl.Float64, strict=False)
    return s.to_numpy().astype("float64")


def as_category(frame: pl.DataFrame, variable_id: str, variable: Variable) -> pl.Series:
    """取一列并统一成字符串类别，缺失为 null。

    诊断与结局在宽表里是 0/1，这里映射成「否」「是」，
    好让它们和其他分类变量走同一条渲染与统计路径。
    """
    s = frame[variable_id]
    if variable.source in ("condition", "outcome"):
        return (
            s.cast(pl.Int64, strict=False)
            .replace_strict({1: "是", 0: "否"}, default=None, return_dtype=pl.Utf8)
        )
    return s.cast(pl.Utf8, strict=False)


def levels_of(series: pl.Series, variable: Variable) -> list[str]:
    """类别水平。优先用变量目录里登记的顺序，未登记的补在后面。"""
    present = series.drop_nulls().unique().sort().to_list()
    if variable.source in ("condition", "outcome"):
        return [x for x in ("否", "是") if x in present]
    if variable.levels:
        ordered = [x for x in variable.levels if x in present]
        return ordered + [x for x in present if x not in ordered]
    return present
