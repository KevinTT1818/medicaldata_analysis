"""设计矩阵构建：CDM 宽表 -> 回归模型能吃的数值矩阵。

Cox 和 logistic 共用。分类变量做哑变量编码，第一个水平作参照组，
并把「哪一列对应哪个变量的哪个水平」的元信息带出去，
好让森林图的行标签能写成「胸痛类型：无症状（参照：典型心绞痛）」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from ..cdm import values
from ..cohort.builder import Variable


@dataclass
class Term:
    column: str          # 设计矩阵里的列名
    variable: str        # 来源变量 ID
    label: str           # 森林图上显示的行标签
    reference: str | None = None   # 分类变量的参照水平


class DesignError(ValueError):
    pass


_numeric = values.as_numeric
_category = values.as_category


def build_design(
    frame: pl.DataFrame,
    catalog: dict[str, Variable],
    covariates: list[str],
) -> tuple[pd.DataFrame, list[Term], np.ndarray]:
    """返回 (设计矩阵, 列元信息, 完整行掩码)。

    掩码为 False 的行含缺失值，调用方据此做 complete-case 处理并报告删了多少。
    """
    columns: dict[str, np.ndarray] = {}
    terms: list[Term] = []

    for vid in covariates:
        v = catalog[vid]
        if v.kind == "continuous":
            columns[vid] = _numeric(frame, vid)
            unit = f"（{v.unit}）" if v.unit else ""
            terms.append(Term(column=vid, variable=vid, label=f"{v.label}{unit}"))
            continue

        cats = _category(frame, vid, v)
        present = cats.drop_nulls().unique().sort().to_list()
        if len(present) < 2:
            raise DesignError(f"「{v.label}」在当前队列里只有一个水平，无法进入模型")

        # 参照组：条件/结局用「否」，其余用排序后的第一个水平
        reference = "否" if "否" in present else present[0]
        arr = cats.to_numpy()
        isnull = cats.is_null().to_numpy()

        for level in present:
            if level == reference:
                continue
            col = f"{vid}={level}"
            enc = (arr == level).astype("float64")
            enc[isnull] = np.nan
            columns[col] = enc
            terms.append(Term(
                column=col, variable=vid,
                label=f"{v.label}：{level}", reference=reference,
            ))

    if not columns:
        raise DesignError("至少要选一个协变量")

    design = pd.DataFrame(columns)
    complete = design.notna().all(axis=1).to_numpy()

    # 零方差列进不了模型，会让矩阵奇异
    zero_var = [
        t.label for t in terms
        if np.nanstd(design[t.column].to_numpy()[complete]) == 0
    ]
    if zero_var:
        raise DesignError(f"这些变量在完整病例里没有变异，无法估计：{'、'.join(zero_var)}")

    return design, terms, complete
