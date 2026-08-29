"""描述性统计算子：Baseline Table 1、缺失情况。"""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from scipy import stats

from ..cdm import values
from ..cohort.builder import Variable
from . import weights as W
from .base import Analysis, AnalysisContext, AnalysisResult
from .registry import register
from . import widgets

# Shapiro-Wilk 在 n > 5000 时不可靠（且几乎必然拒绝正态）。
# 超过这个量级依中心极限定理按参数法处理。
SHAPIRO_MAX_N = 5000

OVERALL = "__overall__"


# ---------------------------------------------------------------- 取数辅助

_sample_weights = W.sample_weights


_as_numeric = values.as_numeric
_as_category = values.as_category
_levels_of = values.levels_of


def _digits(values: np.ndarray) -> int:
    """按量级选小数位：大数 1 位，小数 2 位。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1
    return 1 if float(np.max(np.abs(finite))) >= 50 else 2


def _fmt(x: float | None, d: int) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


# ---------------------------------------------------------------- 统计

def _is_normal(arrays: list[np.ndarray], alpha: float) -> bool:
    """各组是否都通过正态性检验。任一组不通过就走非参数法。"""
    for a in arrays:
        n = a.size
        if n < 3:
            return False
        if n > SHAPIRO_MAX_N:
            continue  # 大样本按中心极限定理处理
        if stats.shapiro(a).pvalue <= alpha:
            return False
    return True


def _continuous_row(
    frame: pl.DataFrame, vid: str, v: Variable,
    group_masks: list[tuple[str, np.ndarray]], alpha: float,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    vals = _as_numeric(frame, vid)
    valid = np.isfinite(vals)
    n_missing = int((~valid).sum())
    d = _digits(vals)
    weighted = sample_weights is not None

    per_group = [(label, vals[mask & valid]) for label, mask in group_masks]
    testable = [a for _, a in per_group if a.size >= 2 and np.ptp(a) > 0]
    # 正态性判断只看分布形状，用未加权样本即可 —— 加权改变的是各观测的代表性，
    # 不改变分布形状。选定统计量之后再按权重计算。
    normal = _is_normal([a for _, a in per_group] or [vals[valid]], alpha)

    def describe_masked(mask: np.ndarray) -> str:
        a = vals[mask]
        if a.size == 0:
            return "—"
        if weighted:
            w = sample_weights[mask]  # type: ignore[index]
            if normal:
                return f"{_fmt(W.mean(a, w), d)} ± {_fmt(W.sd(a, w), d)}"
            return (f"{_fmt(W.quantile(a, w, 0.5), d)} "
                    f"({_fmt(W.quantile(a, w, 0.25), d)}–{_fmt(W.quantile(a, w, 0.75), d)})")
        if normal:
            sd = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
            return f"{_fmt(float(np.mean(a)), d)} ± {_fmt(sd, d)}"
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        return f"{_fmt(float(med), d)} ({_fmt(float(q1), d)}–{_fmt(float(q3), d)})"

    cells = {OVERALL: describe_masked(valid)}
    for label, mask in group_masks:
        cells[label] = describe_masked(mask & valid)

    p: float | None = None
    test: str | None = None
    warnings: list[str] = []

    # 加权模式下不出 p 值：复杂抽样的标准误要 Taylor 线性化 + 分层/PSU，
    # 拿未加权的检验统计量配加权的点估计是自相矛盾的
    if weighted:
        pass
    elif len(testable) >= 2 and len(testable) == len(per_group):
        if len(testable) == 2:
            if normal:
                p = float(stats.ttest_ind(*testable, equal_var=False).pvalue)
                test = "Welch t 检验"
            else:
                p = float(stats.mannwhitneyu(*testable, alternative="two-sided").pvalue)
                test = "Mann-Whitney U"
        else:
            if normal:
                p = float(stats.f_oneway(*testable).pvalue)
                test = "单因素方差分析"
            else:
                p = float(stats.kruskal(*testable).pvalue)
                test = "Kruskal-Wallis"
    elif group_masks and not weighted and len(testable) != len(per_group):
        warnings.append("有分组样本量不足或方差为零，未做检验")

    return {
        "variable": vid,
        "label": v.label,
        "unit": v.unit,
        "kind": "continuous",
        "stat": "mean_sd" if normal else "median_iqr",
        "n_missing": n_missing,
        "cells": cells,
        "levels": None,
        "p": p,
        "p_adj": None,
        "test": test,
        "warnings": warnings,
    }


def _categorical_row(
    frame: pl.DataFrame, vid: str, v: Variable,
    group_masks: list[tuple[str, np.ndarray]],
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    weighted = sample_weights is not None
    cats = _as_category(frame, vid, v)
    arr = cats.to_numpy()
    notnull = cats.is_not_null().to_numpy()
    n_missing = int((~notnull).sum())
    levels = _levels_of(cats, v)

    # 二分类的诊断/结局按惯例只显示阳性那一行
    single = v.kind == "binary" and v.source in ("condition", "outcome") and "是" in levels
    shown = ["是"] if single else levels

    def cell(mask: np.ndarray, level: str) -> str:
        sel = mask & notnull
        denom = int(sel.sum())
        if denom == 0:
            return "—"
        n = int((arr[sel] == level).sum())
        if weighted:
            # 例数是实际观测数，百分比是人群估计 —— NHANES 类研究的惯例写法
            pct = W.proportion(arr[sel] == level, sample_weights[sel]) * 100  # type: ignore[index]
            return f"{n} ({pct:.1f})"
        return f"{n} ({n / denom * 100:.1f})"

    all_mask = np.ones(frame.height, dtype=bool)
    level_rows = [
        {
            "label": lv,
            "cells": {OVERALL: cell(all_mask, lv),
                      **{label: cell(mask, lv) for label, mask in group_masks}},
        }
        for lv in shown
    ]

    p: float | None = None
    test: str | None = None
    warnings: list[str] = []

    if group_masks and len(levels) >= 2 and not weighted:
        table = np.array([
            [int((arr[mask & notnull] == lv).sum()) for _, mask in group_masks]
            for lv in levels
        ], dtype=float)
        # 去掉全零的行/列，否则 chi2 会报错
        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]

        if table.shape[0] >= 2 and table.shape[1] >= 2:
            res = stats.chi2_contingency(table)
            min_expected = float(res.expected_freq.min())
            if min_expected < 5:
                if table.shape == (2, 2):
                    p = float(stats.fisher_exact(table.astype(int)).pvalue)
                    test = "Fisher 精确检验"
                else:
                    p = float(res.pvalue)
                    test = "卡方检验"
                    warnings.append(
                        f"最小期望频数 {min_expected:.1f} < 5，卡方结果可能不可靠"
                    )
            else:
                p = float(res.pvalue)
                test = "卡方检验"

    return {
        "variable": vid,
        "label": v.label,
        "unit": None,
        "kind": "categorical",
        "stat": "n_pct",
        "n_missing": n_missing,
        "cells": {},
        "levels": level_rows,
        "p": p,
        "p_adj": None,
        "test": test,
        "warnings": warnings,
    }


# ---------------------------------------------------------------- 算子

@register
class BaselineTable(Analysis):
    id = "describe.baseline_table"
    label = "Baseline / Table 1"
    description = (
        "临床研究的基线特征表。自动判断连续/分类变量，自动做正态性检验并据此"
        "选择参数或非参数方法，输出 n、集中趋势、离散程度、构成比与 p 值。"
    )
    result_kind = "baseline_table"

    class Params(BaseModel):
        variables: list[str] = Field(..., json_schema_extra=widgets.variables(
            "纳入变量", "出现在表格行上的变量"))
        group_by: str | None = Field(None, json_schema_extra=widgets.variable(
            "分组变量", "留空则只出总体列", widgets.GROUPING))
        weighting: Literal["auto", "weighted", "unweighted"] = Field(
            "auto", title="抽样权重",
            description="调查类数据必须加权才是人群估计；加权模式不出 p 值",
            json_schema_extra={"x-enum-labels": {
                "auto": "自动（有权重就加权）",
                "weighted": "强制加权",
                "unweighted": "不加权（仅描述样本）",
            }})
        p_adjust: Literal["none", "bonferroni", "fdr_bh"] = Field(
            "fdr_bh", title="多重比较校正",
            description="一次做几十个检验时必须校正，否则假阳性必然出现",
            json_schema_extra={"x-enum-labels": {
                "fdr_bh": "Benjamini-Hochberg FDR",
                "bonferroni": "Bonferroni",
                "none": "不校正",
            }})
        normality_alpha: float = Field(
            0.05, ge=0.001, le=0.2, title="正态性检验水平",
            description="各组都通过才用参数方法，否则走非参数")

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        p: BaselineTable.Params = ctx.params  # type: ignore[assignment]
        frame = ctx.frame

        raw_weights = _sample_weights(frame)
        available = W.has_weights(raw_weights)
        if p.weighting == "weighted" and not available:
            raise ValueError("该数据集没有抽样权重，无法做加权估计")
        use_weights = raw_weights if (available and p.weighting != "unweighted") else None
        weighted = use_weights is not None

        group_masks: list[tuple[str, np.ndarray]] = []
        groups_meta: list[dict[str, Any]] = []

        if p.group_by:
            gvar = ctx.catalog[p.group_by]
            gcats = _as_category(frame, p.group_by, gvar)
            garr = gcats.to_numpy()
            for lv in _levels_of(gcats, gvar):
                mask = garr == lv
                group_masks.append((lv, mask))
                groups_meta.append({
                    "key": lv, "label": lv, "n": int(mask.sum()),
                    "weighted_pct": (
                        round(W.proportion(mask, use_weights) * 100, 1)
                        if weighted else None
                    ),
                })

        analysis_vars = [v for v in p.variables if v != p.group_by]
        rows: list[dict[str, Any]] = []

        for i, vid in enumerate(analysis_vars):
            ctx.progress(f"计算 {ctx.catalog[vid].label}", 0.1 + 0.8 * i / max(len(analysis_vars), 1))
            v = ctx.catalog[vid]
            if v.kind == "continuous":
                rows.append(_continuous_row(
                    frame, vid, v, group_masks, p.normality_alpha, use_weights))
            else:
                rows.append(_categorical_row(frame, vid, v, group_masks, use_weights))

        # --- 多重比较校正 ---
        idx = [i for i, r in enumerate(rows) if r["p"] is not None]
        ps = [rows[i]["p"] for i in idx]
        if ps and p.p_adjust != "none":
            if p.p_adjust == "fdr_bh":
                adj = list(stats.false_discovery_control(np.array(ps), method="bh"))
            else:
                adj = [min(x * len(ps), 1.0) for x in ps]
            for i, a in zip(idx, adj):
                rows[i]["p_adj"] = float(a)

        notes: list[str] = []
        analysis_warnings: list[str] = []

        if weighted:
            notes.append(
                "已按抽样权重给出人群估计。分类变量括号内是加权后的人群构成比，"
                "括号外的例数仍是实际观测数。"
            )
            notes.append(
                "加权模式不出 p 值：复杂抽样下的标准误需要 Taylor 线性化并考虑分层与"
                "初级抽样单元，本版本未实现。需要 p 值请切到「不加权」，"
                "但那时的结论只适用于样本本身。"
            )
            notes.append(
                f"Kish 有效样本量 {W.effective_n(use_weights):.0f}"
                f"（实际 {frame.height} 例）—— 权重差异越大，有效信息越少。"
            )
        elif available:
            analysis_warnings.append(
                "该数据集带复杂抽样权重，但本次未加权。结果只描述样本本身，"
                "不能外推到人群。"
            )

        if ps:
            method_label = {"fdr_bh": "Benjamini-Hochberg FDR",
                            "bonferroni": "Bonferroni", "none": "未校正"}[p.p_adjust]
            notes.append(
                f"共做了 {len(ps)} 次假设检验，已按 {method_label} 校正。"
                if p.p_adjust != "none" else
                f"共做了 {len(ps)} 次假设检验，未做多重比较校正，p 值仅供描述。"
            )
        if any(r["n_missing"] for r in rows):
            notes.append("含缺失值的变量按逐变量剔除（pairwise deletion）处理，缺失数已单列。")

        return AnalysisResult(
            kind=self.result_kind,
            payload={
                "kind": self.result_kind,
                "overall_n": frame.height,
                "groups": groups_meta,
                "group_by": p.group_by,
                "group_by_label": ctx.catalog[p.group_by].label if p.group_by else None,
                "rows": rows,
                "p_adjust": p.p_adjust,
                "weighted": weighted,
                "weights_available": available,
                "effective_n": (
                    round(W.effective_n(use_weights), 1) if weighted else None
                ),
                "notes": notes,
            },
            warnings=analysis_warnings + [w for r in rows for w in r["warnings"]],
        )


@register
class Missingness(Analysis):
    id = "describe.missingness"
    label = "缺失情况"
    description = "逐变量缺失率，以及缺失记录的共现情况。"
    result_kind = "missingness"

    class Params(BaseModel):
        variables: list[str] = Field(..., json_schema_extra=widgets.variables(
            "检查变量", "逐个统计缺失率"))

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        frame, n = ctx.frame, ctx.frame.height
        vars_ = ctx.params.variables  # type: ignore[attr-defined]

        rows = []
        for vid in vars_:
            miss = int(frame[vid].is_null().sum())
            rows.append({
                "variable": vid,
                "label": ctx.catalog[vid].label,
                "n_missing": miss,
                "pct_missing": round(miss / n * 100, 2) if n else 0.0,
            })
        rows.sort(key=lambda r: -r["n_missing"])

        null_matrix = np.column_stack([frame[v].is_null().to_numpy() for v in vars_]) \
            if vars_ else np.zeros((n, 0), dtype=bool)
        per_person = null_matrix.sum(axis=1)
        complete = int((per_person == 0).sum())

        return AnalysisResult(
            kind=self.result_kind,
            payload={
                "kind": self.result_kind,
                "overall_n": n,
                "rows": rows,
                "complete_cases": complete,
                "complete_pct": round(complete / n * 100, 2) if n else 0.0,
            },
        )


@register
class Distribution(Analysis):
    id = "describe.distribution"
    label = "变量分布"
    description = (
        "连续变量出直方图与箱线图（Tukey 1.5×IQR 界定离群点），"
        "分类变量出构成比。可按分组并排对比。"
    )
    result_kind = "distribution"

    class Params(BaseModel):
        variables: list[str] = Field(..., json_schema_extra=widgets.variables(
            "变量", "每个变量出一块图"))
        group_by: str | None = Field(None, json_schema_extra=widgets.variable(
            "分组变量", "留空则只看总体", widgets.GROUPING))
        bins: int = Field(20, ge=5, le=60, title="分箱数",
                          description="各组共用同一套分箱边界，才能并排比较")

    @staticmethod
    def _box(a: np.ndarray) -> dict[str, Any]:
        """Tukey 箱线图五数概括 + 离群点。"""
        q1, med, q3 = (float(x) for x in np.percentile(a, [25, 50, 75]))
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        inside = a[(a >= lo_fence) & (a <= hi_fence)]
        outliers = a[(a < lo_fence) | (a > hi_fence)]
        return {
            "min": float(inside.min()) if inside.size else float(a.min()),
            "q1": q1,
            "median": med,
            "q3": q3,
            "max": float(inside.max()) if inside.size else float(a.max()),
            "outliers": [float(x) for x in np.sort(outliers)[:200]],
            "mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)) if a.size > 1 else None,
            "n": int(a.size),
        }

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        p: Distribution.Params = ctx.params  # type: ignore[assignment]
        frame = ctx.frame

        if p.group_by:
            gvar = ctx.catalog[p.group_by]
            gcats = _as_category(frame, p.group_by, gvar)
            garr = gcats.to_numpy()
            groups = [(lv, garr == lv) for lv in _levels_of(gcats, gvar)]
        else:
            groups = [("总体", np.ones(frame.height, dtype=bool))]

        panels: list[dict[str, Any]] = []
        targets = [v for v in p.variables if v != p.group_by]

        for i, vid in enumerate(targets):
            v = ctx.catalog[vid]
            ctx.progress(f"计算 {v.label} 的分布", 0.1 + 0.8 * i / max(len(targets), 1))

            if v.kind == "continuous":
                vals = _as_numeric(frame, vid)
                finite = np.isfinite(vals)
                if not finite.any():
                    continue
                # 各组共用同一套分箱边界，否则并排的直方图没法比
                edges = np.histogram_bin_edges(vals[finite], bins=p.bins)
                panels.append({
                    "variable": vid,
                    "label": v.label,
                    "unit": v.unit,
                    "kind": "continuous",
                    "bin_edges": [round(float(x), 4) for x in edges],
                    "series": [
                        {
                            "name": name,
                            "counts": [int(c) for c in
                                       np.histogram(vals[mask & finite], bins=edges)[0]],
                            "box": self._box(vals[mask & finite]),
                        }
                        for name, mask in groups if (mask & finite).any()
                    ],
                    "n_missing": int((~finite).sum()),
                })
            else:
                cats = _as_category(frame, vid, v)
                arr = cats.to_numpy()
                notnull = cats.is_not_null().to_numpy()
                levels = _levels_of(cats, v)
                panels.append({
                    "variable": vid,
                    "label": v.label,
                    "unit": None,
                    "kind": "categorical",
                    "levels": levels,
                    "series": [
                        {
                            "name": name,
                            "counts": [int((arr[mask & notnull] == lv).sum()) for lv in levels],
                            "total": int((mask & notnull).sum()),
                        }
                        for name, mask in groups
                    ],
                    "n_missing": int((~notnull).sum()),
                })

        return AnalysisResult(
            kind=self.result_kind,
            payload={
                "kind": self.result_kind,
                "overall_n": frame.height,
                "group_by_label": ctx.catalog[p.group_by].label if p.group_by else None,
                "grouped": bool(p.group_by),
                "panels": panels,
            },
            warnings=W.warn_if_survey_data(frame),
        )
