"""描述性统计算子：Baseline Table 1、缺失情况。"""
from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from scipy import stats

from ..config import MAX_VARIABLES_PER_ANALYSIS
from ..cdm import values
from ..cohort.builder import PSU_COLUMN, STRATUM_COLUMN, Variable
from . import imputation
from . import survey
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
    design: survey.Design | None = None,
) -> dict[str, Any]:
    vals = _as_numeric(frame, vid)
    valid = np.isfinite(vals)
    n_missing = int((~valid).sum())
    d = _digits(vals)
    weighted = sample_weights is not None

    per_group = [(label, vals[mask & valid]) for label, mask in group_masks]
    testable = [a for _, a in per_group if a.size >= 2 and np.ptp(a) > 0]
    # 加权模式一律报均值：设计校正的 Wald 检验比的是各组均值，
    # 表里显示中位数却拿均值去检验，读者对不上。
    # 未加权时仍按正态性在均值和中位数之间选。
    normal = True if weighted else _is_normal(
        [a for _, a in per_group] or [vals[valid]], alpha
    )

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

    def raw_masked(mask: np.ndarray) -> dict[str, Any] | None:
        """单元格背后的数字。合并 m 份插补时用它，格式化字符串没法合并。"""
        a = vals[mask]
        if a.size == 0:
            return None
        if weighted and design is not None:
            estimate = survey.domain_mean(vals, design, mask)
            return {"mean": estimate.value, "var_mean": float(estimate.se**2),
                    "sd": W.sd(a, sample_weights[mask]), "n": int(a.size)}  # type: ignore[index]
        mean = float(np.mean(a))
        sd = float(np.std(a, ddof=1)) if a.size > 1 else float("nan")
        var_mean = float(sd**2 / a.size) if a.size > 1 else float("nan")
        return {"mean": mean, "var_mean": var_mean, "sd": sd, "n": int(a.size)}

    cells = {OVERALL: describe_masked(valid)}
    raw_cells: dict[str, Any] = {OVERALL: raw_masked(valid)}
    for label, mask in group_masks:
        cells[label] = describe_masked(mask & valid)
        raw_cells[label] = raw_masked(mask & valid)

    p: float | None = None
    test: str | None = None
    warnings: list[str] = []
    continuous_chi_square: dict[str, Any] | None = None

    # 加权时走设计校正的 Wald 检验：把统计量线性化后按层内 PSU 的离散度求方差。
    # 用未加权的 t 检验配加权的点估计是自相矛盾的，所以两条路互斥。
    if weighted:
        if design is not None and group_masks:
            outcome = survey.wald_test(
                vals, design, [mask & valid for _, mask in group_masks]
            )
            p, test = outcome["p"], outcome["test"]
            if outcome.get("detail"):
                warnings.append(f"「{v.label}」未做检验：{outcome['detail']}")
    elif len(testable) >= 2 and len(testable) == len(per_group):
        if len(testable) == 2:
            if normal:
                p = float(stats.ttest_ind(*testable, equal_var=False).pvalue)
                test = "Welch t 检验"
            else:
                p = float(stats.mannwhitneyu(*testable, alternative="two-sided").pvalue)
                test = "Mann-Whitney U"
        else:
            k = len(testable) - 1
            if normal:
                outcome = stats.f_oneway(*testable)
                p = float(outcome.pvalue)
                test = "单因素方差分析"
                # F 乘以分子自由度近似服从卡方(k)，供 D2 合并用
                continuous_chi_square = {"chi2": float(outcome.statistic) * k, "k": k}
            else:
                outcome = stats.kruskal(*testable)
                p = float(outcome.pvalue)
                test = "Kruskal-Wallis"
                # Kruskal 的 H 本身就近似卡方(k)
                continuous_chi_square = {"chi2": float(outcome.statistic), "k": k}
    elif group_masks and not weighted and len(testable) != len(per_group):
        warnings.append("有分组样本量不足或方差为零，未做检验")

    # 两组时把均值差和它的方差留出来：合并 m 份插补时走 Rubin 规则比 D2 准
    contrast: dict[str, Any] | None = None
    if len(group_masks) == 2:
        first = raw_cells.get(group_masks[0][0])
        second = raw_cells.get(group_masks[1][0])
        if first and second and np.isfinite(first["var_mean"]) and np.isfinite(second["var_mean"]):
            v1, v2 = first["var_mean"], second["var_mean"]
            n1, n2 = first["n"], second["n"]
            # Welch-Satterthwaite 自由度。合并时要把它交给 Barnard-Rubin，
            # 否则 FMI 趋近 0 时会退化成 z 检验而不是原来的 t 检验。
            welch_df = float("nan")
            if n1 > 1 and n2 > 1 and (v1 + v2) > 0:
                welch_df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
            contrast = {
                "diff": second["mean"] - first["mean"],
                "var_diff": v1 + v2,
                "df": welch_df,
            }

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
        # 未走插补时没有缺失信息占比，但键要在 —— 前端按固定形状渲染
        "fmi": None,
        "warnings": warnings,
        "raw": {"cells": raw_cells, "contrast": contrast,
                "chi_square": continuous_chi_square, "digits": d},
    }


def _categorical_row(
    frame: pl.DataFrame, vid: str, v: Variable,
    group_masks: list[tuple[str, np.ndarray]],
    sample_weights: np.ndarray | None = None,
    design: survey.Design | None = None,
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
    def raw_cell(mask: np.ndarray, level: str) -> dict[str, Any] | None:
        sel = mask & notnull
        denom = int(sel.sum())
        if denom == 0:
            return None
        count = int((arr[sel] == level).sum())
        if weighted and sample_weights is not None:
            proportion = W.proportion(arr[sel] == level, sample_weights[sel])
        else:
            proportion = count / denom
        # 构成比的抽样方差，合并时用
        variance = proportion * (1 - proportion) / denom if denom else float("nan")
        return {"proportion": float(proportion), "var": float(variance),
                "count": count, "denom": denom}

    level_rows = [
        {
            "label": lv,
            "cells": {OVERALL: cell(all_mask, lv),
                      **{label: cell(mask, lv) for label, mask in group_masks}},
        }
        for lv in shown
    ]
    raw_levels = {
        lv: {OVERALL: raw_cell(all_mask, lv),
             **{label: raw_cell(mask, lv) for label, mask in group_masks}}
        for lv in shown
    }

    p: float | None = None
    test: str | None = None
    warnings: list[str] = []
    chi_square: dict[str, Any] | None = None

    if weighted and design is not None and group_masks and len(levels) >= 2:
        outcome = survey.categorical_wald_test(
            arr, levels, design, [mask & notnull for _, mask in group_masks]
        )
        p, test = outcome["p"], outcome["test"]
        if outcome.get("detail"):
            warnings.append(f"「{v.label}」未做检验：{outcome['detail']}")
    elif group_masks and len(levels) >= 2 and not weighted:
        table = np.array([
            [int((arr[mask & notnull] == lv).sum()) for _, mask in group_masks]
            for lv in levels
        ], dtype=float)
        # 去掉全零的行/列，否则 chi2 会报错
        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]

        if table.shape[0] >= 2 and table.shape[1] >= 2:
            res = stats.chi2_contingency(table)
            chi_square = {"chi2": float(res.statistic), "k": int(res.dof)}
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
        "fmi": None,
        "warnings": warnings,
        "raw": {"levels": raw_levels, "chi_square": chi_square, "shown": shown},
    }


# ---------------------------------------------------------------- 合并 m 份插补

def _pool_continuous(variants: list[dict[str, Any]], m: int) -> dict[str, Any]:
    """合并一个连续变量在 m 份插补上的结果。"""
    base = dict(variants[0])
    digits = variants[0]["raw"]["digits"]
    keys = list(variants[0]["raw"]["cells"])

    cells: dict[str, str] = {}
    for key in keys:
        raw = [v["raw"]["cells"].get(key) for v in variants]
        usable = [r for r in raw if r and np.isfinite(r["mean"])]
        if not usable:
            cells[key] = "—"
            continue
        pooled = imputation.pool(
            [r["mean"] for r in usable],
            [r["var_mean"] for r in usable],
            complete_df=float(np.mean([r["n"] for r in usable])) - 1,
        )
        # 标准差是人群离散度，不是估计量的不确定性，取各份平均即可
        sd = float(np.nanmean([r["sd"] for r in usable]))
        cells[key] = f"{_fmt(pooled.estimate, digits)} ± {_fmt(sd, digits)}"

    p_value, fmi, test = _pool_p(variants, m)
    base.update(
        cells=cells,
        # 插补后一律报均值：检验比的是均值，显示中位数读者对不上
        stat="mean_sd",
        p=p_value, p_adj=None, test=test, fmi=fmi,
        warnings=[w for v in variants for w in v["warnings"]][:1],
    )
    base.pop("raw", None)
    return base


def _pool_categorical(variants: list[dict[str, Any]], m: int) -> dict[str, Any]:
    """合并一个分类变量在 m 份插补上的结果。"""
    base = dict(variants[0])
    shown = variants[0]["raw"]["shown"]
    keys = list(next(iter(variants[0]["raw"]["levels"].values())))

    level_rows = []
    fmis: list[float] = []
    for level in shown:
        cells: dict[str, str] = {}
        for key in keys:
            raw = [v["raw"]["levels"][level].get(key) for v in variants]
            usable = [r for r in raw if r]
            if not usable:
                cells[key] = "—"
                continue
            pooled = imputation.pool(
                [r["proportion"] for r in usable],
                [r["var"] for r in usable],
                complete_df=float(np.mean([r["denom"] for r in usable])) - 1,
            )
            fmis.append(pooled.fmi)
            denom = float(np.mean([r["denom"] for r in usable]))
            # 插补后例数不再是整数观测，这里给的是合并构成比换算回去的估计数
            count = int(round(pooled.estimate * denom))
            cells[key] = f"{count} ({pooled.estimate * 100:.1f})"
        level_rows.append({"label": level, "cells": cells})

    p_value, _, test = _pool_p(variants, m)
    base.update(
        levels=level_rows, cells={},
        p=p_value, p_adj=None, test=test,
        fmi=round(max(fmis), 4) if fmis else None,
        warnings=[w for v in variants for w in v["warnings"]][:1],
    )
    base.pop("raw", None)
    return base


def _pool_p(variants: list[dict[str, Any]], m: int) -> tuple[float | None, float | None, str | None]:
    """合并 p 值。

    两组连续变量能拿到均值差和它的方差，走 Rubin 规则（更准）；
    其余情形只有检验统计量，退到 D2 规则。
    """
    contrasts = [v["raw"].get("contrast") for v in variants]
    if all(c is not None for c in contrasts):
        # 传完整数据自由度，Barnard-Rubin 才能在 FMI 趋近 0 时还原成原检验
        complete_df = float(np.nanmean([c.get("df", np.nan) for c in contrasts]))
        pooled = imputation.pool(
            [c["diff"] for c in contrasts],
            [c["var_diff"] for c in contrasts],
            complete_df=complete_df if np.isfinite(complete_df) else None,
        )
        return pooled.p, round(pooled.fmi, 4), "合并 t 检验（Rubin）"

    chi_squares = [v["raw"].get("chi_square") for v in variants]
    if all(c is not None for c in chi_squares):
        k = chi_squares[0]["k"]
        outcome = imputation.pool_test_statistic(
            [c["chi2"] for c in chi_squares], k, m
        )
        if outcome["p"] is not None:
            return outcome["p"], None, "合并 Wald 检验（D2）"
    return None, None, None


def _pool_rows(
    per_imputation: list[list[dict[str, Any]]], m: int
) -> list[dict[str, Any]]:
    pooled: list[dict[str, Any]] = []
    for index in range(len(per_imputation[0])):
        variants = [batch[index] for batch in per_imputation]
        if variants[0]["kind"] == "continuous":
            pooled.append(_pool_continuous(variants, m))
        else:
            pooled.append(_pool_categorical(variants, m))
    return pooled


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
        variables: list[str] = Field(
            ..., min_length=1, max_length=MAX_VARIABLES_PER_ANALYSIS,
            json_schema_extra=widgets.variables(
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
        missing: Literal["observed", "multiple_imputation"] = Field(
            "observed", title="缺失值处理",
            description=(
                "默认按观测值逐变量统计并单列缺失数，这是基线表的通行做法；"
                "选插补可让分母与模型口径一致"
            ),
            json_schema_extra={"x-enum-labels": {
                "observed": "观测值（逐变量剔除，缺失数单列）",
                "multiple_imputation": "多重插补（与模型口径一致）",
            }})
        n_imputations: int = Field(
            5, ge=2, le=20, title="插补份数",
            description="仅在选多重插补时有效")
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

        design = None
        if weighted:
            strata = (frame[STRATUM_COLUMN].to_numpy()
                      if STRATUM_COLUMN in frame.columns else None)
            psus = (frame[PSU_COLUMN].to_numpy()
                    if PSU_COLUMN in frame.columns else None)
            if strata is not None and psus is not None:
                if not (pl.Series(strata).is_not_null().any()
                        and pl.Series(psus).is_not_null().any()):
                    strata = psus = None
            design = survey.build_design(use_weights, strata, psus)

        analysis_warnings: list[str] = []
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

        # 插补路径：在 m 份补全数据上各算一遍，再合并
        impute_report = None
        if p.missing == "multiple_imputation":
            ctx.progress("多重插补", 0.05)
            try:
                frames, impute_report = imputation.prepare_frames(
                    frame, ctx.catalog, analysis_vars, p.missing, p.n_imputations
                )
            except imputation.ImputationError as exc:
                raise ValueError(str(exc)) from exc
        else:
            frames = [frame]

        def build(source: pl.DataFrame) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for vid in analysis_vars:
                v = ctx.catalog[vid]
                if v.kind == "continuous":
                    out.append(_continuous_row(
                        source, vid, v, group_masks, p.normality_alpha,
                        use_weights, design))
                else:
                    out.append(_categorical_row(
                        source, vid, v, group_masks, use_weights, design))
            return out

        if impute_report is None:
            rows = []
            for i, vid in enumerate(analysis_vars):
                ctx.progress(f"计算 {ctx.catalog[vid].label}",
                             0.1 + 0.8 * i / max(len(analysis_vars), 1))
                v = ctx.catalog[vid]
                if v.kind == "continuous":
                    rows.append(_continuous_row(
                        frame, vid, v, group_masks, p.normality_alpha,
                        use_weights, design))
                else:
                    rows.append(_categorical_row(
                        frame, vid, v, group_masks, use_weights, design))
        else:
            per_imputation = []
            for index, source in enumerate(frames):
                ctx.progress(f"统计第 {index + 1}/{len(frames)} 份插补",
                             0.15 + 0.6 * index / max(len(frames), 1))
                per_imputation.append(build(source))
            ctx.progress("按 Rubin 规则合并", 0.8)
            rows = _pool_rows(per_imputation, len(frames))
            # 插补后的宽表已无缺失，缺失数要从插补报告里取回来
            missing_by_variable = {
                c["variable"]: c["n_missing"] for c in impute_report["columns"]
            }
            for row in rows:
                row["n_missing"] = missing_by_variable.get(row["variable"], 0)

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

        if impute_report is not None:
            notes.append(
                f"缺失值用链式方程多重插补（{impute_report['m']} 份 × "
                f"{impute_report['iterations']} 轮），各单元格按 Rubin 规则合并。"
                f"分类变量的例数是合并构成比换算回去的估计数，不再是整数观测。"
            )
            notes.append(
                "两组连续变量的 p 值走 Rubin 合并（有均值差和方差，更准）；"
                "分类变量与多组比较只有检验统计量，退到 D2 规则。"
                "秩检验与 Fisher 精确检验没有公认的合并方式，插补时不使用。"
            )
            notes.append(
                "基线表的通行做法是报观测值并单列缺失数 —— 读者想知道实际测到了什么。"
                "这里选了插补，好处是分母与模型口径一致，代价是表里的数字不再全是"
                "直接观测到的。两种都对，别在同一篇里混用。"
            )
            heavy = [c for c in impute_report["columns"] if c["pct"] > 40]
            if heavy:
                names = "、".join(ctx.catalog[c["variable"]].label for c in heavy)
                analysis_warnings.append(
                    f"这些变量缺失超过 40%：{names}。表里相应的数字有很大一部分"
                    f"是插补模型给的 —— 看 FMI 列。"
                )

        if weighted:
            notes.append(
                "已按抽样权重给出人群估计。分类变量括号内是加权后的人群构成比，"
                "括号外的例数仍是实际观测数；连续变量一律报均值 ± 标准差 —— "
                "检验比的是各组均值，显示中位数会让读者对不上。"
            )
            if design is not None and not design.approximate:
                notes.append(
                    f"p 值来自设计校正的 Wald 检验：统计量经 Taylor 线性化后，"
                    f"按层内初级抽样单元的离散度估计方差。"
                    f"{design.n_strata} 层、{design.n_psu} 个 PSU，"
                    f"设计自由度 {design.df}。"
                )
                singletons = design.singleton_strata()
                if singletons:
                    analysis_warnings.append(
                        f"有 {singletons} 个层只剩一个初级抽样单元，"
                        f"它们对方差贡献不了信息，标准误会偏小。队列可能切得太碎了。"
                    )
            else:
                analysis_warnings.append(
                    "该数据集只有抽样权重、没有分层与初级抽样单元，"
                    "标准误按有放回抽样近似。聚类会把真实方差抬高，这里会低估。"
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
                "imputation": impute_report,
                "effective_n": (
                    round(W.effective_n(use_weights), 1) if weighted else None
                ),
                "design": (
                    {
                        "n_strata": design.n_strata,
                        "n_psu": design.n_psu,
                        "df": design.df,
                        "approximate": design.approximate,
                    }
                    if design is not None else None
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
        variables: list[str] = Field(
            ..., min_length=1, max_length=MAX_VARIABLES_PER_ANALYSIS,
            json_schema_extra=widgets.variables(
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
        variables: list[str] = Field(
            ..., min_length=1, max_length=MAX_VARIABLES_PER_ANALYSIS,
            json_schema_extra=widgets.variables(
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
