"""生存分析算子：Kaplan-Meier 曲线、Cox 比例风险模型。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from pydantic import BaseModel, Field

from ..cdm import values
from . import weights as W
from ..cohort.builder import TIME_SUFFIX, Variable
from .base import Analysis, AnalysisContext, AnalysisResult
from .design import DesignError, build_design
from .registry import register
from . import widgets


def _survival_inputs(ctx: AnalysisContext, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    """从宽表取出 (随访时长, 是否发生事件)。"""
    time_col = f"{outcome}{TIME_SUFFIX}"
    if time_col not in ctx.frame.columns:
        raise ValueError(f"结局「{outcome}」没有随访时长，无法做生存分析")

    duration = ctx.frame[time_col].cast(pl.Float64, strict=False).to_numpy().astype("float64")
    event = ctx.frame[outcome].cast(pl.Float64, strict=False).to_numpy().astype("float64")

    # 宽表总会有这一列（可能全是 NULL）。全空说明该数据集只记了结局有无、
    # 没记随访时长 —— 这时必须报错，否则会安静地返回一条空曲线。
    if not np.isfinite(duration).any():
        label = ctx.catalog[outcome].label if outcome in ctx.catalog else outcome
        raise ValueError(
            f"「{label}」没有随访时长，做不了生存分析。"
            f"该数据集只记录了结局有无，横断面数据只能用 logistic 回归。"
        )
    return duration, event


def _group_series(ctx: AnalysisContext, group_by: str) -> tuple[np.ndarray, list[str]]:
    v = ctx.catalog[group_by]
    s = values.as_category(ctx.frame, group_by, v)
    return s.to_numpy(), values.levels_of(s, v)


@register
class KaplanMeier(Analysis):
    id = "survival.kaplan_meier"
    label = "Kaplan-Meier 生存曲线"
    description = (
        "非参数生存曲线，带 95% 置信区间、删失标记与 at-risk 人数表。"
        "分组时附 log-rank 检验。曲线是阶梯函数，不是平滑曲线。"
    )
    result_kind = "km_curve"

    class Params(BaseModel):
        outcome: str = Field(..., json_schema_extra=widgets.variable(
            "结局", "必须带随访时长", widgets.SURVIVAL_OUTCOME))
        group_by: str | None = Field(None, json_schema_extra=widgets.variable(
            "分组变量", "留空则出单条总体曲线", widgets.GROUPING))
        conf_level: float = Field(0.95, ge=0.5, le=0.999, title="置信水平",
                                  description="置信带的覆盖概率")

    def required_variables(self, params: BaseModel) -> list[str]:
        out = [params.outcome]  # type: ignore[attr-defined]
        gb = params.group_by    # type: ignore[attr-defined]
        if gb and gb not in out:
            out.append(gb)
        return out

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        p: KaplanMeier.Params = ctx.params  # type: ignore[assignment]
        duration, event = _survival_inputs(ctx, p.outcome)
        valid = np.isfinite(duration) & np.isfinite(event)

        if p.group_by:
            garr, levels = _group_series(ctx, p.group_by)
            groups = [(lv, valid & (garr == lv)) for lv in levels]
        else:
            groups = [("总体", valid)]

        alpha = 1 - p.conf_level
        series: list[dict[str, Any]] = []
        warnings: list[str] = []

        for name, mask in groups:
            d, e = duration[mask], event[mask]
            if d.size == 0:
                warnings.append(f"分组「{name}」没有可用记录，已跳过")
                continue

            fitter = KaplanMeierFitter(alpha=alpha)
            fitter.fit(d, e, label=name)

            sf = fitter.survival_function_[name]
            ci = fitter.confidence_interval_
            times = [float(t) for t in sf.index]

            median = float(fitter.median_survival_time_)
            censored_t = sorted(float(x) for x in d[e == 0])

            series.append({
                "name": name,
                "n": int(d.size),
                "events": int(e.sum()),
                "censored": int((e == 0).sum()),
                # 中位生存时间可能因为未过半而是 inf，前端要能显示「未达到」
                "median_survival": None if not np.isfinite(median) else median,
                "t": times,
                "survival": [float(x) for x in sf.to_numpy()],
                "ci_lower": [float(x) for x in ci.iloc[:, 0].to_numpy()],
                "ci_upper": [float(x) for x in ci.iloc[:, 1].to_numpy()],
                "censor_t": censored_t,
                "censor_s": [float(fitter.predict(t)) for t in censored_t],
            })

        # at-risk 表：KM 图的标准配套，读者据此判断曲线尾部是否可信
        max_t = float(np.nanmax(duration[valid])) if valid.any() else 0.0
        checkpoints = [round(max_t * i / 5, 1) for i in range(6)]
        at_risk = {
            "times": checkpoints,
            "rows": [
                {
                    "name": name,
                    "counts": [int((duration[mask] >= t).sum()) for t in checkpoints],
                }
                for name, mask in groups
            ],
        }

        logrank_p: float | None = None
        test: str | None = None
        if p.group_by and len(series) >= 2:
            gvals = _group_series(ctx, p.group_by)[0][valid]
            result = multivariate_logrank_test(duration[valid], gvals, event[valid])
            logrank_p = float(result.p_value)
            test = "Log-rank 检验"

        return AnalysisResult(
            kind=self.result_kind,
            payload={
                "kind": self.result_kind,
                "time_unit": "天",
                "conf_level": p.conf_level,
                "group_by_label": ctx.catalog[p.group_by].label if p.group_by else None,
                "series": series,
                "at_risk": at_risk,
                "logrank_p": logrank_p,
                "test": test,
            },
            warnings=warnings + W.warn_if_survey_data(ctx.frame),
        )


@register
class CoxRegression(Analysis):
    id = "survival.cox"
    label = "Cox 比例风险模型"
    description = (
        "多因素生存分析。输出各协变量的风险比 HR、95% 置信区间与 p 值，"
        "并检验比例风险假设是否成立。分类变量自动做哑变量编码。"
    )
    result_kind = "forest"

    class Params(BaseModel):
        outcome: str = Field(..., json_schema_extra=widgets.variable(
            "结局", "必须带随访时长", widgets.SURVIVAL_OUTCOME))
        covariates: list[str] = Field(..., json_schema_extra=widgets.variables(
            "协变量", "分类变量会自动做哑变量编码", widgets.COVARIATE))
        conf_level: float = Field(0.95, ge=0.5, le=0.999, title="置信水平",
                                  description="HR 置信区间的覆盖概率")
        check_ph: bool = Field(True, title="检验比例风险假设",
                               description="不满足时 HR 不能按恒定风险比解读")

    def required_variables(self, params: BaseModel) -> list[str]:
        out = list(params.covariates)  # type: ignore[attr-defined]
        if params.outcome not in out:  # type: ignore[attr-defined]
            out.append(params.outcome)  # type: ignore[attr-defined]
        return out

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        p: CoxRegression.Params = ctx.params  # type: ignore[assignment]
        duration, event = _survival_inputs(ctx, p.outcome)

        covariates = [c for c in p.covariates if c != p.outcome]
        design, terms, complete = build_design(ctx.frame, ctx.catalog, covariates)

        usable = complete & np.isfinite(duration) & np.isfinite(event) & (duration > 0)
        n_dropped = int((~usable).sum())
        if usable.sum() < len(terms) + 2:
            raise DesignError("完整病例数太少，不足以估计这个模型")

        df = design[usable].copy()
        df["__T"] = duration[usable]
        df["__E"] = event[usable]

        ctx.progress("拟合 Cox 模型", 0.5)
        fitter = CoxPHFitter(alpha=1 - p.conf_level)
        fitter.fit(df, duration_col="__T", event_col="__E")
        summary = fitter.summary

        # 列名带置信水平（如 "exp(coef) lower 95%"），会随 alpha 变，按前缀匹配
        lo_col = next(c for c in summary.columns if c.startswith("exp(coef) lower"))
        hi_col = next(c for c in summary.columns if c.startswith("exp(coef) upper"))

        rows = [
            {
                "label": t.label,
                "variable": t.variable,
                "reference": t.reference,
                "estimate": float(summary.loc[t.column, "exp(coef)"]),
                "ci_lower": float(summary.loc[t.column, lo_col]),
                "ci_upper": float(summary.loc[t.column, hi_col]),
                "p": float(summary.loc[t.column, "p"]),
            }
            for t in terms
        ]

        ph_warnings: list[str] = []
        ph_rows: list[dict[str, Any]] = []
        if p.check_ph:
            ctx.progress("检验比例风险假设", 0.8)
            ph = proportional_hazard_test(fitter, df, time_transform="rank")
            label_of = {t.column: t.label for t in terms}
            for column, row in ph.summary.iterrows():
                key = column[0] if isinstance(column, tuple) else column
                pv = float(row["p"])
                ph_rows.append({"label": label_of.get(key, str(key)), "p": pv})
                if pv < 0.05:
                    ph_warnings.append(
                        f"「{label_of.get(key, key)}」不满足比例风险假设（p={pv:.3f}），"
                        f"该项的 HR 不宜按恒定风险比解读"
                    )

        notes = [
            f"共 {int(usable.sum())} 例进入模型，{int(event[usable].sum())} 例发生事件。",
            "分类变量用哑变量编码，参照组在行标签中标出。",
        ]
        if n_dropped:
            notes.append(
                f"因协变量缺失剔除 {n_dropped} 例（complete-case）。"
                f"缺失比例较高时应考虑多重插补。"
            )

        return AnalysisResult(
            kind=self.result_kind,
            payload={
                "kind": self.result_kind,
                "model": "Cox 比例风险模型",
                "effect_label": "HR",
                "null_value": 1.0,
                "conf_level": p.conf_level,
                "rows": rows,
                "ph_test": ph_rows,
                "n_used": int(usable.sum()),
                "n_events": int(event[usable].sum()),
                "n_dropped": n_dropped,
                "concordance": float(fitter.concordance_index_),
                "notes": notes,
            },
            warnings=ph_warnings + W.warn_if_survey_data(ctx.frame),
        )
