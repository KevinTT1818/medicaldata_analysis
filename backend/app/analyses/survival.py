"""生存分析算子：Kaplan-Meier 曲线、Cox 比例风险模型。"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from pydantic import BaseModel, Field

from ..config import MAX_VARIABLES_PER_ANALYSIS
from ..cdm import values
from . import weights as W
from ..cohort.builder import TIME_SUFFIX, Variable
from .base import Analysis, AnalysisContext, AnalysisResult
from .design import DesignError, build_design
from . import imputation
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
        covariates: list[str] = Field(
            ..., min_length=1, max_length=MAX_VARIABLES_PER_ANALYSIS,
            json_schema_extra=widgets.variables(
            "协变量", "分类变量会自动做哑变量编码", widgets.COVARIATE))
        conf_level: float = Field(0.95, ge=0.5, le=0.999, title="置信水平",
                                  description="HR 置信区间的覆盖概率")
        check_ph: bool = Field(True, title="检验比例风险假设",
                               description="不满足时 HR 不能按恒定风险比解读")
        missing: Literal["complete_case", "multiple_imputation"] = Field(
            "complete_case", title="缺失值处理",
            description="完全病例会丢人且可能引入偏倚；多重插补把插补的不确定性算进区间",
            json_schema_extra={"x-enum-labels": {
                "complete_case": "完全病例（剔除有缺失的行）",
                "multiple_imputation": "多重插补（MICE + Rubin 合并）",
            }})
        n_imputations: int = Field(
            5, ge=2, le=20, title="插补份数",
            description="仅在选多重插补时有效。份数越多合并越稳，代价是更慢")

    def required_variables(self, params: BaseModel) -> list[str]:
        out = list(params.covariates)  # type: ignore[attr-defined]
        if params.outcome not in out:  # type: ignore[attr-defined]
            out.append(params.outcome)  # type: ignore[attr-defined]
        return out

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        p: CoxRegression.Params = ctx.params  # type: ignore[assignment]
        duration, event = _survival_inputs(ctx, p.outcome)
        covariates = [c for c in p.covariates if c != p.outcome]

        # 随访时长与事件标志缺失的行不能插补 —— 那等于把结局编出来
        survival_ok = np.isfinite(duration) & np.isfinite(event) & (duration > 0)

        ctx.progress("准备数据", 0.15)
        try:
            frames, impute_report = imputation.prepare_frames(
                ctx.frame, ctx.catalog, covariates, p.missing, p.n_imputations
            )
        except imputation.ImputationError as exc:
            raise DesignError(str(exc)) from exc

        fits: list[dict[str, Any]] = []
        terms: list[Any] = []
        for index, frame in enumerate(frames):
            ctx.progress(
                f"拟合 Cox 模型 {index + 1}/{len(frames)}"
                if len(frames) > 1 else "拟合 Cox 模型",
                0.3 + 0.45 * index / max(len(frames), 1),
            )
            fits.append(self._fit_once(ctx, frame, duration, event, survival_ok,
                                       covariates, p))
            terms = fits[-1]["terms"]

        primary = fits[0]
        alpha = 1 - p.conf_level

        if len(fits) > 1:
            ctx.progress("按 Rubin 规则合并", 0.85)
            rows = imputation.pool_effect_rows(
                fits, terms, p.conf_level, primary["df"], offset=0
            )
        else:
            rows = [
                {
                    "label": t.label, "variable": t.variable, "reference": t.reference,
                    **primary["coefficient"](i, alpha),
                    "fmi": None,
                }
                for i, t in enumerate(terms)
            ]

        ph_rows, ph_warnings = self._pooled_ph(fits, terms, p.check_ph, ctx)

        n_used = primary["n_used"]
        n_dropped = primary["n_dropped"]
        n_events = primary["n_events"]

        notes = [
            f"共 {n_used} 例进入模型，{n_events} 例发生事件。",
            "分类变量用哑变量编码，参照组在行标签中标出。",
        ]
        analysis_warnings = list(ph_warnings)

        if impute_report is not None:
            notes.append(
                f"协变量缺失用链式方程多重插补（{impute_report['m']} 份 × "
                f"{impute_report['iterations']} 轮），各系数在对数尺度上按 Rubin 规则"
                f"合并 —— HR 的抽样分布在对数尺度才近似正态，直接对比值求平均会有偏。"
            )
            notes.append(
                "随访时长与事件标志不插补：补协变量可以，补结局等于把答案编出来。"
            )
            notes.append(
                "多重插补假定数据「随机缺失」。若某个亚组的该变量是按设计不采集的，"
                "插补只是在外推。"
            )
            heavy = [c for c in impute_report["columns"] if c["pct"] > 40]
            if heavy:
                names = "、".join(ctx.catalog[c["variable"]].label for c in heavy)
                analysis_warnings.append(
                    f"这些变量缺失超过 40%：{names}。结论有很大一部分是插补模型给的 —— "
                    f"看每个系数的 FMI。"
                )
        elif n_dropped:
            notes.append(f"因协变量缺失剔除 {n_dropped} 例（complete-case）。")
            if n_dropped / max(n_used + n_dropped, 1) > 0.1:
                analysis_warnings.append(
                    f"完全病例分析丢掉了 {n_dropped} 例"
                    f"（{n_dropped / (n_used + n_dropped) * 100:.0f}%）。"
                    f"缺失通常不是随机的，这会引入选择偏倚 —— 可改用多重插补。"
                )

        concordance = float(np.mean([f["concordance"] for f in fits]))
        if len(fits) > 1:
            notes.append(f"C-index 取 {len(fits)} 份插补的平均。")

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
                "imputation": impute_report,
                "n_used": n_used,
                "n_events": n_events,
                "n_dropped": n_dropped,
                "concordance": concordance,
                "notes": notes,
            },
            warnings=(
                analysis_warnings if impute_report is not None
                else analysis_warnings + W.warn_if_survey_data(ctx.frame)
            ),
        )

    def _fit_once(self, ctx, frame, duration, event, survival_ok,
                  covariates, params) -> dict[str, Any]:
        """在一份（可能是插补后的）数据上拟合一次 Cox 模型。"""
        matrix, terms, complete = build_design(frame, ctx.catalog, covariates)
        usable = complete & survival_ok
        n_dropped = int((~usable).sum())
        if usable.sum() < len(terms) + 2:
            raise DesignError("完整病例数太少，不足以估计这个模型")

        table = matrix[usable].copy()
        table["__T"] = duration[usable]
        table["__E"] = event[usable]

        fitter = CoxPHFitter(alpha=1 - params.conf_level)
        fitter.fit(table, duration_col="__T", event_col="__E")
        summary = fitter.summary

        beta = np.array([float(summary.loc[t.column, "coef"]) for t in terms])
        se = np.array([float(summary.loc[t.column, "se(coef)"]) for t in terms])

        # 列名带置信水平（如 "exp(coef) lower 95%"），会随 alpha 变，按前缀匹配
        lo_col = next(c for c in summary.columns if c.startswith("exp(coef) lower"))
        hi_col = next(c for c in summary.columns if c.startswith("exp(coef) upper"))

        def coefficient(i: int, _alpha: float) -> dict[str, float]:
            column = terms[i].column
            return {
                "estimate": float(summary.loc[column, "exp(coef)"]),
                "ci_lower": float(summary.loc[column, lo_col]),
                "ci_upper": float(summary.loc[column, hi_col]),
                "p": float(summary.loc[column, "p"]),
            }

        return {
            "beta": beta, "se": se, "terms": terms, "coefficient": coefficient,
            "fitter": fitter, "table": table,
            "df": float(usable.sum() - len(terms)),
            "n_used": int(usable.sum()), "n_dropped": n_dropped,
            "n_events": int(event[usable].sum()),
            "concordance": float(fitter.concordance_index_),
        }

    @staticmethod
    def _pooled_ph(fits, terms, check_ph: bool,
                   ctx) -> tuple[list[dict[str, Any]], list[str]]:
        """比例风险假设检验。

        多份插补时没有公认的 p 值合并方式（Rubin 规则是给参数估计的，
        不是给检验统计量的）。这里在每一份上各检一次，取**最小** p 值报出 ——
        诊断用途上宁可偏严：只要有一份提示违背，就该引起注意。
        """
        if not check_ph:
            return [], []

        ctx.progress("检验比例风险假设", 0.9)
        label_of = {t.column: t.label for t in terms}
        worst: dict[str, float] = {}

        for fit in fits:
            result = proportional_hazard_test(
                fit["fitter"], fit["table"], time_transform="rank"
            )
            for column, row in result.summary.iterrows():
                key = column[0] if isinstance(column, tuple) else column
                label = label_of.get(key, str(key))
                value = float(row["p"])
                worst[label] = min(worst.get(label, 1.0), value)

        multiple = len(fits) > 1
        rows = [{"label": label, "p": value} for label, value in worst.items()]
        warnings = [
            f"「{label}」不满足比例风险假设（"
            + (f"{len(fits)} 份插补中最小 p={value:.3f}" if multiple else f"p={value:.3f}")
            + "），该项的 HR 不宜按恒定风险比解读"
            for label, value in worst.items() if value < 0.05
        ]
        return rows, warnings
