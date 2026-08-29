"""回归算子：logistic 回归（OR + 森林图 + ROC）。"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats
from pydantic import BaseModel, Field

from ..cohort.builder import PSU_COLUMN, STRATUM_COLUMN
from .base import Analysis, AnalysisContext, AnalysisResult
from .design import DesignError, build_design
from . import survey
from .registry import register
from . import weights as W
from . import widgets


def _binary_outcome(ctx: AnalysisContext, vid: str) -> np.ndarray:
    """把结局变量转成 0/1，非二分类的直接报错。"""
    v = ctx.catalog[vid]
    s = ctx.frame[vid]

    if v.source in ("condition", "outcome"):
        return s.cast(pl.Float64, strict=False).to_numpy().astype("float64")

    cats = s.cast(pl.Utf8, strict=False)
    levels = cats.drop_nulls().unique().sort().to_list()
    if len(levels) != 2:
        raise DesignError(
            f"「{v.label}」有 {len(levels)} 个水平，logistic 回归要求结局是二分类"
        )
    arr = cats.to_numpy()
    out = np.where(arr == levels[1], 1.0, 0.0)
    out[cats.is_null().to_numpy()] = np.nan
    return out


def _survey_design(ctx: AnalysisContext) -> survey.Design | None:
    """从宽表取出抽样设计。没有权重就返回 None。"""
    weights = W.sample_weights(ctx.frame)
    if not W.has_weights(weights):
        return None
    strata = ctx.frame[STRATUM_COLUMN].to_numpy() if STRATUM_COLUMN in ctx.frame.columns else None
    psus = ctx.frame[PSU_COLUMN].to_numpy() if PSU_COLUMN in ctx.frame.columns else None
    if strata is not None and psus is not None:
        if not (pl.Series(strata).is_not_null().any() and pl.Series(psus).is_not_null().any()):
            strata = psus = None
    return survey.build_design(weights, strata, psus)


def _roc(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """ROC 曲线与 AUC。用 Mann-Whitney U 的等价关系算 AUC，不引入 sklearn。"""
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auc": None, "fpr": [], "tpr": []}

    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    auc = float(np.trapezoid(tpr, fpr))

    # 点数太多会拖慢渲染，等距抽样到 200 点以内
    if tpr.size > 200:
        idx = np.unique(np.linspace(0, tpr.size - 1, 200).astype(int))
        tpr, fpr = tpr[idx], fpr[idx]

    return {
        "auc": auc,
        "fpr": [round(float(x), 5) for x in fpr],
        "tpr": [round(float(x), 5) for x in tpr],
    }


@register
class LogisticRegression(Analysis):
    id = "regression.logistic"
    label = "Logistic 回归"
    description = (
        "二分类结局的多因素分析。输出比值比 OR、95% 置信区间与 p 值，"
        "并给出模型的 ROC 曲线与 AUC。分类变量自动做哑变量编码。"
    )
    result_kind = "forest"

    class Params(BaseModel):
        outcome: str = Field(..., json_schema_extra=widgets.variable(
            "结局", "必须是二分类", widgets.BINARY_OUTCOME))
        covariates: list[str] = Field(..., json_schema_extra=widgets.variables(
            "自变量", "分类变量会自动做哑变量编码", widgets.COVARIATE))
        conf_level: float = Field(0.95, ge=0.5, le=0.999, title="置信水平",
                                  description="OR 置信区间的覆盖概率")

    def required_variables(self, params: BaseModel) -> list[str]:
        out = list(params.covariates)  # type: ignore[attr-defined]
        if params.outcome not in out:  # type: ignore[attr-defined]
            out.append(params.outcome)  # type: ignore[attr-defined]
        return out

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        p: LogisticRegression.Params = ctx.params  # type: ignore[assignment]
        y = _binary_outcome(ctx, p.outcome)

        covariates = [c for c in p.covariates if c != p.outcome]
        design, terms, complete = build_design(ctx.frame, ctx.catalog, covariates)

        usable = complete & np.isfinite(y)
        n_dropped = int((~usable).sum())
        if usable.sum() < len(terms) + 2:
            raise DesignError("完整病例数太少，不足以估计这个模型")

        X = sm.add_constant(design[usable].to_numpy(), has_constant="add")
        y_used = y[usable]

        full_design = _survey_design(ctx)
        weighted = full_design is not None
        alpha = 1 - p.conf_level

        ctx.progress("拟合 logistic 模型", 0.5)
        if weighted:
            # 域估计：保留全部 PSU，只把完整病例之外的权重置零。
            # 直接把不完整的行删掉会让某些层/PSU 消失，方差就不对了。
            masked_weights = np.where(usable, full_design.weight, 0.0)
            sub_design = survey.Design(
                weight=masked_weights[usable],
                stratum=full_design.stratum[usable],
                psu=full_design.psu[usable],
                approximate=full_design.approximate,
            )
            try:
                fit = survey.weighted_logit(y_used, X, sub_design)
            except survey.LogitError as exc:
                raise DesignError(str(exc)) from exc

            se = fit.se
            df = fit.df
            if df <= 0:
                raise DesignError("抽样设计自由度为 0，无法给出置信区间")
            t_crit = float(stats.t.ppf(1 - alpha / 2, df))

            def coefficient(i: int) -> tuple[float, float, float, float]:
                b, s = float(fit.beta[i]), float(se[i])
                p_value = float(2 * stats.t.sf(abs(b / s), df)) if s > 0 else float("nan")
                return (float(np.exp(b)), float(np.exp(b - t_crit * s)),
                        float(np.exp(b + t_crit * s)), p_value)

            predicted = fit.fitted
        else:
            try:
                model = sm.Logit(y_used, X).fit(disp=0)
            except Exception as exc:  # noqa: BLE001
                raise DesignError(
                    f"模型未收敛（{exc}）。常见原因是某个自变量把结局完全分开，"
                    f"或自变量之间高度共线。"
                ) from exc
            ci = model.conf_int(alpha=alpha)

            def coefficient(i: int) -> tuple[float, float, float, float]:
                return (float(np.exp(model.params[i])), float(np.exp(ci[i, 0])),
                        float(np.exp(ci[i, 1])), float(model.pvalues[i]))

            predicted = np.asarray(model.predict(X), dtype="float64")
            fit = model

        rows = []
        for i, t in enumerate(terms):
            # 第 0 列是截距，协变量从 1 开始
            estimate, lower, upper, p_value = coefficient(i + 1)
            rows.append({
                "label": t.label, "variable": t.variable, "reference": t.reference,
                "estimate": estimate, "ci_lower": lower, "ci_upper": upper, "p": p_value,
            })

        ctx.progress("计算 ROC", 0.85)
        roc = (
            survey.weighted_auc(y_used, predicted, sub_design.weight)
            if weighted else _roc(y_used, predicted)
        )

        notes = [
            f"共 {int(usable.sum())} 例进入模型，{int(y_used.sum())} 例为阳性结局。",
            "分类变量用哑变量编码，参照组在行标签中标出。",
        ]
        if n_dropped:
            notes.append(f"因缺失剔除 {n_dropped} 例（complete-case）。")

        analysis_warnings: list[str] = []
        if weighted:
            notes.append(
                "已按抽样权重估计：系数是伪极大似然解，置信区间与 p 值来自"
                "设计校正的三明治方差（得分贡献的方差按分层与初级抽样单元估计）。"
            )
            if sub_design.approximate:
                analysis_warnings.append(
                    "只有抽样权重、没有分层与初级抽样单元，方差按有放回抽样近似，会偏小。"
                )
            else:
                notes.append(
                    f"{sub_design.n_strata} 层、{sub_design.n_psu} 个 PSU，"
                    f"设计自由度 {sub_design.df} —— 置信区间用的是该自由度下的 t 分布。"
                )
                singletons = sub_design.singleton_strata()
                if singletons:
                    analysis_warnings.append(
                        f"有 {singletons} 个层只剩一个初级抽样单元，标准误会偏小。"
                    )
            notes.append("AUC 也已加权，与系数口径一致。")
        notes.append("ROC 与 AUC 是在训练数据上算的，属于表观性能，会高估真实泛化能力。")

        return AnalysisResult(
            kind=self.result_kind,
            payload={
                "kind": self.result_kind,
                "model": "Logistic 回归",
                "effect_label": "OR",
                "null_value": 1.0,
                "conf_level": p.conf_level,
                "rows": rows,
                "ph_test": [],
                "roc": roc,
                "weighted": weighted,
                "design": (
                    {
                        "n_strata": sub_design.n_strata, "n_psu": sub_design.n_psu,
                        "df": sub_design.df, "approximate": sub_design.approximate,
                    }
                    if weighted else None
                ),
                # 伪 R² 在加权拟合下没有对应的标准定义，加权时不给
                "pseudo_r2": None if weighted else float(fit.prsquared),
                "n_used": int(usable.sum()),
                "n_events": int(y_used.sum()),
                "n_dropped": n_dropped,
                "notes": notes,
            },
            warnings=analysis_warnings if weighted else W.warn_if_survey_data(ctx.frame),
        )
