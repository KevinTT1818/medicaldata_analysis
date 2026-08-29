"""回归算子：logistic 回归（OR + 森林图 + ROC）。"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel, Field

from .base import Analysis, AnalysisContext, AnalysisResult
from .design import DesignError, build_design
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

        ctx.progress("拟合 logistic 模型", 0.5)
        try:
            fit = sm.Logit(y_used, X).fit(disp=0)
        except Exception as exc:  # noqa: BLE001
            raise DesignError(
                f"模型未收敛（{exc}）。常见原因是某个自变量把结局完全分开，"
                f"或自变量之间高度共线。"
            ) from exc

        alpha = 1 - p.conf_level
        ci = fit.conf_int(alpha=alpha)
        rows = [
            {
                "label": t.label,
                "variable": t.variable,
                "reference": t.reference,
                # 第 0 列是截距，协变量从 1 开始
                "estimate": float(np.exp(fit.params[i + 1])),
                "ci_lower": float(np.exp(ci[i + 1, 0])),
                "ci_upper": float(np.exp(ci[i + 1, 1])),
                "p": float(fit.pvalues[i + 1]),
            }
            for i, t in enumerate(terms)
        ]

        ctx.progress("计算 ROC", 0.85)
        roc = _roc(y_used, np.asarray(fit.predict(X), dtype="float64"))

        notes = [
            f"共 {int(usable.sum())} 例进入模型，{int(y_used.sum())} 例为阳性结局。",
            "分类变量用哑变量编码，参照组在行标签中标出。",
        ]
        if n_dropped:
            notes.append(f"因缺失剔除 {n_dropped} 例（complete-case）。")
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
                "pseudo_r2": float(fit.prsquared),
                "n_used": int(usable.sum()),
                "n_events": int(y_used.sum()),
                "n_dropped": n_dropped,
                "notes": notes,
            },
            warnings=W.warn_if_survey_data(ctx.frame),
        )
