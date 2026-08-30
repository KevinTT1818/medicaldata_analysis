"""回归算子：logistic 回归（OR + 森林图 + ROC）。"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats
from pydantic import BaseModel, Field

from ..cohort.builder import PSU_COLUMN, STRATUM_COLUMN
from .base import Analysis, AnalysisContext, AnalysisResult
from .design import DesignError, build_design
from . import imputation
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
        p: LogisticRegression.Params = ctx.params  # type: ignore[assignment]
        y = _binary_outcome(ctx, p.outcome)
        covariates = [c for c in p.covariates if c != p.outcome]

        full_design = _survey_design(ctx)
        weighted = full_design is not None
        alpha = 1 - p.conf_level
        y_observed = np.isfinite(y)

        # 结局缺失的行不能参与拟合，但它们的协变量对插补仍有信息，
        # 所以插补用全部行，分析只用结局已观测的行。
        frames: list[pl.DataFrame] = [ctx.frame]
        impute_report: dict[str, Any] | None = None
        if p.missing == "multiple_imputation":
            ctx.progress("多重插补", 0.2)
            try:
                frames, impute_report = imputation.impute(
                    ctx.frame, ctx.catalog, covariates, m=p.n_imputations
                )
            except imputation.ImputationError as exc:
                raise DesignError(str(exc)) from exc
            if impute_report.get("skipped"):
                impute_report = None      # 本来就没有缺失，退回单次拟合

        fits: list[dict[str, Any]] = []
        terms: list = []
        for index, frame in enumerate(frames):
            ctx.progress(
                f"拟合模型 {index + 1}/{len(frames)}" if len(frames) > 1 else "拟合模型",
                0.4 + 0.4 * index / max(len(frames), 1),
            )
            fits.append(self._fit_once(ctx, frame, y, y_observed, covariates,
                                       full_design, weighted))
            terms = fits[-1]["terms"]

        primary = fits[0]
        n_used, n_dropped = primary["n_used"], primary["n_dropped"]

        if len(fits) > 1:
            ctx.progress("按 Rubin 规则合并", 0.85)
            rows = self._pool_rows(fits, terms, p.conf_level)
        else:
            rows = [
                {
                    "label": t.label, "variable": t.variable, "reference": t.reference,
                    **primary["coefficient"](i + 1, alpha),
                    "fmi": None,
                }
                for i, t in enumerate(terms)
            ]

        ctx.progress("计算 ROC", 0.92)
        roc = primary["roc"]

        notes = [
            f"共 {n_used} 例进入模型，{int(primary['n_events'])} 例为阳性结局。",
            "分类变量用哑变量编码，参照组在行标签中标出。",
        ]
        analysis_warnings: list[str] = []

        if impute_report is not None:
            notes.append(
                f"缺失值用链式方程多重插补（{impute_report['m']} 份 × "
                f"{impute_report['iterations']} 轮），连续变量走预测均值匹配 —— "
                f"插补出的值一定是数据里真实出现过的。"
                f"各系数按 Rubin 规则合并，插补带来的额外不确定性已计入置信区间。"
            )
            notes.append(
                "多重插补假定数据「随机缺失」：缺不缺可以由已观测的变量解释。"
                "若某个亚组的该变量是**按设计不采集**的（例如 NHANES 不给两岁以下"
                "婴儿计算 BMI），插补只是在外推，数字看着合理但没有依据。"
            )
            heavy = [c for c in impute_report["columns"] if c["pct"] > 40]
            if heavy:
                names = "、".join(
                    ctx.catalog[c["variable"]].label for c in heavy
                )
                analysis_warnings.append(
                    f"这些变量缺失超过 40%：{names}。结论有很大一部分是插补模型给的，"
                    f"不是数据给的 —— 看每个系数的缺失信息占比 FMI。"
                )
        elif n_dropped:
            notes.append(f"因缺失剔除 {n_dropped} 例（complete-case）。")
            if n_dropped / max(n_used + n_dropped, 1) > 0.1:
                analysis_warnings.append(
                    f"完全病例分析丢掉了 {n_dropped} 例"
                    f"（{n_dropped / (n_used + n_dropped) * 100:.0f}%）。"
                    f"缺失通常不是随机的，这会引入选择偏倚 —— 可改用多重插补。"
                )

        if weighted:
            notes.append(
                "已按抽样权重估计：系数是伪极大似然解，置信区间与 p 值来自"
                "设计校正的三明治方差（得分贡献的方差按分层与初级抽样单元估计）。"
            )
            sub = primary["design"]
            if sub.approximate:
                analysis_warnings.append(
                    "只有抽样权重、没有分层与初级抽样单元，方差按有放回抽样近似，会偏小。"
                )
            else:
                notes.append(
                    f"{sub.n_strata} 层、{sub.n_psu} 个 PSU，设计自由度 {sub.df}。"
                )
                if sub.singleton_strata():
                    analysis_warnings.append(
                        f"有 {sub.singleton_strata()} 个层只剩一个初级抽样单元，"
                        f"标准误会偏小。"
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
                        "n_strata": primary["design"].n_strata,
                        "n_psu": primary["design"].n_psu,
                        "df": primary["design"].df,
                        "approximate": primary["design"].approximate,
                    }
                    if weighted else None
                ),
                "imputation": impute_report,
                # 伪 R² 在加权拟合与合并结果下都没有对应的标准定义
                "pseudo_r2": (
                    None if (weighted or impute_report is not None)
                    else primary["pseudo_r2"]
                ),
                "n_used": n_used,
                "n_events": int(primary["n_events"]),
                "n_dropped": n_dropped,
                "notes": notes,
            },
            warnings=(
                analysis_warnings if (weighted or impute_report is not None)
                else analysis_warnings + W.warn_if_survey_data(ctx.frame)
            ),
        )

    def _fit_once(self, ctx, frame, y, y_observed, covariates,
                  full_design, weighted) -> dict[str, Any]:
        """在一份（可能是插补后的）数据上拟合一次，返回系数取用器与配套信息。"""
        matrix, terms, complete = build_design(frame, ctx.catalog, covariates)
        usable = complete & y_observed
        n_dropped = int((~usable).sum())
        if usable.sum() < len(terms) + 2:
            raise DesignError("完整病例数太少，不足以估计这个模型")

        X = sm.add_constant(matrix[usable].to_numpy(), has_constant="add")
        y_used = y[usable]

        if weighted:
            # 域估计：保留全部 PSU，只把分析范围外的权重置零。
            # 直接删行会让某些层/PSU 消失，方差就不对了。
            sub_design = survey.Design(
                weight=full_design.weight[usable],
                stratum=full_design.stratum[usable],
                psu=full_design.psu[usable],
                approximate=full_design.approximate,
            )
            try:
                fit = survey.weighted_logit(y_used, X, sub_design)
            except survey.LogitError as exc:
                raise DesignError(str(exc)) from exc
            if fit.df <= 0:
                raise DesignError("抽样设计自由度为 0，无法给出置信区间")

            beta, se, df = fit.beta, fit.se, float(fit.df)
            predicted = fit.fitted
            roc = survey.weighted_auc(y_used, predicted, sub_design.weight)
            pseudo_r2 = None
        else:
            sub_design = survey.build_design(np.ones(int(usable.sum())), None, None)
            try:
                model = sm.Logit(y_used, X).fit(disp=0)
            except Exception as exc:  # noqa: BLE001
                raise DesignError(
                    f"模型未收敛（{exc}）。常见原因是某个自变量把结局完全分开，"
                    f"或自变量之间高度共线。"
                ) from exc
            beta = np.asarray(model.params, dtype="float64")
            se = np.asarray(model.bse, dtype="float64")
            df = float(usable.sum() - X.shape[1])
            predicted = np.asarray(model.predict(X), dtype="float64")
            roc = _roc(y_used, predicted)
            pseudo_r2 = float(model.prsquared)

        def coefficient(i: int, alpha: float) -> dict[str, float]:
            b, s = float(beta[i]), float(se[i])
            t_crit = float(stats.t.ppf(1 - alpha / 2, df)) if df > 0 else float("nan")
            p_value = float(2 * stats.t.sf(abs(b / s), df)) if s > 0 else float("nan")
            return {
                "estimate": float(np.exp(b)),
                "ci_lower": float(np.exp(b - t_crit * s)),
                "ci_upper": float(np.exp(b + t_crit * s)),
                "p": p_value,
            }

        return {
            "beta": beta, "se": se, "df": df, "terms": terms,
            "coefficient": coefficient, "roc": roc, "design": sub_design,
            "pseudo_r2": pseudo_r2, "n_used": int(usable.sum()),
            "n_dropped": n_dropped, "n_events": float(y_used.sum()),
        }

    @staticmethod
    def _pool_rows(fits: list[dict[str, Any]], terms: list,
                   conf_level: float) -> list[dict[str, Any]]:
        """按 Rubin 规则把 m 次拟合合并成一组系数。"""
        complete_df = fits[0]["df"]
        rows = []
        for i, t in enumerate(terms):
            estimates = [float(f["beta"][i + 1]) for f in fits]
            variances = [float(f["se"][i + 1]) ** 2 for f in fits]
            pooled = imputation.pool(estimates, variances, conf_level, complete_df)
            rows.append({
                "label": t.label, "variable": t.variable, "reference": t.reference,
                "estimate": float(np.exp(pooled.estimate)),
                "ci_lower": float(np.exp(pooled.ci_low)),
                "ci_upper": float(np.exp(pooled.ci_high)),
                "p": pooled.p,
                # 缺失信息占比：这个系数有多少不确定性是插补带来的
                "fmi": round(pooled.fmi, 4),
            })
        return rows
