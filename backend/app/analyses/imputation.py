"""多重插补（链式方程 MICE）与 Rubin 合并规则。

完全病例分析是设计文档里标为「结论会错」的坑：NHANES 上五个变量就能丢掉
34% 的样本，而且丢的方式不是随机的 —— 做没做某项检查本身就和人群特征相关，
删掉他们会引入选择偏倚，剩下的样本不再代表原来的人群。

多重插补的思路是：不猜「那个值到底是多少」，而是承认它有不确定性 ——
生成 m 份各不相同的补全数据，在每一份上做同样的分析，
再用 Rubin 规则把 m 个结果合并，把「插补带来的额外不确定性」算进标准误里。
插补越不确定，合并后的区间就越宽，这正是它比单次填补诚实的地方。

连续变量用预测均值匹配（PMM）：不用回归预测值本身，而是从预测值最接近的
若干个**实际观测**里随机抽一个。这样插补出来的一定是真实出现过的值 ——
不会补出负数年龄，也不会把偏态分布补成正态。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ..cdm import values as cdm_values
from ..cohort.builder import WEIGHT_COLUMN, Variable

#: PMM 的候选供体数。太小会重复用同几个值，太大就退化成回归填补。
PMM_DONORS = 5


class ImputationError(ValueError):
    pass


@dataclass
class Column:
    variable_id: str
    kind: str                      # continuous | categorical
    values: np.ndarray             # 连续为浮点，分类为整数编码；缺失为 nan
    levels: list[str] | None
    missing: np.ndarray            # 布尔掩码

    @property
    def n_missing(self) -> int:
        return int(self.missing.sum())


def _extract(frame: pl.DataFrame, catalog: dict[str, Variable],
             variable_ids: list[str]) -> list[Column]:
    columns: list[Column] = []
    for vid in variable_ids:
        variable = catalog[vid]
        if variable.kind == "continuous":
            raw = cdm_values.as_numeric(frame, vid)
            missing = ~np.isfinite(raw)
            columns.append(Column(vid, "continuous", raw.astype("float64"), None, missing))
        else:
            series = cdm_values.as_category(frame, vid, variable)
            levels = cdm_values.levels_of(series, variable)
            arr = series.to_numpy()
            codes = np.full(arr.size, np.nan)
            for index, level in enumerate(levels):
                codes[arr == level] = index
            columns.append(
                Column(vid, "categorical", codes, levels, ~np.isfinite(codes))
            )
    return columns


def _predictor_matrix(columns: list[Column], exclude: int,
                      extra: np.ndarray | None) -> np.ndarray:
    """用除 exclude 外的所有列拼预测矩阵。分类列做哑变量，去掉一个水平避免共线。"""
    blocks = [np.ones((columns[0].values.size, 1))]
    for index, column in enumerate(columns):
        if index == exclude:
            continue
        if column.kind == "continuous":
            blocks.append(column.values[:, None])
        else:
            n_levels = len(column.levels or [])
            for level in range(1, n_levels):      # 第 0 个水平作参照
                blocks.append((column.values == level).astype("float64")[:, None])
    if extra is not None:
        blocks.append(extra[:, None])
    return np.hstack(blocks)


def _ols_predict(X: np.ndarray, y: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """在 fit_rows 上拟合最小二乘，对全部行给出预测值。"""
    A, b = X[fit_rows], y[fit_rows]
    coefficients, *_ = np.linalg.lstsq(A, b, rcond=None)
    return X @ coefficients


def _pmm(predicted: np.ndarray, observed_values: np.ndarray,
         observed_rows: np.ndarray, target_rows: np.ndarray,
         rng: np.random.Generator) -> np.ndarray:
    """预测均值匹配：从预测值最接近的若干个实际观测里随机抽一个。"""
    donor_predictions = predicted[observed_rows]
    donor_values = observed_values[observed_rows]
    out = np.empty(int(target_rows.sum()))

    for position, prediction in enumerate(predicted[target_rows]):
        distances = np.abs(donor_predictions - prediction)
        k = min(PMM_DONORS, distances.size)
        nearest = np.argpartition(distances, k - 1)[:k]
        out[position] = donor_values[rng.choice(nearest)]
    return out


def _draw_category(X: np.ndarray, codes: np.ndarray, fit_rows: np.ndarray,
                   target_rows: np.ndarray, n_levels: int,
                   rng: np.random.Generator) -> np.ndarray:
    """对每个水平拟合一对多 logistic，归一化后按概率抽取。"""
    scores = np.zeros((int(target_rows.sum()), n_levels))
    for level in range(n_levels):
        y = (codes == level).astype("float64")
        if y[fit_rows].sum() in (0, fit_rows.sum()):
            # 该水平在观测里全有或全无，退化为常数概率
            scores[:, level] = y[fit_rows].mean()
            continue
        predicted = _ols_predict(X, y, fit_rows)      # 线性概率模型，足够作提议分布
        scores[:, level] = np.clip(predicted[target_rows], 1e-6, 1.0)

    scores /= scores.sum(axis=1, keepdims=True)
    return np.array([rng.choice(n_levels, p=row) for row in scores], dtype="float64")


def impute(
    frame: pl.DataFrame,
    catalog: dict[str, Variable],
    variable_ids: list[str],
    m: int = 5,
    iterations: int = 5,
    seed: int = 0,
) -> tuple[list[pl.DataFrame], dict[str, Any]]:
    """生成 m 份补全后的宽表，以及一份插补情况说明。

    抽样权重若存在会作为预测变量进入插补模型 —— 缺失往往和抽样设计相关，
    把权重放进去能让插补更贴近真实分布。
    """
    columns = _extract(frame, catalog, variable_ids)
    incomplete = [i for i, c in enumerate(columns) if c.n_missing]
    if not incomplete:
        return [frame], {"m": 1, "imputed_rows": 0, "columns": [], "skipped": True}

    if any(c.n_missing == c.values.size for c in columns):
        raise ImputationError("有变量全部缺失，无法插补")

    extra = None
    if WEIGHT_COLUMN in frame.columns:
        weights = frame[WEIGHT_COLUMN].cast(pl.Float64, strict=False).to_numpy()
        if np.isfinite(weights).any():
            extra = np.nan_to_num(weights, nan=float(np.nanmedian(weights)))

    # 缺失少的先补，它们的插补值质量更高，能给后面的变量当更好的预测
    order = sorted(incomplete, key=lambda i: columns[i].n_missing)
    any_missing = np.zeros(frame.height, dtype=bool)
    for column in columns:
        any_missing |= column.missing

    imputed_frames: list[pl.DataFrame] = []
    for chain in range(m):
        rng = np.random.default_rng(seed + chain)
        state = [c.values.copy() for c in columns]

        # 初始化：从该列的观测值里随机抽
        for i in incomplete:
            observed = state[i][~columns[i].missing]
            state[i][columns[i].missing] = rng.choice(observed, columns[i].n_missing)

        for _ in range(iterations):
            for i in order:
                target = columns[i]
                working = [
                    Column(c.variable_id, c.kind, state[j], c.levels, c.missing)
                    for j, c in enumerate(columns)
                ]
                X = _predictor_matrix(working, exclude=i, extra=extra)
                fit_rows = ~target.missing

                if target.kind == "continuous":
                    predicted = _ols_predict(X, state[i], fit_rows)
                    state[i][target.missing] = _pmm(
                        predicted, columns[i].values, fit_rows, target.missing, rng
                    )
                else:
                    state[i][target.missing] = _draw_category(
                        X, state[i], fit_rows, target.missing,
                        len(target.levels or []), rng
                    )

        imputed_frames.append(_rebuild(frame, columns, state))

    report = {
        "m": m,
        "iterations": iterations,
        "imputed_rows": int(any_missing.sum()),
        "total_rows": frame.height,
        "columns": [
            {
                "variable": columns[i].variable_id,
                "n_missing": columns[i].n_missing,
                "pct": round(columns[i].n_missing / frame.height * 100, 1),
                "kind": columns[i].kind,
            }
            for i in order
        ],
        "skipped": False,
    }
    return imputed_frames, report


def _rebuild(frame: pl.DataFrame, columns: list[Column],
             state: list[np.ndarray]) -> pl.DataFrame:
    """把补全后的数值写回宽表。列的字符串形态要和原来一致，下游才认得。"""
    out = frame
    for column, filled in zip(columns, state):
        if column.kind == "continuous":
            series = pl.Series(column.variable_id, filled)
        else:
            levels = column.levels or []
            series = pl.Series(
                column.variable_id,
                [levels[int(code)] if np.isfinite(code) else None for code in filled],
                dtype=pl.Utf8,
            )
        out = out.with_columns(series)
    return out


# ---------------------------------------------------------------- Rubin 合并

@dataclass
class Pooled:
    estimate: float
    se: float
    df: float
    ci_low: float
    ci_high: float
    p: float
    #: 缺失信息占比：这个系数有多少信息量是靠插补补出来的
    fmi: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate, "se": self.se, "df": round(self.df, 1),
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p": self.p, "fmi": round(self.fmi, 4),
        }


def pool(
    estimates: list[float],
    variances: list[float],
    conf_level: float = 0.95,
    complete_df: float | None = None,
) -> Pooled:
    """Rubin 合并规则。

    总方差 T = Ū + (1 + 1/m) B：组内方差 Ū 是「每份补全数据自身的不确定性」，
    组间方差 B 是「不同插补给出的答案有多不一致」——后者正是完全病例分析
    和单次填补都算不出来的那部分。

    自由度用 Barnard-Rubin 修正，样本本身自由度有限时不会给出虚高的自由度。
    """
    m = len(estimates)
    if m < 2:
        raise ImputationError("合并至少需要两份插补结果")

    Q = float(np.mean(estimates))
    U = float(np.mean(variances))                       # 组内
    B = float(np.var(estimates, ddof=1))                # 组间
    T = U + (1 + 1 / m) * B

    if T <= 0:
        raise ImputationError("合并后的方差非正，无法给出区间")

    lam = (1 + 1 / m) * B / T                           # 方差中来自插补的比例
    lam = min(max(lam, 1e-10), 1 - 1e-10)
    df_old = (m - 1) / lam**2

    if complete_df is not None and complete_df > 0:
        df_obs = ((complete_df + 1) / (complete_df + 3)) * complete_df * (1 - lam)
        df = df_old * df_obs / (df_old + df_obs)
    else:
        df = df_old

    se = float(np.sqrt(T))
    t_crit = float(stats.t.ppf(0.5 + conf_level / 2, df))
    p = float(2 * stats.t.sf(abs(Q / se), df)) if se > 0 else float("nan")

    return Pooled(
        estimate=Q, se=se, df=df,
        ci_low=Q - t_crit * se, ci_high=Q + t_crit * se,
        p=p, fmi=lam,
    )


def prepare_frames(
    frame: pl.DataFrame,
    catalog: dict[str, Variable],
    variable_ids: list[str],
    mode: str,
    m: int,
) -> tuple[list[pl.DataFrame], dict[str, Any] | None]:
    """按缺失处理方式准备待拟合的数据。

    选完全病例就原样返回一份；选多重插补就返回 m 份，外加一份插补说明。
    本来就没有缺失时退回单份，不白跑 m 次。
    """
    if mode != "multiple_imputation":
        return [frame], None

    frames, report = impute(frame, catalog, variable_ids, m=m)
    if report.get("skipped"):
        return [frame], None
    return frames, report


def pool_effect_rows(
    fits: list[dict[str, Any]],
    terms: list[Any],
    conf_level: float,
    complete_df: float | None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """把 m 次拟合的系数按 Rubin 规则合并成一组效应量。

    合并在**对数尺度**上做 —— HR 与 OR 的抽样分布在对数尺度上才近似正态，
    直接对比值求平均会有偏。合并完再指数回去。

    offset 是系数数组里协变量的起始下标：logistic 有截距所以是 1，Cox 没有所以是 0。
    """
    rows = []
    for i, term in enumerate(terms):
        estimates = [float(f["beta"][i + offset]) for f in fits]
        variances = [float(f["se"][i + offset]) ** 2 for f in fits]
        pooled = pool(estimates, variances, conf_level, complete_df)
        rows.append({
            "label": term.label,
            "variable": term.variable,
            "reference": term.reference,
            "estimate": float(np.exp(pooled.estimate)),
            "ci_lower": float(np.exp(pooled.ci_low)),
            "ci_upper": float(np.exp(pooled.ci_high)),
            "p": pooled.p,
            "fmi": round(pooled.fmi, 4),
        })
    return rows


def pool_test_statistic(
    statistics: list[float], k: int, m: int | None = None
) -> dict[str, Any]:
    """合并 m 份插补上的检验统计量（Li-Meng-Raghunathan-Rubin 的 D2 规则）。

    Rubin 规则是给**参数估计**的：有点估计也有方差，能算组间方差。
    检验统计量没有这些，只能用另一套：D2 把 m 个卡方统计量合成一个 F。

    D2 的精度不如「先合并参数估计再检验」（D1），所以两组均值比较那种能拿到
    差值和方差的情形仍走 Rubin；只有分类变量的卡方、多组比较这类拿不到
    干净参数估计的，才退到 D2。
    """
    values = np.asarray([s for s in statistics if np.isfinite(s) and s >= 0])
    m = m or values.size
    if values.size < 2 or k < 1:
        return {"p": None, "detail": "合并检验至少需要两份有效统计量"}

    mean_stat = float(values.mean())
    # r 衡量插补带来的额外变异，用 sqrt 尺度更稳（卡方是右偏的）
    r = (1 + 1 / m) * float(np.var(np.sqrt(values), ddof=1))
    denominator = 1 + r
    if denominator <= 0:
        return {"p": None, "detail": "合并后的方差非正"}

    d2 = (mean_stat / k - (m - 1) / (m + 1) * r) / denominator
    if not np.isfinite(d2) or d2 < 0:
        # 插补间差异过大时 D2 可能算成负数，此时不给 p 值而不是给个假的
        return {"p": None, "detail": "各份插补的检验结果差异过大，合并统计量不可用"}

    if r <= 0:
        nu = float("inf")
    else:
        nu = k ** (-3 / m) * (m - 1) * (1 + 1 / r) ** 2

    p = float(stats.f.sf(d2, k, nu)) if np.isfinite(nu) else float(stats.chi2.sf(d2 * k, k))
    return {"p": p, "f": round(d2, 4), "df_num": k,
            "df_den": None if not np.isfinite(nu) else round(nu, 1), "detail": None}
