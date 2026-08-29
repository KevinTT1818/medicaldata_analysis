"""队列构建：CDM 长表 -> 分析用宽表。

变量 ID 方案：
    person.<column>        人口学列
    measurement.<code>     测量项（数值或分类）
    condition.<code>       诊断，存在与否 -> 0/1
    outcome.<event_type>   结局事件 -> 0/1
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from .. import store
from ..config import (
    CATEGORICAL_MAX_LEVELS,
    MIN_VARIABLE_COVERAGE,
    MIN_VARIABLE_PATIENTS,
)

if TYPE_CHECKING:
    from .filters import Node

Kind = Literal["continuous", "categorical", "binary"]

#: visit_type 的显示名。适配器写的是英文类型，这里统一翻译。
VISIT_TYPE_LABEL = {
    "inpatient": "住院",
    "icu": "ICU 停留",
    "outpatient": "门诊",
    "emergency": "急诊",
}

PERSON_COLUMNS = {
    "age_at_index": ("年龄", "岁", "continuous"),
    "gender": ("性别", None, "categorical"),
    "race": ("种族", None, "categorical"),
    "ethnicity": ("族裔", None, "categorical"),
    "birth_year": ("出生年", None, "continuous"),
}


@dataclass
class Variable:
    id: str
    label: str
    kind: Kind
    source: str
    unit: str | None = None
    n_available: int = 0
    n_missing: int = 0
    levels: list[str] | None = None
    #: 结局变量是否带随访时长，决定能否跑 KM / Cox
    has_survival: bool = False
    #: 二分类变量的阳性人数。诊断/结局在 presence-only 语义下人人都有值
    #: （n_available = 总人数），光看这个分不出哪些诊断常见。
    n_positive: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _infer_numeric_kind(n_distinct: int, all_integer: bool) -> Kind:
    if all_integer and n_distinct <= CATEGORICAL_MAX_LEVELS:
        return "categorical" if n_distinct > 2 else "binary"
    return "continuous"


#: list_variables 上一次隐藏了多少个低频变量，供 API 报给用户
_hidden_counts: dict[str, int] = {}


def hidden_variable_count(dataset: str) -> int:
    return _hidden_counts.get(dataset, 0)


def list_variables(dataset: str) -> list[Variable]:
    """列出某数据集在 CDM 里可用的全部分析变量。

    覆盖人数太少的编码不列出 —— MIMIC 有几百个只出现在一两个人身上的 ICD 码，
    全列出来选择器没法用，而且这种变量也做不了统计。隐藏数另行报出。
    """
    out: list[Variable] = []
    hidden = 0

    with store.read() as cur:
        total = cur.execute(
            "SELECT count(*) FROM person WHERE dataset = ?", [dataset]
        ).fetchone()[0]
        if not total:
            return []

        # --- person 列 ---
        for col, (label, unit, kind) in PERSON_COLUMNS.items():
            n_ok, n_distinct = cur.execute(
                f"SELECT count({col}), count(DISTINCT {col}) FROM person WHERE dataset = ?",
                [dataset],
            ).fetchone()
            if not n_ok:
                continue
            levels = None
            if kind == "categorical":
                levels = [r[0] for r in cur.execute(
                    f"SELECT DISTINCT {col} FROM person "
                    f"WHERE dataset = ? AND {col} IS NOT NULL ORDER BY 1", [dataset]
                ).fetchall()]
                if len(levels) == 2:
                    kind = "binary"
            out.append(Variable(
                id=f"person.{col}", label=label, kind=kind, source="person",
                unit=unit, n_available=n_ok, n_missing=total - n_ok, levels=levels,
            ))

        # --- measurement ---
        rows = cur.execute(
            """SELECT code,
                      any_value(code_text)                        AS label,
                      any_value(unit)                             AS unit,
                      count(DISTINCT person_id)                   AS n_person,
                      count(value_num)                            AS n_num,
                      count(value_text)                           AS n_text,
                      count(DISTINCT value_num)                   AS d_num,
                      coalesce(bool_and(value_num = floor(value_num)), false) AS all_int
               FROM measurement WHERE dataset = ?
               GROUP BY code ORDER BY code""",
            [dataset],
        ).fetchall()

        # 覆盖人数低于这条线的编码不列出（MIMIC 有几百个只出现一两次的 ICD 码）
        min_patients = max(MIN_VARIABLE_PATIENTS, int(total * MIN_VARIABLE_COVERAGE))

        for code, label, unit, n_person, n_num, n_text, d_num, all_int in rows:
            if n_person < min_patients:
                hidden += 1
                continue
            if n_text and not n_num:
                levels = [r[0] for r in cur.execute(
                    "SELECT DISTINCT value_text FROM measurement "
                    "WHERE dataset = ? AND code = ? AND value_text IS NOT NULL ORDER BY 1",
                    [dataset, code],
                ).fetchall()]
                kind: Kind = "binary" if len(levels) == 2 else "categorical"
            else:
                levels = None
                kind = _infer_numeric_kind(d_num, bool(all_int))
                if kind in ("categorical", "binary"):
                    levels = [str(r[0]) for r in cur.execute(
                        "SELECT DISTINCT value_num FROM measurement "
                        "WHERE dataset = ? AND code = ? AND value_num IS NOT NULL ORDER BY 1",
                        [dataset, code],
                    ).fetchall()]

            out.append(Variable(
                id=f"measurement.{code}", label=label or code, kind=kind,
                source="measurement", unit=unit, n_available=n_person,
                n_missing=total - n_person, levels=levels,
            ))

        # --- condition ---
        for code, label, n_person in cur.execute(
            """SELECT code, any_value(code_text), count(DISTINCT person_id)
               FROM condition WHERE dataset = ? GROUP BY code ORDER BY code""",
            [dataset],
        ).fetchall():
            if n_person < min_patients:
                hidden += 1
                continue
            out.append(Variable(
                id=f"condition.{code}", label=label or code, kind="binary",
                source="condition", n_available=total, n_missing=0,
                levels=["否", "是"], n_positive=n_person,
            ))

        # --- visit：派生成人级变量 ---
        #
        # 分析单位始终是人（所有算子都假设一人一行），所以不把就诊变成分析单位，
        # 而是把就诊事实派生成人级变量：住过几次、住了多久、末次去向。
        # 这样「ICU 停留超过 3 天」「再入院」就能直接进条件树。
        visit_types = [r[0] for r in cur.execute(
            """SELECT DISTINCT visit_type FROM visit
               WHERE dataset = ? AND visit_type IS NOT NULL ORDER BY 1""",
            [dataset],
        ).fetchall()]

        for visit_type in visit_types:
            label = VISIT_TYPE_LABEL.get(visit_type, visit_type)
            n_person, n_los = cur.execute(
                """SELECT count(DISTINCT person_id),
                          count(DISTINCT person_id) FILTER (
                            WHERE start_ts IS NOT NULL AND end_ts IS NOT NULL)
                   FROM visit WHERE dataset = ? AND visit_type = ?""",
                [dataset, visit_type],
            ).fetchone()

            # 次数：没有该类就诊的人是 0 次，不是「未知」
            out.append(Variable(
                id=f"visit.count:{visit_type}", label=f"{label}次数", kind="continuous",
                source="visit", unit="次", n_available=total, n_missing=0,
            ))
            if n_los:
                # 时长：没住过的人是「未定义」而不是 0 天 —— 把没住过的人当 0 天
                # 混进均值，回答的就不是「住院的人住了多久」这个问题了
                for metric, suffix in (("los_total", "总时长"), ("los_max", "单次最长")):
                    out.append(Variable(
                        id=f"visit.{metric}:{visit_type}",
                        label=f"{label}{suffix}", kind="continuous",
                        source="visit", unit="天",
                        n_available=n_los, n_missing=total - n_los,
                    ))

            # 结束状态按就诊类型分开。住院的是出院去向（HOME、REHAB…），
            # ICU 停留的是末次监护单元（MICU…），两者不是一回事，
            # 混进同一个变量会让人以为「MICU」是一种出院去向。
            statuses = [r[0] for r in cur.execute(
                """SELECT DISTINCT discharge_status FROM visit
                   WHERE dataset = ? AND visit_type = ? AND discharge_status IS NOT NULL
                   ORDER BY 1""",
                [dataset, visit_type],
            ).fetchall()]
            if 2 <= len(statuses) <= CATEGORICAL_MAX_LEVELS * 3:
                n_status = cur.execute(
                    """SELECT count(DISTINCT person_id) FROM visit
                       WHERE dataset = ? AND visit_type = ? AND discharge_status IS NOT NULL""",
                    [dataset, visit_type],
                ).fetchone()[0]
                out.append(Variable(
                    id=f"visit.discharge_status:{visit_type}",
                    label=f"{label}结束状态", kind="categorical", source="visit",
                    n_available=n_status, n_missing=total - n_status,
                    levels=statuses,
                ))

        # --- outcome ---
        for event_type, n_person, n_fu, n_event in cur.execute(
            """SELECT event_type, count(DISTINCT person_id), count(followup_days),
                      count(*) FILTER (WHERE is_event)
               FROM outcome WHERE dataset = ? GROUP BY event_type ORDER BY 1""",
            [dataset],
        ).fetchall():
            out.append(Variable(
                id=f"outcome.{event_type}", label=f"结局：{event_type}", kind="binary",
                source="outcome", n_available=n_person,
                n_missing=total - n_person, levels=["否", "是"],
                has_survival=bool(n_fu), n_positive=n_event,
            ))

    _hidden_counts[dataset] = hidden
    return out


#: 结局变量的随访时长列名后缀。生存分析靠这个拿到 duration。
TIME_SUFFIX = "#time"

#: 一人一项有多个值时（住院时序数据的常态）如何取一个代表值。
#: 首次是基线值、末次是出院时的值、极值反映最严重状态 —— 三者会给出不同结论，
#: 所以必须由调用方显式指定，不能藏在实现里。默认取首次（基线，Table 1 的惯例）。
MeasurementAgg = Literal["first", "last", "mean", "min", "max"]
DEFAULT_AGG: MeasurementAgg = "first"

#: 抽样设计在宽表里的保留列名。它们是设计元数据不是分析变量，
#: 不出现在变量目录里，但每张宽表都带着 —— 加权点估计只需要权重，
#: 但要算标准误就必须同时有分层与初级抽样单元。
WEIGHT_COLUMN = "__weight"
STRATUM_COLUMN = "__stratum"
PSU_COLUMN = "__psu"


def _select_expr(
    var_id: str, idx: int, agg: MeasurementAgg = DEFAULT_AGG
) -> tuple[list[str], str | None, Any | None]:
    """返回 (SELECT 片段列表, JOIN 片段, 要绑定的参数值)。

    参数值由本函数一并给出 —— 由调用方去猜「这个变量要绑什么」，
    就会出现「JOIN 里没用到 $c0 却仍然绑了它」这类参数数量对不上的错。
    不需要绑定时返回 None。结局变量会多出一列随访时长。
    """
    source, _, key = var_id.partition(".")
    alias = f'"{var_id}"'

    if source == "person":
        return [f"p.{key} AS {alias}"], None, None

    j = f"j{idx}"
    if source == "measurement":
        # 数值按指定策略聚合；文本值没有大小可言，一律取时间上的首个
        numeric_expr = {
            "first": "first(value_num ORDER BY ts NULLS FIRST)",
            "last": "last(value_num ORDER BY ts NULLS FIRST)",
            "mean": "avg(value_num)",
            "min": "min(value_num)",
            "max": "max(value_num)",
        }[agg]
        join = (
            f"LEFT JOIN (SELECT person_id,"
            f" {numeric_expr} AS vnum,"
            f" first(value_text ORDER BY ts NULLS FIRST) AS vtext"
            f" FROM measurement WHERE dataset = $ds AND code = $c{idx}"
            f" GROUP BY person_id) {j} ON {j}.person_id = p.person_id"
        )
        return [f"coalesce({j}.vtext, CAST({j}.vnum AS VARCHAR)) AS {alias}"], join, key

    if source == "condition":
        join = (
            f"LEFT JOIN (SELECT DISTINCT person_id FROM condition"
            f" WHERE dataset = $ds AND code = $c{idx}) {j}"
            f" ON {j}.person_id = p.person_id"
        )
        return [f"CAST({j}.person_id IS NOT NULL AS INTEGER) AS {alias}"], join, key

    if source == "visit":
        metric, _, visit_type = key.partition(":")

        if metric == "discharge_status":
            join = (
                f"LEFT JOIN (SELECT person_id,"
                f" last(discharge_status ORDER BY end_ts NULLS FIRST) AS status"
                f" FROM visit WHERE dataset = $ds AND visit_type = $c{idx}"
                f" AND discharge_status IS NOT NULL"
                f" GROUP BY person_id) {j} ON {j}.person_id = p.person_id"
            )
            return [f"{j}.status AS {alias}"], join, visit_type

        # 时长按秒算再折成天，避免整除丢掉不足一天的部分
        los = "date_diff('second', start_ts, end_ts) / 86400.0"
        aggregate = {
            "count": "count(*)",
            "los_total": f"sum({los})",
            "los_max": f"max({los})",
        }[metric]
        join = (
            f"LEFT JOIN (SELECT person_id, {aggregate} AS v"
            f" FROM visit WHERE dataset = $ds AND visit_type = $c{idx}"
            f" GROUP BY person_id) {j} ON {j}.person_id = p.person_id"
        )
        # 次数缺省补 0（没住过就是 0 次）；时长缺省保持 NULL（没住过谈不上住了几天）
        expr = f"coalesce({j}.v, 0)" if metric == "count" else f"{j}.v"
        return [f"CAST({expr} AS VARCHAR) AS {alias}"], join, visit_type

    if source == "outcome":
        join = (
            f"LEFT JOIN (SELECT person_id, bool_or(is_event) AS ev,"
            f" max(followup_days) AS fu FROM outcome"
            f" WHERE dataset = $ds AND event_type = $c{idx}"
            f" GROUP BY person_id) {j} ON {j}.person_id = p.person_id"
        )
        return (
            [f"CAST({j}.ev AS INTEGER) AS {alias}",
             f'{j}.fu AS "{var_id}{TIME_SUFFIX}"'],
            join,
            key,
        )

    raise ValueError(f"未知变量来源：{source}")


def person_count(dataset: str) -> int:
    """数据集的全量人数。下推筛选后拿不到原始行数，单独查一次。"""
    with store.read() as cur:
        row = cur.execute(
            "SELECT count(*) FROM person WHERE dataset = ?", [dataset]
        ).fetchone()
    return int(row[0]) if row else 0


def has_survey_design(dataset: str) -> bool:
    """是否带完整的抽样设计（权重 + 分层 + 初级抽样单元）。

    只有权重没有分层/PSU 时点估计仍然正确，但标准误只能按简单随机抽样近似，
    会低估真实的抽样误差 —— 聚类会把方差抬上去。
    """
    with store.read() as cur:
        row = cur.execute(
            """SELECT count(sample_weight), count(stratum), count(psu)
               FROM person WHERE dataset = ?""",
            [dataset],
        ).fetchone()
    return bool(row and row[0] and row[1] and row[2])


def has_repeated_measures(dataset: str) -> bool:
    """数据集里是否存在「一个人同一个测量项有多个值」。

    有的话聚合策略就会影响结论，前端才需要把这个选项显示出来；
    横断面数据一人一值，取哪个都一样，不该拿这个选项去打扰用户。
    """
    with store.read() as cur:
        row = cur.execute(
            """SELECT count(*) FROM (
                 SELECT 1 FROM measurement WHERE dataset = ?
                 GROUP BY person_id, code HAVING count(*) > 1 LIMIT 1)""",
            [dataset],
        ).fetchone()
    return bool(row and row[0])


def has_sample_weights(dataset: str) -> bool:
    """该数据集是否带复杂抽样设计的权重。"""
    with store.read() as cur:
        row = cur.execute(
            "SELECT count(sample_weight) FROM person WHERE dataset = ?", [dataset]
        ).fetchone()
    return bool(row and row[0])


def build_feature_frame(
    dataset: str,
    variables: list[str],
    where: "Node | None" = None,
    agg: MeasurementAgg = DEFAULT_AGG,
) -> pl.DataFrame:
    """把 CDM 反向透视成一人一行的宽表。

    测量值统一取字符串形态（数值列在算子里按需转回 float），
    这样分类型和数值型测量能走同一条路径。

    传了 where 就把队列条件下推到 SQL，在物化到 Python 之前先筛掉不要的人。

    agg 决定一人一项有多个值时取哪一个。横断面数据集只有一个值，取哪个都一样；
    住院时序数据（MIMIC）平均每人每项 43 个值，这个选择会实实在在改变结论。
    """
    selects = [
        "p.person_id",
        f'p.sample_weight AS "{WEIGHT_COLUMN}"',
        f'p.stratum AS "{STRATUM_COLUMN}"',
        f'p.psu AS "{PSU_COLUMN}"',
    ]
    joins: list[str] = []
    params: dict[str, Any] = {"ds": dataset}

    for i, vid in enumerate(variables):
        sel, join, bound = _select_expr(vid, i, agg)
        selects.extend(sel)
        if join:
            joins.append(join)
        if bound is not None:
            params[f"c{i}"] = bound

    inner = (
        f"SELECT {', '.join(selects)} FROM person p "
        + " ".join(joins)
        + " WHERE p.dataset = $ds"
    )

    sql = f"{inner} ORDER BY person_id"
    if where is not None:
        from . import filters  # 延迟导入：filters 依赖本模块的 Variable

        clause = filters.to_sql(where, {v.id: v for v in list_variables(dataset)}, params)
        if clause:
            sql = f"SELECT * FROM ({inner}) t WHERE {clause} ORDER BY person_id"

    with store.read() as cur:
        return cur.execute(sql, params).pl()
