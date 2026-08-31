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


#: 每张来源表在宽表 SQL 里的连接别名。同一张表的所有变量共用一次扫描。
#:
#: 早先的写法是每个变量各开一个 LEFT JOIN 子查询，代价随变量数爆炸而与行数无关：
#: MIMIC 只有 100 行，5/20/60/189 个变量分别要 7/41/349/11930 毫秒，
#: 同一条查询加 EXPLAIN ANALYZE 会直接吃掉 6.3 GiB 内存 —— 186 个 JOIN 塞进
#: 一条查询，查询计划器就失控了。界面上点一下「全选」就能触发。
#:
#: 改成按来源表分组、一次扫描 + FILTER 分别聚合之后，JOIN 数量固定为最多 4 个，
#: 与选了多少变量无关：同样的 5/20/80/189 个变量是 7/50/179/238 毫秒。
#:
#: 代价是变量少时略慢（20 个变量 36→50ms，NHANES 的 14 个变量 43→54ms）——
#: 分组扫描的固定开销在变量少时摊不开。横断面数据集一人一项只有一个值，
#: 免掉有序聚合能把这部分赚回来，但判断"有没有重复测量"本身就要 5-7ms，
#: 和省下的时间相当，而判断错了会静默给出不同的数值（MIMIC 上已验证）。
#: 拿一个静默出错的风险去换十几毫秒不划算，所以只保留这一条路径。
SOURCE_ALIAS = {
    "measurement": "m",
    "condition": "cd",
    "visit": "v",
    "outcome": "o",
}

#: 一个变量在宽表里占的列。多数变量一列，结局变量多一列随访时长。
_Plan = dict[int, list[str]]


def _measurement_plan(
    entries: list[tuple[int, str, str]], agg: MeasurementAgg, params: dict[str, Any]
) -> tuple[_Plan, str]:
    # 数值按指定策略聚合；文本值没有大小可言，一律取时间上的首个
    numeric_expr = {
        "first": "first(value_num ORDER BY ts NULLS FIRST)",
        "last": "last(value_num ORDER BY ts NULLS FIRST)",
        "mean": "avg(value_num)",
        "min": "min(value_num)",
        "max": "max(value_num)",
    }[agg]
    a = SOURCE_ALIAS["measurement"]
    plan: _Plan = {}
    cols: list[str] = []
    codes: list[str] = []

    for idx, vid, code in entries:
        key = f"c{idx}"
        params[key] = code
        codes.append(f"${key}")
        # 第二阶段只是把每 (人, 项) 的那一个值挪到对应的列上，any_value 足够
        cols.append(f"any_value(vnum) FILTER (WHERE code = ${key}) AS num{idx}")
        cols.append(f"any_value(vtext) FILTER (WHERE code = ${key}) AS txt{idx}")
        plan[idx] = [f'coalesce({a}.txt{idx}, CAST({a}.num{idx} AS VARCHAR)) AS "{vid}"']

    # 两阶段：先按 (人, 项) 各聚合一次，再摊平成宽表。
    # 一步到位地写成「每个变量一个带 ORDER BY 的 FILTER 聚合」的话，
    # 有序聚合的次数是 2×变量数，每次都要在组内排序；MIMIC 的 32 个测量项
    # （135756 行）要 270ms。先分组聚合再摊平只排一次，降到 170ms。
    inner = (
        f"SELECT person_id, code, {numeric_expr} AS vnum,"
        f" first(value_text ORDER BY ts NULLS FIRST) AS vtext"
        f" FROM measurement WHERE dataset = $ds AND code IN ({', '.join(codes)})"
        f" GROUP BY person_id, code"
    )
    join = (
        f"LEFT JOIN (SELECT person_id, {', '.join(cols)} FROM ({inner})"
        f" GROUP BY person_id) {a} ON {a}.person_id = p.person_id"
    )
    return plan, join


def _condition_plan(
    entries: list[tuple[int, str, str]], params: dict[str, Any]
) -> tuple[_Plan, str]:
    a = SOURCE_ALIAS["condition"]
    plan: _Plan = {}
    cols: list[str] = []
    codes: list[str] = []

    for idx, vid, code in entries:
        key = f"c{idx}"
        params[key] = code
        codes.append(f"${key}")
        cols.append(f"count(*) FILTER (WHERE code = ${key}) > 0 AS has{idx}")
        # 一条诊断都没有的人根本不在子查询里，LEFT JOIN 给 NULL —— 也是「否」
        plan[idx] = [f'CAST(coalesce({a}.has{idx}, false) AS INTEGER) AS "{vid}"']

    join = (
        f"LEFT JOIN (SELECT person_id, {', '.join(cols)}"
        f" FROM condition WHERE dataset = $ds AND code IN ({', '.join(codes)})"
        f" GROUP BY person_id) {a} ON {a}.person_id = p.person_id"
    )
    return plan, join


def _visit_plan(
    entries: list[tuple[int, str, str]], params: dict[str, Any]
) -> tuple[_Plan, str]:
    a = SOURCE_ALIAS["visit"]
    # 时长按秒算再折成天，避免整除丢掉不足一天的部分
    los = "date_diff('second', start_ts, end_ts) / 86400.0"
    plan: _Plan = {}
    cols: list[str] = []
    types: list[str] = []

    for idx, vid, key_str in entries:
        metric, _, visit_type = key_str.partition(":")
        key = f"c{idx}"
        params[key] = visit_type
        types.append(f"${key}")

        if metric == "discharge_status":
            cols.append(
                f"last(discharge_status ORDER BY end_ts NULLS FIRST)"
                f" FILTER (WHERE visit_type = ${key} AND discharge_status IS NOT NULL)"
                f" AS st{idx}"
            )
            plan[idx] = [f'{a}.st{idx} AS "{vid}"']
            continue

        aggregate = {
            "count": "count(*)",
            "los_total": f"sum({los})",
            "los_max": f"max({los})",
        }[metric]
        cols.append(f"{aggregate} FILTER (WHERE visit_type = ${key}) AS val{idx}")
        # 次数缺省补 0（没住过就是 0 次）；时长缺省保持 NULL（没住过谈不上住了几天）
        expr = f"coalesce({a}.val{idx}, 0)" if metric == "count" else f"{a}.val{idx}"
        plan[idx] = [f'CAST({expr} AS VARCHAR) AS "{vid}"']

    join = (
        f"LEFT JOIN (SELECT person_id, {', '.join(cols)}"
        f" FROM visit WHERE dataset = $ds AND visit_type IN ({', '.join(types)})"
        f" GROUP BY person_id) {a} ON {a}.person_id = p.person_id"
    )
    return plan, join


def _outcome_plan(
    entries: list[tuple[int, str, str]], params: dict[str, Any]
) -> tuple[_Plan, str]:
    a = SOURCE_ALIAS["outcome"]
    plan: _Plan = {}
    cols: list[str] = []
    events: list[str] = []

    for idx, vid, event_type in entries:
        key = f"c{idx}"
        params[key] = event_type
        events.append(f"${key}")
        cols.append(f"bool_or(is_event) FILTER (WHERE event_type = ${key}) AS ev{idx}")
        cols.append(f"max(followup_days) FILTER (WHERE event_type = ${key}) AS fu{idx}")
        plan[idx] = [
            f'CAST({a}.ev{idx} AS INTEGER) AS "{vid}"',
            f'{a}.fu{idx} AS "{vid}{TIME_SUFFIX}"',
        ]

    join = (
        f"LEFT JOIN (SELECT person_id, {', '.join(cols)}"
        f" FROM outcome WHERE dataset = $ds AND event_type IN ({', '.join(events)})"
        f" GROUP BY person_id) {a} ON {a}.person_id = p.person_id"
    )
    return plan, join


_PLANNERS = {
    "condition": _condition_plan,
    "visit": _visit_plan,
    "outcome": _outcome_plan,
}


def _compile_variables(
    variables: list[str], agg: MeasurementAgg, params: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """把变量列表编译成 (SELECT 片段, JOIN 片段)。

    按来源表分组，每张表只扫一次；同一张表里各变量的聚合靠 FILTER 分开算。
    输出的列顺序仍按传入的变量顺序，不受内部分组影响。

    绑定参数由本函数一并生成 —— 由调用方去猜「这个变量要绑什么」，
    就会出现「JOIN 里没用到 $c0 却仍然绑了它」这类参数数量对不上的错。
    """
    grouped: dict[str, list[tuple[int, str, str]]] = {}
    plan: _Plan = {}

    for idx, vid in enumerate(variables):
        source, _, key = vid.partition(".")
        if source == "person":
            plan[idx] = [f'p.{key} AS "{vid}"']
            continue
        if source not in SOURCE_ALIAS:
            raise ValueError(f"未知变量来源：{source}")
        grouped.setdefault(source, []).append((idx, vid, key))

    joins: list[str] = []
    for source, entries in grouped.items():
        if source == "measurement":
            part, join = _measurement_plan(entries, agg, params)
        else:
            part, join = _PLANNERS[source](entries, params)
        plan.update(part)
        joins.append(join)

    selects = [expr for idx in range(len(variables)) for expr in plan[idx]]
    return selects, joins


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
    params: dict[str, Any] = {"ds": dataset}
    var_selects, joins = _compile_variables(variables, agg, params)

    selects = [
        "p.person_id",
        f'p.sample_weight AS "{WEIGHT_COLUMN}"',
        f'p.stratum AS "{STRATUM_COLUMN}"',
        f'p.psu AS "{PSU_COLUMN}"',
        *var_selects,
    ]

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
