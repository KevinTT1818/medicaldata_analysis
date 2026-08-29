"""队列筛选条件：嵌套的 AND / OR 条件树，以及 CONSORT 式入排流程统计。

缺失值语义：除 is_null / not_null 外，任何比较遇到缺失一律判 False。
这与 SQL 的 NULL 语义一致，临床上也更稳妥 —— 无法确认满足条件的人不该进队列。
但这会静默丢人，所以流程统计里把「不满足条件」和「该变量缺失」分开计数。
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from ..cdm import values
from .builder import Variable

Op = Literal[
    "eq", "ne", "lt", "lte", "gt", "gte",
    "between", "in", "not_in", "is_null", "not_null",
]

_CATEGORICAL_OPS: list[Op] = ["in", "not_in", "eq", "ne", "is_null", "not_null"]

#: 各类变量允许的运算符。前端据此决定给哪些选项。
#: binary 与 categorical 用同一套 —— 二分类只是水平数为 2 的分类变量，
#: 同一个概念变量在不同数据集里水平数可能不同，可用运算符不该跟着变。
ALLOWED_OPS: dict[str, list[Op]] = {
    "continuous": ["between", "gte", "gt", "lte", "lt", "eq", "ne", "is_null", "not_null"],
    "categorical": _CATEGORICAL_OPS,
    "binary": _CATEGORICAL_OPS,
}

OP_SYMBOL: dict[Op, str] = {
    "eq": "=", "ne": "≠", "lt": "<", "lte": "≤", "gt": ">", "gte": "≥",
    "between": "介于", "in": "属于", "not_in": "不属于",
    "is_null": "缺失", "not_null": "非缺失",
}

NO_VALUE_OPS = ("is_null", "not_null")

KIND_LABEL = {"continuous": "连续型", "categorical": "分类型", "binary": "二分类"}


class FilterError(ValueError):
    pass


class Condition(BaseModel):
    kind: Literal["condition"] = "condition"
    variable: str
    op: Op
    value: Any = None


class Group(BaseModel):
    kind: Literal["group"] = "group"
    op: Literal["and", "or"] = "and"
    children: list["Node"] = Field(default_factory=list)


Node = Annotated[Union[Condition, Group], Field(discriminator="kind")]
Group.model_rebuild()


_NODE_ADAPTER = None


def dump_node(node: Node) -> dict[str, Any]:
    """条件树转成可 JSON 序列化的 dict，用于指纹与持久化。"""
    global _NODE_ADAPTER
    if _NODE_ADAPTER is None:
        from pydantic import TypeAdapter
        _NODE_ADAPTER = TypeAdapter(Node)
    return _NODE_ADAPTER.dump_python(node, mode="json")


def collect_variables(node: Node | None) -> list[str]:
    """条件树里用到的全部变量，用于决定宽表要取哪些列。"""
    if node is None:
        return []
    if isinstance(node, Condition):
        return [node.variable]
    seen: list[str] = []
    for child in node.children:
        for v in collect_variables(child):
            if v not in seen:
                seen.append(v)
    return seen


def describe(condition: Condition, variable: Variable) -> str:
    """条件的人类可读描述，用在 CONSORT 流程图的每一步上。"""
    unit = f" {variable.unit}" if variable.unit else ""
    symbol = OP_SYMBOL[condition.op]

    if condition.op in NO_VALUE_OPS:
        return f"{variable.label} {symbol}"
    if condition.op == "between":
        lo, hi = condition.value
        return f"{variable.label} 介于 {lo}–{hi}{unit}"
    if condition.op in ("in", "not_in"):
        items = "、".join(str(x) for x in condition.value)
        return f"{variable.label} {symbol} {{{items}}}"
    return f"{variable.label} {symbol} {condition.value}{unit}"


def _missing_mask(frame: pl.DataFrame, variable: Variable) -> np.ndarray:
    if variable.kind == "continuous":
        return ~np.isfinite(values.as_numeric(frame, variable.id))
    return values.as_category(frame, variable.id, variable).is_null().to_numpy()


def _evaluate_condition(
    frame: pl.DataFrame, catalog: dict[str, Variable], condition: Condition
) -> np.ndarray:
    if condition.variable not in catalog:
        raise FilterError(f"数据集里没有变量：{condition.variable}")
    variable = catalog[condition.variable]

    allowed = ALLOWED_OPS[variable.kind]
    if condition.op not in allowed:
        raise FilterError(
            f"「{variable.label}」是{KIND_LABEL[variable.kind]}变量，"
            f"不支持「{OP_SYMBOL[condition.op]}」运算"
        )

    missing = _missing_mask(frame, variable)
    if condition.op == "is_null":
        return missing
    if condition.op == "not_null":
        return ~missing

    if condition.value is None:
        raise FilterError(f"「{variable.label}」的 {OP_SYMBOL[condition.op]} 条件缺少取值")

    if variable.kind == "continuous":
        column = values.as_numeric(frame, variable.id)
        # 先把取值解析出来。FilterError 继承自 ValueError，
        # 解析用的 except 必须只包住解析本身，否则会把更精确的报错替换掉。
        try:
            if condition.op == "between":
                lo, hi = (float(x) for x in condition.value)
            else:
                target = float(condition.value)
        except (TypeError, ValueError) as exc:
            raise FilterError(
                f"「{variable.label}」的取值不是数字：{condition.value}"
            ) from exc

        if condition.op == "between":
            if lo > hi:
                raise FilterError(f"「{variable.label}」的区间下界大于上界")
            result = (column >= lo) & (column <= hi)
        else:
            result = {
                "eq": column == target, "ne": column != target,
                "lt": column < target, "lte": column <= target,
                "gt": column > target, "gte": column >= target,
            }[condition.op]
    else:
        column = values.as_category(frame, variable.id, variable).to_numpy()
        if condition.op == "in":
            wanted = {str(x) for x in condition.value}
            result = np.array([str(x) in wanted for x in column])
        elif condition.op == "not_in":
            wanted = {str(x) for x in condition.value}
            result = np.array([str(x) not in wanted for x in column])
        else:
            target = str(condition.value)
            result = column == target if condition.op == "eq" else column != target

    # 缺失一律判 False，包括 ne / not_in —— 无法确认的人不进队列
    return np.asarray(result) & ~missing


def evaluate(frame: pl.DataFrame, catalog: dict[str, Variable], node: Node) -> np.ndarray:
    if isinstance(node, Condition):
        return _evaluate_condition(frame, catalog, node)

    if not node.children:
        # 空分组不筛任何人，等价于「全部纳入」
        return np.ones(frame.height, dtype=bool)

    masks = [evaluate(frame, catalog, child) for child in node.children]
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask if node.op == "and" else combined | mask
    return combined


def consort_flow(
    frame: pl.DataFrame, catalog: dict[str, Variable], node: Node | None
) -> dict[str, Any]:
    """CONSORT 式入排流程。

    顶层是 AND 分组时逐条累计施加，能给出每一步排除了多少人；
    顶层是 OR 或单条件时无法拆成序列，只报一步。
    """
    total = frame.height
    # 空分组等价于「没有筛选条件」，不该产生任何流程步骤
    if node is None or (isinstance(node, Group) and not node.children):
        return {"total": total, "final_n": total, "steps": [], "sequential": True}

    if isinstance(node, Group) and node.op == "and":
        steps, running = [], np.ones(total, dtype=bool)
        for child in node.children:
            mask = evaluate(frame, catalog, child)
            before = int(running.sum())
            after_mask = running & mask
            after = int(after_mask.sum())

            # 在这一步被排除的人里，有多少是因为该变量缺失而非不满足条件
            missing_share = 0
            if isinstance(child, Condition) and child.op not in NO_VALUE_OPS:
                missing = _missing_mask(frame, catalog[child.variable])
                missing_share = int((running & ~mask & missing).sum())

            steps.append({
                "label": (
                    describe(child, catalog[child.variable])
                    if isinstance(child, Condition)
                    else f"（{'且' if child.op == 'and' else '或'} 组合条件）"
                ),
                "n_before": before,
                "n_after": after,
                "n_excluded": before - after,
                "n_excluded_missing": missing_share,
            })
            running = after_mask

        return {"total": total, "final_n": int(running.sum()),
                "steps": steps, "sequential": True}

    mask = evaluate(frame, catalog, node)
    kept = int(mask.sum())
    return {
        "total": total,
        "final_n": kept,
        "steps": [{
            "label": (
                describe(node, catalog[node.variable])
                if isinstance(node, Condition) else "组合条件"
            ),
            "n_before": total, "n_after": kept,
            "n_excluded": total - kept, "n_excluded_missing": 0,
        }],
        # OR 条件无法拆成先后步骤，前端据此不画流程箭头
        "sequential": False,
    }


# ---------------------------------------------------------------- SQL 下推

def to_sql(
    node: Node | None,
    catalog: dict[str, Variable],
    params: dict[str, Any],
    prefix: str = "t",
) -> str | None:
    """把条件树编译成 SQL 的 WHERE 片段，让 DuckDB 在物化之前就把人筛掉。

    取值一律走绑定参数，不做字符串拼接。

    缺失语义与 Python 求值完全一致：SQL 里 NULL 参与比较得到 NULL，
    行不会被 WHERE 选中 —— 这正是「缺失一律判 False」，`ne` 和 `not_in` 也不例外。
    两条路径必须给出同一批人，否则同一个队列在预览和分析里人数会对不上。
    """
    if node is None:
        return None

    if isinstance(node, Group):
        if not node.children:
            return None
        parts = [to_sql(child, catalog, params, prefix) for child in node.children]
        kept = [x for x in parts if x]
        if not kept:
            return None
        joiner = " AND " if node.op == "and" else " OR "
        return "(" + joiner.join(kept) + ")"

    variable = catalog.get(node.variable)
    if variable is None:
        raise FilterError(f"数据集里没有变量：{node.variable}")

    column = f'{prefix}."{node.variable}"'
    numeric = variable.kind == "continuous"
    # 诊断与结局在宽表里是 0/1 整数，Python 侧求值时会先映射成「是 / 否」再比较，
    # SQL 侧必须做同一层映射，否则拿字符串去比整数列会直接报转换错。
    boolean_flag = variable.source in ("condition", "outcome")

    if numeric:
        # 宽表里测量列是 VARCHAR，数值比较前要转回来；TRY_CAST 转不动时给 NULL
        expr = f"TRY_CAST({column} AS DOUBLE)"
    elif boolean_flag:
        expr = column
    else:
        expr = column

    if node.op == "is_null":
        return f"{expr} IS NULL"
    if node.op == "not_null":
        return f"{expr} IS NOT NULL"

    def bind(value: Any) -> str:
        key = f"f{len(params)}"
        params[key] = value
        return f"${key}"

    def encode(value: Any) -> Any:
        """把用户输入的取值转成该列在宽表里的实际类型。"""
        if numeric:
            return float(value)
        if boolean_flag:
            text = str(value)
            if text not in ("是", "否"):
                raise FilterError(
                    f"「{variable.label}」只能取「是」或「否」，收到：{text}"
                )
            return 1 if text == "是" else 0
        return str(value)

    if node.op == "between":
        lo, hi = (float(x) for x in node.value)
        if lo > hi:
            raise FilterError(f"「{variable.label}」的区间下界大于上界")
        return f"{expr} BETWEEN {bind(lo)} AND {bind(hi)}"

    if node.op in ("in", "not_in"):
        placeholders = ", ".join(bind(encode(x)) for x in node.value)
        negate = "NOT " if node.op == "not_in" else ""
        return f"{expr} {negate}IN ({placeholders})"

    symbol = {"eq": "=", "ne": "<>", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[node.op]
    return f"{expr} {symbol} {bind(encode(node.value))}"
