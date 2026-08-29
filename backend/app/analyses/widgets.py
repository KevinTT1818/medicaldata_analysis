"""参数字段的前端控件标注。

JSON Schema 只描述数据类型，表达不了「这个字符串是一个变量 ID，
且只能选带随访时长的结局变量」。这里用 x- 扩展字段补上这层语义，
前端 SchemaForm 据此选择渲染成变量选择器还是普通输入框。
"""
from __future__ import annotations

from typing import Any

# 变量筛选器：前端据此过滤可选变量
ANY = "any"                        # 全部变量
GROUPING = "grouping"              # 可作分组的（分类/二分类）
SURVIVAL_OUTCOME = "survival"      # 带随访时长的结局
BINARY_OUTCOME = "binary"          # 二分类结局
COVARIATE = "covariate"            # 可作协变量的（排除纯标识列）


def variables(title: str, description: str, filter_: str = ANY) -> dict[str, Any]:
    """多选变量。"""
    return {"x-widget": "variables", "x-variable-filter": filter_,
            "title": title, "description": description}


def variable(title: str, description: str, filter_: str = ANY) -> dict[str, Any]:
    """单选变量。"""
    return {"x-widget": "variable", "x-variable-filter": filter_,
            "title": title, "description": description}
