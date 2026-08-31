"""分析算子的抽象契约。

每个算子自描述：需要哪些变量、接受哪些参数（Pydantic 模型，
会被转成 JSON Schema 交给前端渲染表单）、产出什么形状的结果。
加一个新分析 = 加一个这样的类，前端零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl
from pydantic import BaseModel

from ..cohort.builder import Variable


@dataclass
class AnalysisContext:
    dataset: str
    frame: pl.DataFrame                 # 一人一行的宽表
    catalog: dict[str, Variable]        # 变量元信息（类型、单位、水平）
    params: BaseModel
    progress: Callable[[str, float], None] = lambda s, p: None


@dataclass
class AnalysisResult:
    kind: str                           # 前端据此选渲染组件
    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class Analysis(ABC):
    id: str
    label: str
    description: str
    result_kind: str
    Params: type[BaseModel]

    def required_variables(self, params: BaseModel) -> list[str]:
        """本次运行需要从 CDM 取哪些变量。runner 据此构建宽表。"""
        vars_: list[str] = list(getattr(params, "variables", []) or [])
        gb = getattr(params, "group_by", None)
        if gb and gb not in vars_:
            vars_.append(gb)
        return vars_

    @abstractmethod
    def run(self, ctx: AnalysisContext) -> AnalysisResult: ...
