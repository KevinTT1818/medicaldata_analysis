"""算子注册表。

GET /api/analyses/registry 把每个算子的 Params 转成 JSON Schema 返回，
前端据此动态渲染参数表单。
"""
from __future__ import annotations

from typing import Any

from .base import Analysis

_REGISTRY: dict[str, Analysis] = {}


def register(cls: type[Analysis]) -> type[Analysis]:
    _REGISTRY[cls.id] = cls()
    return cls


def get(analysis_id: str) -> Analysis:
    if analysis_id not in _REGISTRY:
        raise KeyError(f"未知算子：{analysis_id}")
    return _REGISTRY[analysis_id]


def describe_all() -> list[dict[str, Any]]:
    return [
        {
            "id": a.id,
            "label": a.label,
            "description": a.description,
            "result_kind": a.result_kind,
            "params_schema": a.Params.model_json_schema(),
        }
        for a in _REGISTRY.values()
    ]
