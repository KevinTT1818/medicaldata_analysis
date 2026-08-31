"""报告的数据模型。

报告存的是**分析的定义**，不是算出来的数字 —— 这样才能在数据更新后重跑，
也才能回答「三个月前那份报告现在还成立吗」。

同时存一份保存时的结果指纹（snapshot）。重跑时拿新结果的指纹和它比，
就能明确指出哪一节的数字变了、变在哪里。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..cohort.builder import DEFAULT_AGG, MeasurementAgg
from ..cohort.filters import Node


class ReportSection(BaseModel):
    """报告的一节：一次分析的完整定义。"""

    title: str = Field(..., min_length=1, max_length=120)
    #: 作者对这一节的解读或说明，导出时排在结果下方
    note: str | None = None
    dataset: str
    analysis: str
    params: dict[str, Any] = Field(default_factory=dict)
    #: 队列：引用已保存的，或内联条件树
    cohort_id: str | None = None
    cohort: Node | None = None
    measurement_agg: MeasurementAgg = DEFAULT_AGG


class ReportInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=160)
    description: str | None = None
    sections: list[ReportSection] = Field(default_factory=list)


class SectionSnapshot(BaseModel):
    """保存报告时记下的、这一节结果的可比对指纹。"""

    fingerprint: str          # 分析定义的指纹（队列 + 参数 + 聚合）
    dataset_version: str      # 数据集版本（导入时间戳）
    result_hash: str          # 结果内容的哈希
    n: int | None = None      # 纳入例数，变了最容易被人一眼看出
