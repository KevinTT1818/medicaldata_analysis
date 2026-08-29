"""Adapter 抽象基类。

一个 adapter 负责把一个数据集的原始文件转成 CDM 的 6 张表。
这是全系统唯一需要为每个数据集重写的地方 —— 加数据集只改这里，
所有已有的分析算子立刻可用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

CdmTables = dict[str, pl.DataFrame]


class Adapter(ABC):
    dataset_id: str
    label: str
    description: str
    source_url: str
    version: str

    @abstractmethod
    def is_available(self) -> bool:
        """原始文件是否已下载到 data/raw/。"""

    @abstractmethod
    def build(self) -> CdmTables:
        """解析原始文件，返回 CDM 各表的 DataFrame（键为表名）。"""


_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    _REGISTRY[adapter.dataset_id] = adapter
    return adapter


def get(dataset_id: str) -> Adapter:
    if dataset_id not in _REGISTRY:
        raise KeyError(f"未知数据集：{dataset_id}")
    return _REGISTRY[dataset_id]


def all_adapters() -> list[Adapter]:
    return list(_REGISTRY.values())
