"""队列预览：命中人数、CONSORT 入排流程、人口学摘要。"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..cdm import values
from . import filters
from .builder import (
    DEFAULT_AGG,
    MeasurementAgg,
    Variable,
    build_feature_frame,
    list_variables,
)

#: 摘要里固定展示的人口学变量。数据集没有的自动跳过。
SUMMARY_VARIABLES = ("person.age_at_index", "person.gender")


def _summarise(frame, catalog: dict[str, Variable], mask: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for vid in SUMMARY_VARIABLES:
        variable = catalog.get(vid)
        if variable is None:
            continue

        if variable.kind == "continuous":
            column = values.as_numeric(frame, vid)[mask]
            column = column[np.isfinite(column)]
            if column.size == 0:
                continue
            q1, median, q3 = (float(x) for x in np.percentile(column, [25, 50, 75]))
            out.append({
                "variable": vid, "label": variable.label, "unit": variable.unit,
                "kind": "continuous", "n": int(column.size),
                "median": round(median, 2), "q1": round(q1, 2), "q3": round(q3, 2),
            })
        else:
            series = values.as_category(frame, vid, variable)
            arr = series.to_numpy()[mask]
            levels = values.levels_of(series, variable)
            total = int(sum(1 for x in arr if x is not None))
            out.append({
                "variable": vid, "label": variable.label, "unit": None,
                "kind": "categorical", "n": total,
                "levels": [
                    {
                        "label": lv,
                        "n": int((arr == lv).sum()),
                        "pct": round(float((arr == lv).sum()) / total * 100, 1) if total else 0.0,
                    }
                    for lv in levels
                ],
            })
    return out


def preview(
    dataset: str,
    node: filters.Node | None,
    agg: MeasurementAgg = DEFAULT_AGG,
) -> dict[str, Any]:
    """不落盘地算一遍队列，返回命中人数、入排流程与人口学摘要。

    agg 必须与后续分析用的一致，否则预览人数和实际分析人数会对不上。
    """
    catalog = {v.id: v for v in list_variables(dataset)}
    if not catalog:
        raise KeyError(f"数据集 {dataset} 尚未导入")

    needed = list(dict.fromkeys(
        [*filters.collect_variables(node),
         *(v for v in SUMMARY_VARIABLES if v in catalog)]
    ))
    frame = build_feature_frame(dataset, needed, agg=agg)

    mask = (
        np.ones(frame.height, dtype=bool) if node is None
        else filters.evaluate(frame, catalog, node)
    )
    flow = filters.consort_flow(frame, catalog, node)

    return {
        "dataset": dataset,
        "total": frame.height,
        "n": int(mask.sum()),
        "pct": round(float(mask.sum()) / frame.height * 100, 1) if frame.height else 0.0,
        "flow": flow,
        "summary": _summarise(frame, catalog, mask),
    }
