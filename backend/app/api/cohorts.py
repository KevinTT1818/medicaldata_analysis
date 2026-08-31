from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..cohort.builder import DEFAULT_AGG, MeasurementAgg
from ..cohort import preview as preview_module
from ..cohort import store as cohort_store
from ..cohort.filters import ALLOWED_OPS, OP_SYMBOL, FilterError, Node

router = APIRouter(prefix="/api/cohorts", tags=["cohorts"])


class PreviewRequest(BaseModel):
    dataset: str
    definition: Node | None = None
    measurement_agg: MeasurementAgg = DEFAULT_AGG


class SaveRequest(BaseModel):
    dataset: str
    name: str = Field(..., min_length=1, max_length=80)
    description: str | None = None
    definition: Node


@router.get("/operators")
def operators() -> dict[str, Any]:
    """各类变量支持的运算符，前端据此渲染条件编辑器。"""
    return {
        "allowed": ALLOWED_OPS,
        "symbols": OP_SYMBOL,
        "no_value_ops": ["is_null", "not_null"],
        "list_ops": ["in", "not_in"],
        "range_ops": ["between"],
    }


@router.post("/preview")
def preview(req: PreviewRequest) -> dict[str, Any]:
    try:
        return preview_module.preview(req.dataset, req.definition, req.measurement_agg)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FilterError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("")
def list_cohorts(dataset: str | None = None) -> list[dict[str, Any]]:
    return cohort_store.list_for(dataset)


@router.post("", status_code=201)
def create(req: SaveRequest) -> dict[str, Any]:
    return cohort_store.save(req.dataset, req.name, req.definition, req.description)


@router.get("/{cohort_id}")
def get_cohort(cohort_id: str) -> dict[str, Any]:
    try:
        return cohort_store.get(cohort_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.put("/{cohort_id}")
def update(cohort_id: str, req: SaveRequest) -> dict[str, Any]:
    try:
        return cohort_store.save(
            req.dataset, req.name, req.definition, req.description, cohort_id=cohort_id
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/{cohort_id}", status_code=204)
def remove(cohort_id: str) -> None:
    cohort_store.delete(cohort_id)
