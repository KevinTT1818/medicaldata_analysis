from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..analyses import describe, regression, survival  # noqa: F401  触发注册
from ..analyses import registry
from ..cohort.builder import DEFAULT_AGG, MeasurementAgg
from ..cohort.filters import FilterError, Node
from ..jobs import runner

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


class AnalysisRequest(BaseModel):
    dataset: str
    analysis: str
    params: dict = Field(default_factory=dict)
    #: 内联条件树，或引用一个已保存的队列，二选一
    cohort: Node | None = None
    cohort_id: str | None = None
    #: 一人一项有多个值时取哪一个。住院时序数据上这个选择会改变结论。
    measurement_agg: MeasurementAgg = DEFAULT_AGG


@router.get("/registry")
def analysis_registry() -> list[dict]:
    return registry.describe_all()


@router.post("", status_code=202)
def submit(req: AnalysisRequest) -> dict:
    try:
        registry.get(req.analysis)
        return runner.submit_analysis(
            req.dataset, req.analysis, req.params,
            cohort=req.cohort, cohort_id=req.cohort_id,
            measurement_agg=req.measurement_agg,
        ).public()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FilterError as exc:
        raise HTTPException(400, str(exc)) from exc
