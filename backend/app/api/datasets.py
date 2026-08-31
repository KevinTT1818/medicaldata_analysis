from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import store
from ..adapters import base as adapters
from ..adapters import heart_failure, mimic_demo, nhanes, uci_heart  # noqa: F401  触发注册
from ..cohort import builder
from ..jobs import runner

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _imported() -> dict[str, dict]:
    with store.read() as cur:
        rows = cur.execute(
            "SELECT dataset, n_person, imported_at, version FROM dataset_registry"
        ).fetchall()
    return {r[0]: {"n_person": r[1], "imported_at": str(r[2]), "version": r[3]} for r in rows}


@router.get("")
def list_datasets() -> list[dict]:
    done = _imported()
    return [
        {
            "id": a.dataset_id,
            "label": a.label,
            "description": a.description,
            "source_url": a.source_url,
            "version": a.version,
            "available": a.is_available(),
            "imported": a.dataset_id in done,
            **done.get(a.dataset_id, {}),
        }
        for a in adapters.all_adapters()
    ]


@router.post("/{dataset_id}/import", status_code=202)
def import_dataset(dataset_id: str) -> dict:
    try:
        adapters.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return runner.submit_import(dataset_id).public()


@router.get("/{dataset_id}/schema")
def dataset_schema(dataset_id: str) -> dict:
    variables = builder.list_variables(dataset_id)
    if not variables:
        raise HTTPException(404, f"数据集 {dataset_id} 尚未导入")
    return {
        "dataset": dataset_id,
        "n_person": max(v.n_available + v.n_missing for v in variables),
        "hidden_variables": builder.hidden_variable_count(dataset_id),
        "has_repeated_measures": builder.has_repeated_measures(dataset_id),
        "has_sample_weights": builder.has_sample_weights(dataset_id),
        "variables": [v.to_dict() for v in variables],
    }
