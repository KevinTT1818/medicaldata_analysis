from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..cohort.filters import FilterError
from ..report import docx_export
from ..report import runner as report_runner
from ..report import store as report_store
from ..report.models import ReportInput, SectionSnapshot

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ExportRequest(BaseModel):
    """导出请求。

    图表由前端把 ECharts 画布转成 PNG data URL 一起送上来 —— 键是小节序号，
    值是该节的图片列表（一节可能有多张图）。服务端不重画：重画要引一套绘图依赖，
    还得和用户屏幕上看到的完全一致，那是两份必然互相偏离的实现。
    """

    images: dict[str, list[str]] = Field(default_factory=dict)


class SaveWithBaseline(BaseModel):
    report: ReportInput
    #: 是否在保存时跑一遍并把结果记为基线快照，供日后核对可复现性
    baseline: bool = True


@router.get("")
def list_reports() -> list[dict[str, Any]]:
    return report_store.list_all()


@router.post("", status_code=201)
def create(req: SaveWithBaseline) -> dict[str, Any]:
    try:
        snapshot = _build_snapshot(req) if req.baseline else None
        return report_store.save(req.report, snapshot=snapshot)
    except FilterError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    try:
        return report_store.get(report_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.put("/{report_id}")
def update(report_id: str, req: SaveWithBaseline) -> dict[str, Any]:
    try:
        snapshot = _build_snapshot(req) if req.baseline else None
        return report_store.save(req.report, snapshot=snapshot, report_id=report_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FilterError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/{report_id}", status_code=204)
def remove(report_id: str) -> None:
    report_store.delete(report_id)


@router.post("/{report_id}/run")
def run(report_id: str) -> dict[str, Any]:
    """重跑整份报告，并与保存时的基线快照逐节比对。"""
    try:
        return report_runner.run_report(report_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{report_id}/rebaseline")
def rebaseline(report_id: str) -> dict[str, Any]:
    """以当前结果为准重设基线。数据更新后确认过新数字无误时用。"""
    try:
        sections = report_store.load_sections(report_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    outputs = [report_runner.run_section(s) for s in sections]
    failed = [o["title"] for o in outputs if o["status"] == "failed"]
    if failed:
        raise HTTPException(409, f"这些小节跑不通，不能重设基线：{'、'.join(failed)}")

    snapshot = [SectionSnapshot(**o["snapshot"]) for o in outputs]
    report_store.update_snapshot(report_id, snapshot)
    return report_store.get(report_id)


def _build_snapshot(req: SaveWithBaseline) -> list[SectionSnapshot]:
    outputs = [report_runner.run_section(s) for s in req.report.sections]
    failed = [o["title"] for o in outputs if o["status"] == "failed"]
    if failed:
        raise HTTPException(
            400, f"这些小节跑不通，无法建立基线：{'、'.join(failed)}"
        )
    return [SectionSnapshot(**o["snapshot"]) for o in outputs]


@router.post("/{report_id}/export/docx")
def export_docx(report_id: str, req: ExportRequest) -> StreamingResponse:
    """重跑报告并导出成 Word 文档。结果走缓存，重跑通常很快。"""
    try:
        run = report_runner.run_report(report_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    buffer = docx_export.build(run, req.images)
    filename = quote(f"{run['report']['title']}.docx")
    return StreamingResponse(
        buffer,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={
            # 文件名含中文，必须用 RFC 5987 的 filename* 形式
            "Content-Disposition": f"attachment; filename*=UTF-8\'\'{filename}"
        },
    )
