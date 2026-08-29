from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..jobs import runner

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(limit: int = 50) -> list[dict]:
    return runner.list_jobs(limit)


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return runner.get(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{job_id}/result")
def get_result(job_id: str) -> dict:
    try:
        job = runner.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job.status == "failed":
        raise HTTPException(409, job.error or "任务失败")
    if job.status != "succeeded":
        raise HTTPException(409, f"任务尚未完成（当前：{job.stage}）")
    return {"job": job.public(), "result": job.result}


@router.get("/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    try:
        runner.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    async def stream():
        last = -1
        while True:
            job = runner.get(job_id)
            if job.version != last:
                last = job.version
                yield f"data: {json.dumps(job.public(), ensure_ascii=False)}\n\n"
            if job.status in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
