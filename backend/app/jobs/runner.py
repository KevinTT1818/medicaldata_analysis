"""异步任务执行。

M1 用进程内线程池 + 内存任务表。够单机单人用。
等到需要多 worker、任务持久化、失败重试时再换 Celery / RQ —— 不要提前上。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ..adapters import heart_failure as _hf        # noqa: F401  触发适配器注册
from ..adapters import mimic_demo as _mimic        # noqa: F401
from ..adapters import nhanes as _nhanes           # noqa: F401
from ..adapters import uci_heart as _uci           # noqa: F401
from ..analyses import describe as _describe      # noqa: F401  触发算子注册
from ..analyses import regression as _regression  # noqa: F401
from ..analyses import survival as _survival      # noqa: F401
from ..analyses import registry
from ..analyses.base import AnalysisContext
from ..config import MAX_RETAINED_JOBS
from ..cohort import builder, filters
from ..cohort import store as cohort_store
from . import cache

Status = Literal["pending", "running", "succeeded", "failed"]

log = logging.getLogger("app.jobs")

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="analysis")
_JOBS: dict[str, "Job"] = {}
_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    kind: str                       # analysis | import
    spec: dict[str, Any]
    status: Status = "pending"
    stage: str = "排队中"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    cached: bool = False             # 结果是否来自缓存
    error: str | None = None
    created_at: str = ""
    finished_at: str | None = None
    version: int = 0                # 每次状态变化 +1，SSE 据此推送
    cohort_n: int | None = None     # 队列筛选后的人数
    cohort_total: int | None = None # 数据集全量人数

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "stage": self.stage, "progress": round(self.progress, 4),
            "error": self.error, "warnings": self.warnings,
            "created_at": self.created_at, "finished_at": self.finished_at,
            "cohort_n": self.cohort_n, "cohort_total": self.cohort_total,
            "cached": self.cached,
            "spec": self.spec,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def spec_hash(spec: dict[str, Any]) -> str:
    """可复现性指纹：队列定义 + 算子 + 参数 + 数据集，一起哈希。"""
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def wait(job: Job, timeout: float = 300.0, interval: float = 0.02) -> Job:
    """阻塞到任务终结。报告要按顺序跑完所有小节，需要同步等待。"""
    deadline = time.monotonic() + timeout
    while job.status not in ("succeeded", "failed"):
        if time.monotonic() > deadline:
            raise TimeoutError(f"任务 {job.id} 超过 {timeout} 秒仍未结束")
        time.sleep(interval)
    return job


def get(job_id: str) -> Job:
    with _LOCK:
        if job_id not in _JOBS:
            raise KeyError(f"未知任务：{job_id}")
        return _JOBS[job_id]


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    return [j.public() for j in jobs[:limit]]


def _evict_locked() -> int:
    """把最早结束的任务从内存表里丢掉，只保留 MAX_RETAINED_JOBS 条。

    只淘汰已终结的任务 —— 运行中的丢掉会让前端的进度流查不到自己。
    结果本来就有落盘缓存，丢掉内存副本不影响可复现，只是要重新提交一次。
    必须在持有 _LOCK 的情况下调用。
    """
    if len(_JOBS) <= MAX_RETAINED_JOBS:
        return 0
    done = sorted(
        (j for j in _JOBS.values() if j.status in ("succeeded", "failed")),
        key=lambda j: j.finished_at or j.created_at,
    )
    drop = len(_JOBS) - MAX_RETAINED_JOBS
    removed = 0
    for job in done[:drop]:
        del _JOBS[job.id]
        removed += 1
    if removed:
        log.debug("任务表淘汰 %d 条，当前 %d 条", removed, len(_JOBS))
    return removed


def job_count() -> int:
    with _LOCK:
        return len(_JOBS)


def _submit(kind: str, spec: dict[str, Any], fn) -> Job:
    """提交一个任务。fn 收到 (report, job) —— 直接传 job，
    不能靠调用方在 _submit 返回后再回填，那样工作线程可能已经跑完了。"""
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, spec=spec, created_at=_now())
    with _LOCK:
        _JOBS[job.id] = job
        _evict_locked()
    log.info("提交 %s 任务 %s %s", kind, job.id, spec.get("analysis") or spec.get("dataset", ""))

    def report(stage: str, pct: float) -> None:
        job.stage, job.progress, job.version = stage, pct, job.version + 1

    def wrapped() -> None:
        job.status, job.stage, job.version = "running", "启动", job.version + 1
        started = time.perf_counter()
        try:
            payload, warnings = fn(report, job)
            job.result, job.warnings = payload, warnings
            job.status, job.stage, job.progress = "succeeded", "完成", 1.0
            log.info(
                "任务 %s 完成  %.0fms%s%s", job.id, (time.perf_counter() - started) * 1000,
                "  命中缓存" if job.cached else "",
                f"  队列 {job.cohort_n}/{job.cohort_total} 例" if job.cohort_n is not None else "",
            )
        except Exception as exc:                      # noqa: BLE001
            job.status, job.stage = "failed", "失败"
            job.error = f"{type(exc).__name__}: {exc}"
            # exc_info：把栈留在日志里。原先是 traceback.print_exc()，
            # 打到 stdout 且不带任务号，事后对不上是哪一次分析。
            log.error("任务 %s 失败：%s", job.id, job.error,
                      exc_info=True, extra={"spec": job.spec})
        finally:
            job.finished_at = _now()
            job.version += 1

    _POOL.submit(wrapped)
    return job


def submit_analysis(
    dataset: str,
    analysis_id: str,
    params: dict[str, Any],
    cohort: filters.Node | None = None,
    cohort_id: str | None = None,
    measurement_agg: builder.MeasurementAgg = builder.DEFAULT_AGG,
) -> Job:
    """提交一次分析。cohort 传内联条件树，cohort_id 引用已保存的队列，二选一。

    measurement_agg 决定一人一项有多个值时取哪一个。它会改变结论，所以进指纹。
    """
    node = cohort_store.load_node(cohort_id) if cohort_id else cohort

    spec: dict[str, Any] = {
        "dataset": dataset, "analysis": analysis_id, "params": params,
        "measurement_agg": measurement_agg,
    }
    if node is not None:
        # 队列定义进指纹 —— 同样的算子换个队列必须是另一次分析
        spec["cohort"] = filters.dump_node(node)
    if cohort_id:
        spec["cohort_id"] = cohort_id
    spec["fingerprint"] = spec_hash(spec)

    cache_key = cache.key_for(spec, cache.dataset_version(dataset))

    def run(report, job: Job):
        report("查缓存", 0.01)
        hit = cache.get(cache_key)
        if hit is not None:
            job.cached = True
            job.cohort_n = hit.get("cohort_n")
            job.cohort_total = hit.get("cohort_total")
            report("命中缓存", 1.0)
            return hit["payload"], hit.get("warnings", [])

        analysis = registry.get(analysis_id)
        parsed = analysis.Params(**params)

        report("读取变量目录", 0.02)
        catalog = {v.id: v for v in builder.list_variables(dataset)}

        needed = analysis.required_variables(parsed)
        unknown = [v for v in needed if v not in catalog]
        if unknown:
            raise ValueError(f"数据集 {dataset} 里没有这些变量：{', '.join(unknown)}")
        if not needed:
            raise ValueError("至少要选一个变量")

        # 筛选条件用到的变量也要取进宽表，算子只读自己那几列，多带的无害
        columns = list(dict.fromkeys([*needed, *filters.collect_variables(node)]))

        report("构建队列宽表", 0.06)
        # 队列条件下推到 SQL：在物化到 Python 之前就把人筛掉
        frame = builder.build_feature_frame(
            dataset, columns, where=node, agg=measurement_agg)
        total = builder.person_count(dataset)

        if node is not None and frame.height == 0:
            raise ValueError("筛选条件太严，队列里一个人都没有")

        job.cohort_n, job.cohort_total = frame.height, total
        report(f"队列 {frame.height} / {total} 例", 0.12)

        ctx = AnalysisContext(dataset=dataset, frame=frame, catalog=catalog,
                              params=parsed, progress=report)
        result = analysis.run(ctx)

        report("写缓存", 0.97)
        cache.put(cache_key, result.payload, result.warnings, frame.height, total)
        return result.payload, result.warnings

    return _submit("analysis", spec, run)


def submit_import(dataset: str) -> Job:
    from ..cdm import etl

    def run(report, job: Job):
        return etl.import_dataset(dataset, progress=report), []

    return _submit("import", {"dataset": dataset}, run)
