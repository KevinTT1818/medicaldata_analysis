"""报告的运行与可复现性核对。

设计文档给 M5 定的验收标准是「把一份三个月前的报告重跑一遍，数字完全一致」。
这里把那句话做成可执行的检查：重跑每一节，把新结果的指纹和保存时的快照逐条比对，
不一致就明确指出是哪一节、变了什么。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..jobs import cache, runner as job_runner
from .models import ReportSection, SectionSnapshot
from . import store as report_store


def result_hash(payload: dict[str, Any]) -> str:
    """结果内容的哈希。键排序后序列化，保证同样的数字得到同样的哈希。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _extract_n(payload: dict[str, Any]) -> int | None:
    """从结果里取「纳入例数」。各类结果放的位置不同。"""
    for key in ("overall_n", "n_used"):
        if isinstance(payload.get(key), int):
            return payload[key]
    series = payload.get("series")
    if isinstance(series, list) and series and isinstance(series[0], dict):
        total = sum(s.get("n", 0) for s in series if isinstance(s, dict))
        return total or None
    return None


def run_section(section: ReportSection) -> dict[str, Any]:
    """跑一节，等它结束，返回结果与快照。"""
    job = job_runner.submit_analysis(
        section.dataset, section.analysis, section.params,
        cohort=section.cohort, cohort_id=section.cohort_id,
        measurement_agg=section.measurement_agg,
    )
    job_runner.wait(job)

    if job.status == "failed":
        return {
            "title": section.title,
            "status": "failed",
            "error": job.error,
            "job": job.public(),
            "result": None,
            "snapshot": None,
        }

    payload = job.result or {}
    snapshot = SectionSnapshot(
        fingerprint=job.spec.get("fingerprint", ""),
        dataset_version=cache.dataset_version(section.dataset),
        result_hash=result_hash(payload),
        n=_extract_n(payload),
    )
    return {
        "title": section.title,
        "note": section.note,
        "status": "succeeded",
        "error": None,
        "job": job.public(),
        "result": payload,
        "snapshot": snapshot.model_dump(),
    }


def run_report(report_id: str) -> dict[str, Any]:
    """重跑整份报告，并与保存时的快照比对。"""
    meta = report_store.get(report_id)
    sections = report_store.load_sections(report_id)
    baseline = report_store.load_snapshot(report_id)

    outputs = [run_section(s) for s in sections]
    comparisons = _compare(outputs, baseline)

    return {
        "report": {k: meta[k] for k in ("id", "title", "description",
                                        "created_at", "updated_at")},
        "sections": outputs,
        # 没有基线只是「没核对过」，不是「可复现」。报成 True 会给人虚假的安心。
        "reproducible": bool(comparisons) and all(
            c["verdict"] == "match" for c in comparisons
        ),
        "has_baseline": baseline is not None,
        "comparisons": comparisons,
    }


def _compare(
    outputs: list[dict[str, Any]], baseline: list[SectionSnapshot] | None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        entry: dict[str, Any] = {"index": index, "title": output["title"]}

        if output["status"] == "failed":
            entry.update(verdict="failed", detail=output["error"])
            result.append(entry)
            continue
        if baseline is None or index >= len(baseline):
            entry.update(verdict="no_baseline", detail="没有基线快照可比对")
            result.append(entry)
            continue

        before = baseline[index]
        after = SectionSnapshot(**output["snapshot"])

        if before.result_hash == after.result_hash:
            entry.update(verdict="match", detail="结果与保存时完全一致")
        else:
            changes = []
            if before.fingerprint != after.fingerprint:
                changes.append("分析定义变了（队列 / 参数 / 聚合策略）")
            if before.dataset_version != after.dataset_version:
                changes.append(
                    f"数据集重新导入过（{before.dataset_version} → {after.dataset_version}）"
                )
            if before.n != after.n:
                changes.append(f"纳入例数 {before.n} → {after.n}")
            if not changes:
                changes.append("定义和数据版本都没变，但数字变了 —— 检查算子代码是否改动")
            entry.update(verdict="changed", detail="；".join(changes))

        entry["baseline"] = before.model_dump()
        entry["current"] = after.model_dump()
        result.append(entry)
    return result
