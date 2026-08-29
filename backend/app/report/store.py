"""报告定义的持久化。"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import TypeAdapter

from .. import store
from .models import ReportInput, ReportSection, SectionSnapshot

_SECTIONS = TypeAdapter(list[ReportSection])
_SNAPSHOT = TypeAdapter(list[SectionSnapshot])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "title": row[1],
        "description": row[2],
        "sections": json.loads(row[3]),
        "snapshot": json.loads(row[4]) if row[4] else None,
        "created_at": str(row[5]),
        "updated_at": str(row[6]),
    }


_SELECT = """SELECT report_id, title, description, sections, snapshot, created_at, updated_at
             FROM report_definition"""


def save(
    payload: ReportInput,
    snapshot: list[SectionSnapshot] | None = None,
    report_id: str | None = None,
) -> dict[str, Any]:
    """新建或更新报告定义。

    更新时不传 snapshot 会**保留原有基线**，不是把它抹掉 —— 抹掉就再也答不出
    「和当初那份比，数字变了没有」。定义改了而基线还是旧的，正是需要被报出来的
    「分析定义变了」这种情况。要换基线得显式调 update_snapshot。
    """
    sections_json = json.dumps(
        _SECTIONS.dump_python(payload.sections, mode="json"), ensure_ascii=False
    )
    snapshot_json = (
        json.dumps(_SNAPSHOT.dump_python(snapshot, mode="json"), ensure_ascii=False)
        if snapshot is not None else None
    )
    now = _now()

    with store.write() as conn:
        if report_id:
            exists = conn.execute(
                "SELECT 1 FROM report_definition WHERE report_id = ?", [report_id]
            ).fetchone()
            if exists is None:
                raise KeyError(f"未知报告：{report_id}")
            if snapshot_json is None:
                conn.execute(
                    """UPDATE report_definition
                       SET title = ?, description = ?, sections = ?, updated_at = ?
                       WHERE report_id = ?""",
                    [payload.title, payload.description, sections_json, now, report_id],
                )
            else:
                conn.execute(
                    """UPDATE report_definition
                       SET title = ?, description = ?, sections = ?, snapshot = ?,
                           updated_at = ?
                       WHERE report_id = ?""",
                    [payload.title, payload.description, sections_json,
                     snapshot_json, now, report_id],
                )
        else:
            report_id = uuid.uuid4().hex[:12]
            conn.execute(
                """INSERT INTO report_definition
                   (report_id, title, description, sections, snapshot, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [report_id, payload.title, payload.description, sections_json,
                 snapshot_json, now, now],
            )
    return get(report_id)


def update_snapshot(report_id: str, snapshot: list[SectionSnapshot]) -> None:
    """只更新快照，不动报告定义。用于「以当前结果为准重新基线」。"""
    blob = json.dumps(_SNAPSHOT.dump_python(snapshot, mode="json"), ensure_ascii=False)
    with store.write() as conn:
        conn.execute(
            "UPDATE report_definition SET snapshot = ?, updated_at = ? WHERE report_id = ?",
            [blob, _now(), report_id],
        )


def get(report_id: str) -> dict[str, Any]:
    with store.read() as cur:
        row = cur.execute(f"{_SELECT} WHERE report_id = ?", [report_id]).fetchone()
    if row is None:
        raise KeyError(f"未知报告：{report_id}")
    return _row_to_dict(row)


def load_sections(report_id: str) -> list[ReportSection]:
    return _SECTIONS.validate_python(get(report_id)["sections"])


def load_snapshot(report_id: str) -> list[SectionSnapshot] | None:
    raw = get(report_id)["snapshot"]
    return _SNAPSHOT.validate_python(raw) if raw else None


def list_all() -> list[dict[str, Any]]:
    with store.read() as cur:
        return [_row_to_dict(r) for r in
                cur.execute(f"{_SELECT} ORDER BY updated_at DESC").fetchall()]


def delete(report_id: str) -> None:
    with store.write() as conn:
        conn.execute("DELETE FROM report_definition WHERE report_id = ?", [report_id])
