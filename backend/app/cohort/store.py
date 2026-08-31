"""队列定义的持久化。

存的是条件树本身，不是命中的人员名单 —— 数据集重新导入后队列定义依然有效，
这也是分析可复现的前提：保存队列定义 + 参数 + 数据集版本，就能一键重跑。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import TypeAdapter

from .. import store
from .filters import Node

_NODE = TypeAdapter(Node)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "id": row[0],
        "dataset": row[1],
        "name": row[2],
        "description": row[3],
        "definition": json.loads(row[4]),
        "created_at": str(row[5]),
        "updated_at": str(row[6]),
    }


def save(dataset: str, name: str, definition: Node,
         description: str | None = None, cohort_id: str | None = None) -> dict[str, Any]:
    """新建或覆盖一个队列定义。"""
    payload = json.dumps(_NODE.dump_python(definition, mode="json"), ensure_ascii=False)
    now = _now()

    with store.write() as conn:
        if cohort_id:
            existing = conn.execute(
                "SELECT created_at FROM cohort_definition WHERE cohort_id = ?", [cohort_id]
            ).fetchone()
            if existing is None:
                raise KeyError(f"未知队列：{cohort_id}")
            conn.execute(
                """UPDATE cohort_definition
                   SET dataset = ?, name = ?, description = ?, definition = ?, updated_at = ?
                   WHERE cohort_id = ?""",
                [dataset, name, description, payload, now, cohort_id],
            )
        else:
            cohort_id = uuid.uuid4().hex[:12]
            conn.execute(
                """INSERT INTO cohort_definition
                   (cohort_id, dataset, name, description, definition, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [cohort_id, dataset, name, description, payload, now, now],
            )

    return get(cohort_id)


def get(cohort_id: str) -> dict[str, Any]:
    with store.read() as cur:
        row = cur.execute(
            """SELECT cohort_id, dataset, name, description, definition, created_at, updated_at
               FROM cohort_definition WHERE cohort_id = ?""", [cohort_id]
        ).fetchone()
    if row is None:
        raise KeyError(f"未知队列：{cohort_id}")
    return _row_to_dict(row)


def load_node(cohort_id: str) -> Node:
    return _NODE.validate_python(get(cohort_id)["definition"])


def list_for(dataset: str | None = None) -> list[dict[str, Any]]:
    sql = ("""SELECT cohort_id, dataset, name, description, definition, created_at, updated_at
              FROM cohort_definition""")
    params: list[Any] = []
    if dataset:
        sql += " WHERE dataset = ?"
        params.append(dataset)
    sql += " ORDER BY updated_at DESC"

    with store.read() as cur:
        return [_row_to_dict(r) for r in cur.execute(sql, params).fetchall()]


def delete(cohort_id: str) -> None:
    with store.write() as conn:
        conn.execute("DELETE FROM cohort_definition WHERE cohort_id = ?", [cohort_id])
