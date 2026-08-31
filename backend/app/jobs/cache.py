"""分析结果缓存。

缓存键 = 数据集 + 数据集版本 + 算子 + 参数 + 队列定义。
数据集版本取导入时间戳 —— 重新导入数据后旧结果自动失效，
不会拿着过期结论继续用。

落盘而不是放内存：重启后仍然命中，且缓存文件本身就是「这次分析算过什么」的记录。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from datetime import datetime, timezone
from typing import Any

from .. import store
from ..config import CACHE_DIR


def _json_default(obj: Any) -> Any:
    """兜底处理 numpy 标量。算子应当自己转成 Python 类型，
    这里只是防止漏网的一个把整份结果写崩。"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"无法序列化 {type(obj).__name__}")


def dataset_version(dataset: str) -> str:
    """数据集的版本标记。取导入时间戳；没导入过就返回空串。"""
    with store.read() as cur:
        row = cur.execute(
            "SELECT imported_at FROM dataset_registry WHERE dataset = ?", [dataset]
        ).fetchone()
    return str(row[0]) if row else ""


def key_for(spec: dict[str, Any], version: str) -> str:
    payload = {"spec": {k: v for k, v in spec.items() if k != "fingerprint"},
               "dataset_version": version}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def get(cache_key: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 缓存坏了就当没有，重新算一遍即可
        path.unlink(missing_ok=True)
        return None


def put(cache_key: str, payload: dict[str, Any], warnings: list[str],
        cohort_n: int | None, cohort_total: int | None) -> None:
    record = {
        "payload": payload,
        "warnings": warnings,
        "cohort_n": cohort_n,
        "cohort_total": cohort_total,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path = CACHE_DIR / f"{cache_key}.json"
    try:
        path.write_text(json.dumps(record, ensure_ascii=False, default=_json_default),
                        encoding="utf-8")
    except OSError:
        pass  # 缓存写不进去不该让分析失败


def clear(dataset: str | None = None) -> int:
    """清空缓存。传 dataset 时也只能全清 —— 键是哈希，反查不了归属。"""
    removed = 0
    for path in CACHE_DIR.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed
