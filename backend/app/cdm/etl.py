"""ETL：adapter 产出 -> Parquet + DuckDB。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from .. import store
from ..adapters import base as adapters
from ..config import CDM_DIR
from . import schema

Progress = Callable[[str, float], None]


def import_dataset(dataset_id: str, progress: Progress | None = None) -> dict:
    """把一个数据集导入 CDM。幂等：重复导入会先清空该数据集的旧数据。"""
    def report(stage: str, pct: float) -> None:
        if progress:
            progress(stage, pct)

    adapter = adapters.get(dataset_id)
    if not adapter.is_available():
        raise FileNotFoundError(
            f"{adapter.label} 的原始文件不存在，请先下载到 data/raw/{dataset_id}/"
        )

    schema.init()

    report("解析原始文件", 0.1)
    tables = adapter.build()

    report("清空旧数据", 0.3)
    schema.clear_dataset(dataset_id)

    out_dir = CDM_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with store.write() as conn:
        for i, name in enumerate(schema.TABLES):
            df = tables[name]
            report(f"写入 {name}", 0.35 + 0.55 * i / len(schema.TABLES))

            path = out_dir / f"{name}.parquet"
            df.write_parquet(path)
            counts[name] = df.height

            if df.height:
                # 直接从 Parquet 灌入，不经内存注册：省掉 pyarrow 依赖，
                # 且 Parquet 文件本身就是可复现的落盘产物。
                # BY NAME 按列名对齐，不依赖 DataFrame 的列顺序。
                conn.execute(
                    f"INSERT INTO {name} BY NAME "
                    f"SELECT * FROM read_parquet(?)", [str(path)]
                )

        conn.execute(
            """INSERT INTO dataset_registry
               (dataset, label, description, source_url, version, n_person, imported_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [adapter.dataset_id, adapter.label, adapter.description,
             adapter.source_url, adapter.version, counts["person"],
             datetime.now(timezone.utc)],
        )

    report("完成", 1.0)
    return {"dataset": dataset_id, "counts": counts}
