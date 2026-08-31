"""ETL：adapter 产出 -> Parquet + DuckDB。"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .. import store
from ..adapters import base as adapters
from ..config import CDM_DIR
from . import schema

Progress = Callable[[str, float], None]

#: 落盘前的临时后缀。写完并且数据库事务提交之后才改名到正式位置，
#: 保证磁盘上的 Parquet 与数据库内容不会各说各话。
STAGING_SUFFIX = ".incoming"


def import_dataset(dataset_id: str, progress: Progress | None = None) -> dict:
    """把一个数据集导入 CDM。

    幂等：重复导入会先清空该数据集的旧数据。整个替换过程是原子的——
    清空与写入在同一个事务里，并发的分析在提交之前只会看到旧数据，
    绝不会看到空表或写了一半的表。失败时数据库与磁盘都保持导入前的状态。
    """
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

    out_dir = CDM_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    staged: list[tuple[Path, Path]] = []   # (临时文件, 正式位置)

    try:
        report("写入 Parquet", 0.3)
        for i, name in enumerate(schema.TABLES):
            df = tables[name]
            report(f"写入 {name}", 0.3 + 0.4 * i / len(schema.TABLES))

            final = out_dir / f"{name}.parquet"
            tmp = final.with_suffix(final.suffix + STAGING_SUFFIX)
            df.write_parquet(tmp)
            staged.append((tmp, final))
            counts[name] = df.height

        # 清空 + 灌入 + 登记必须在同一个事务里。DuckDB 的快照隔离让并发的
        # 读者在 COMMIT 之前始终看到旧数据；拆成多个自动提交的语句会让
        # 这期间的分析基于空表算出一个看起来正常、实际错误的结果。
        report("导入数据库", 0.7)
        with store.transaction() as conn:
            schema.clear_dataset(dataset_id, conn=conn)

            for i, name in enumerate(schema.TABLES):
                report(f"导入 {name}", 0.7 + 0.25 * i / len(schema.TABLES))
                if not counts[name]:
                    continue
                tmp = out_dir / f"{name}.parquet{STAGING_SUFFIX}"
                # 直接从 Parquet 灌入，不经内存注册：省掉 pyarrow 依赖，
                # 且 Parquet 文件本身就是可复现的落盘产物。
                # BY NAME 按列名对齐，不依赖 DataFrame 的列顺序。
                conn.execute(
                    f"INSERT INTO {name} BY NAME "
                    f"SELECT * FROM read_parquet(?)", [str(tmp)]
                )

            conn.execute(
                """INSERT INTO dataset_registry
                   (dataset, label, description, source_url, version, n_person, imported_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [adapter.dataset_id, adapter.label, adapter.description,
                 adapter.source_url, adapter.version, counts["person"],
                 datetime.now(timezone.utc)],
            )

        # 事务已提交，数据库不再依赖这些文件的路径，可以安全改名。
        report("落盘", 0.97)
        for tmp, final in staged:
            os.replace(tmp, final)
        staged.clear()
    finally:
        # 任何失败路径都不留下 .incoming 垃圾。已改名的不在列表里。
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)

    report("完成", 1.0)
    return {"dataset": dataset_id, "counts": counts}
