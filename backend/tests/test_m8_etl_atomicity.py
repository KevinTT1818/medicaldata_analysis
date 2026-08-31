"""重新导入必须是原子的。

审计时发现：`import_dataset` 先 DELETE 再 INSERT，两步各自提交，中间这段
时间里并发的分析会读到一张空表，然后**正常返回**一个基于空数据的结果 ——
不报错、不告警，只是数字是错的。以 2ms 间隔轮询复现时，读到的人数取值是
[0, 9254]。

这里的测试守住修复：清空与写入并入同一个事务，靠 DuckDB 的快照隔离让并发
读者在提交前只看到旧数据。
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

from app import store
from app.cdm import etl, schema
from app.adapters import uci_heart  # noqa: F401  触发注册
from app.config import CDM_DIR

DS = "uci_heart"
N_PERSON = 303


def person_count() -> int:
    with store.read() as conn:
        return conn.execute(
            "SELECT count(*) FROM person WHERE dataset = ?", [DS]
        ).fetchone()[0]


class ReimportIsAtomicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)

    def test_concurrent_reader_never_sees_partial_data(self):
        """重新导入期间，并发的读只能看到完整的旧数据或完整的新数据。"""
        seen: list[int] = []
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                seen.append(person_count())
                time.sleep(0.0005)

        reader = threading.Thread(target=poll, daemon=True)
        reader.start()
        try:
            time.sleep(0.01)
            for _ in range(3):
                etl.import_dataset(DS)
            time.sleep(0.01)
        finally:
            stop.set()
            reader.join(timeout=5)

        self.assertGreater(len(seen), 20, "轮询次数太少，测试没有真正覆盖导入窗口")
        self.assertEqual(
            sorted(set(seen)), [N_PERSON],
            f"并发读看到了不完整的数据：{sorted(set(seen))}",
        )

    def test_failed_import_leaves_old_data_intact(self):
        """事务中途失败必须整体回滚，旧数据一行不少。"""
        real_clear = schema.clear_dataset

        def clear_then_fail(dataset, conn=None):
            real_clear(dataset, conn=conn)          # 删除确实发生了
            raise RuntimeError("注入的故障：清空之后、写入之前")

        with mock.patch.object(schema, "clear_dataset", clear_then_fail):
            with self.assertRaises(RuntimeError):
                etl.import_dataset(DS)

        self.assertEqual(person_count(), N_PERSON, "失败的导入把旧数据弄丢了")

    def test_registry_stays_consistent_with_person_table(self):
        """登记表里的人数与实际行数必须一致——两者由同一个事务写入。"""
        etl.import_dataset(DS)
        with store.read() as conn:
            registered = conn.execute(
                "SELECT n_person FROM dataset_registry WHERE dataset = ?", [DS]
            ).fetchone()[0]
        self.assertEqual(registered, person_count())


class StagingFilesTest(unittest.TestCase):
    def test_successful_import_leaves_no_staging_files(self):
        etl.import_dataset(DS)
        leftovers = list((CDM_DIR / DS).glob(f"*{etl.STAGING_SUFFIX}"))
        self.assertEqual(leftovers, [], f"导入成功后残留了临时文件：{leftovers}")

    def test_failed_import_leaves_no_staging_files(self):
        def boom(dataset, conn=None):
            raise RuntimeError("注入的故障")

        with mock.patch.object(schema, "clear_dataset", boom):
            with self.assertRaises(RuntimeError):
                etl.import_dataset(DS)

        leftovers = list((CDM_DIR / DS).glob(f"*{etl.STAGING_SUFFIX}"))
        self.assertEqual(leftovers, [], f"导入失败后残留了临时文件：{leftovers}")

    def test_parquet_files_are_in_place_after_import(self):
        etl.import_dataset(DS)
        for name in schema.TABLES:
            self.assertTrue(
                (CDM_DIR / DS / f"{name}.parquet").exists(),
                f"{name}.parquet 没有改名到正式位置",
            )


class TransactionPrimitiveTest(unittest.TestCase):
    """store.transaction() 本身的语义。"""

    def setUp(self):
        with store.write() as conn:
            conn.execute("CREATE OR REPLACE TABLE _tx_probe (v INTEGER)")
            conn.execute("INSERT INTO _tx_probe VALUES (1)")

    def tearDown(self):
        with store.write() as conn:
            conn.execute("DROP TABLE IF EXISTS _tx_probe")

    def test_rollback_on_exception(self):
        with self.assertRaises(RuntimeError):
            with store.transaction() as conn:
                conn.execute("INSERT INTO _tx_probe VALUES (2)")
                raise RuntimeError("boom")

        with store.read() as conn:
            rows = conn.execute("SELECT count(*) FROM _tx_probe").fetchone()[0]
        self.assertEqual(rows, 1, "异常之后写入没有回滚")

    def test_commit_on_success(self):
        with store.transaction() as conn:
            conn.execute("INSERT INTO _tx_probe VALUES (3)")

        with store.read() as conn:
            rows = conn.execute("SELECT count(*) FROM _tx_probe").fetchone()[0]
        self.assertEqual(rows, 2)

    def test_nested_write_does_not_deadlock(self):
        """transaction() 里再取 write()（同线程）必须能拿到锁。"""
        with store.transaction() as outer:
            outer.execute("INSERT INTO _tx_probe VALUES (4)")
            with store.write() as inner:
                self.assertIs(inner, outer)


if __name__ == "__main__":
    unittest.main()
