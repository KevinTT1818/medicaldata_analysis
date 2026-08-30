"""API 访问控制、任务表淘汰、日志。

审计发现：6 个 API 模块里 Depends 出现次数全是 0 —— 任何能连上这个端口
的人都可以删掉全部报告与队列、重新导入数据集（后者还会触发脏读）。
任务表只增不减，60 次运行驻留 60 条，每条挂着完整的结果 payload。
日志设施只有一个 traceback.print_exc()，出了事对不上是哪次分析。
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

from app import security
from app.cdm import schema
from app.config import MAX_RETAINED_JOBS
from app.jobs import runner

PROTECTED = (
    ("get", "/api/analyses/registry"),
    ("get", "/api/cohorts"),
    ("get", "/api/cohorts/operators"),
    ("get", "/api/reports"),
    ("get", "/api/jobs"),
    ("get", "/api/datasets"),
    ("delete", "/api/cohorts/anything"),
    ("delete", "/api/reports/anything"),
    ("post", "/api/datasets/uci_heart/import"),
)

KEY = "test-key-9f3a"


def make_client(api_key: str | None):
    """按给定密钥重建应用。config 在导入时读环境变量，所以要重载。"""
    from fastapi.testclient import TestClient

    if api_key:
        os.environ["MEDDATA_API_KEY"] = api_key
    else:
        os.environ.pop("MEDDATA_API_KEY", None)

    for name in [m for m in sys.modules if m.startswith("app")]:
        del sys.modules[name]
    main = importlib.import_module("app.main")
    return TestClient(main.app)


class OpenModeTest(unittest.TestCase):
    """没设密钥时开箱行为不变 —— 本机单人跑 uvicorn + ng serve 不该被打断。"""

    @classmethod
    def setUpClass(cls):
        cls.client = make_client(None)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        make_client(None)   # 复位模块状态，免得影响后面的测试文件

    def test_health_reports_no_auth_required(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertFalse(body["auth_required"])

    def test_all_endpoints_reachable(self):
        for method, path in PROTECTED:
            with self.subTest(path=path):
                self.assertNotEqual(getattr(self.client, method)(path).status_code, 401)


class GuardedModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = make_client(KEY)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        make_client(None)

    def test_health_stays_open_and_advertises_auth(self):
        """健康检查故意不设防：前端要能在没密钥时问出「这台服务器要不要密钥」。"""
        body = self.client.get("/api/health").json()
        self.assertTrue(body["auth_required"])

    def test_every_endpoint_rejects_missing_key(self):
        for method, path in PROTECTED:
            with self.subTest(path=path):
                self.assertEqual(getattr(self.client, method)(path).status_code, 401)

    def test_every_endpoint_rejects_wrong_key(self):
        for method, path in PROTECTED:
            with self.subTest(path=path):
                r = getattr(self.client, method)(path, headers={"X-API-Key": "nope"})
                self.assertEqual(r.status_code, 401)

    def test_correct_key_passes(self):
        for method, path in PROTECTED:
            with self.subTest(path=path):
                r = getattr(self.client, method)(path, headers={"X-API-Key": KEY})
                self.assertNotEqual(r.status_code, 401)

    def test_rejection_names_the_header(self):
        r = self.client.get("/api/cohorts")
        self.assertIn("X-API-Key", r.json()["detail"])

    def test_prefix_of_the_key_is_not_accepted(self):
        r = self.client.get("/api/cohorts", headers={"X-API-Key": KEY[:-1]})
        self.assertEqual(r.status_code, 401)


class ApiKeyComparisonTest(unittest.TestCase):
    """依赖本身的语义，不经 HTTP。"""

    def test_no_key_configured_lets_everything_through(self):
        self.assertIsNone(security.require_api_key(None))

    def test_configured_key_requires_exact_match(self):
        from fastapi import HTTPException

        original = security.API_KEY
        security.API_KEY = "abc123"
        try:
            self.assertIsNone(security.require_api_key("abc123"))
            for bad in (None, "", "abc12", "abc1234", "ABC123"):
                with self.subTest(value=bad), self.assertRaises(HTTPException) as cm:
                    security.require_api_key(bad)
                self.assertEqual(cm.exception.status_code, 401)
        finally:
            security.API_KEY = original


class JobEvictionTest(unittest.TestCase):
    """任务表不能只增不减。"""

    def setUp(self):
        schema.init()
        with runner._LOCK:
            runner._JOBS.clear()

    def tearDown(self):
        with runner._LOCK:
            runner._JOBS.clear()

    def _finished(self, n: int) -> None:
        for i in range(n):
            job = runner.Job(id=f"j{i:05d}", kind="analysis", spec={},
                             created_at=f"2026-01-01T00:00:{i % 60:02d}",
                             finished_at=f"2026-01-01T00:00:{i % 60:02d}",
                             status="succeeded")
            runner._JOBS[job.id] = job

    def test_stays_under_the_cap(self):
        self._finished(MAX_RETAINED_JOBS + 50)
        with runner._LOCK:
            runner._evict_locked()
        self.assertEqual(runner.job_count(), MAX_RETAINED_JOBS)

    def test_running_jobs_are_never_evicted(self):
        """运行中的任务被丢掉，前端的进度流就查不到自己了。"""
        for i in range(MAX_RETAINED_JOBS + 20):
            runner._JOBS[f"live{i}"] = runner.Job(
                id=f"live{i}", kind="analysis", spec={},
                created_at="2026-01-01T00:00:00", status="running")
        with runner._LOCK:
            runner._evict_locked()
        self.assertEqual(runner.job_count(), MAX_RETAINED_JOBS + 20)
        self.assertTrue(all(j.status == "running" for j in runner._JOBS.values()))

    def test_evicts_oldest_first(self):
        self._finished(MAX_RETAINED_JOBS)
        newest = runner.Job(id="newest", kind="analysis", spec={},
                            created_at="2026-12-31T23:59:59",
                            finished_at="2026-12-31T23:59:59", status="succeeded")
        runner._JOBS["newest"] = newest
        with runner._LOCK:
            runner._evict_locked()
        self.assertIn("newest", runner._JOBS)
        self.assertNotIn("j00000", runner._JOBS)

    def test_no_eviction_below_the_cap(self):
        self._finished(10)
        with runner._LOCK:
            self.assertEqual(runner._evict_locked(), 0)
        self.assertEqual(runner.job_count(), 10)

    def test_submitting_triggers_eviction(self):
        self._finished(MAX_RETAINED_JOBS + 5)
        runner.submit_import("uci_heart")
        self.assertLessEqual(runner.job_count(), MAX_RETAINED_JOBS + 1)


class JobLoggingTest(unittest.TestCase):
    """失败要带任务号进日志，否则事后对不上是哪一次分析。"""

    def setUp(self):
        schema.init()

    def test_failure_is_logged_with_job_id_and_stack(self):
        with self.assertLogs("app.jobs", level="ERROR") as captured:
            job = runner.wait(runner.submit_analysis(
                "uci_heart", "describe.baseline_table",
                {"variables": ["person.does_not_exist"]}, None, None, "first"))
        self.assertEqual(job.status, "failed")
        blob = "\n".join(captured.output)
        self.assertIn(job.id, blob)
        self.assertIn("Traceback", blob)

    def test_submission_is_logged(self):
        with self.assertLogs("app.jobs", level="INFO") as captured:
            job = runner.submit_import("uci_heart")
        self.assertTrue(any(job.id in line for line in captured.output))


class LoggingSetupTest(unittest.TestCase):
    def test_configure_is_idempotent(self):
        from app import logging_setup

        logging_setup.configure()
        before = len(logging.getLogger("app").handlers)
        logging_setup.configure()
        self.assertEqual(len(logging.getLogger("app").handlers), before)

    def test_no_bare_print_left_in_app(self):
        """print 打到 stdout、没有级别也没有时间戳，等于没有日志。

        用 AST 而不是文本匹配 —— 注释和文档字符串里提到 print 是正常的。
        """
        import ast
        import pathlib

        offenders = []
        for path in pathlib.Path(_ROOT / "app").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "print":
                    offenders.append(f"{path.name}:{node.lineno} print()")
                elif (isinstance(fn, ast.Attribute) and fn.attr == "print_exc"):
                    offenders.append(f"{path.name}:{node.lineno} print_exc()")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
