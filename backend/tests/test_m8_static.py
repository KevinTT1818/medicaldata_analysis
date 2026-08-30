"""前端静态资源托管。

只在 MEDDATA_STATIC_DIR 指向存在的目录时挂载 —— 开发时不设，前端仍走
ng serve + proxy，行为完全不变；容器里指向 ng build 的产物，一个进程
同时供 API 和界面，不用再拉 nginx。
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录


def make_client(static_dir: str | None):
    from fastapi.testclient import TestClient

    if static_dir:
        os.environ["MEDDATA_STATIC_DIR"] = static_dir
    else:
        os.environ.pop("MEDDATA_STATIC_DIR", None)
    for name in [m for m in sys.modules if m.startswith("app")]:
        del sys.modules[name]
    return TestClient(importlib.import_module("app.main").app)


class NoStaticDirTest(unittest.TestCase):
    """不设变量时不该多出任何路由。"""

    @classmethod
    def tearDownClass(cls):
        make_client(None)

    def test_root_is_not_served(self):
        with make_client(None) as client:
            self.assertEqual(client.get("/").status_code, 404)

    def test_api_still_works(self):
        with make_client(None) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)

    def test_missing_directory_is_ignored(self):
        with make_client("/nonexistent/path/xyz") as client:
            self.assertEqual(client.get("/").status_code, 404)
            self.assertEqual(client.get("/api/health").status_code, 200)


class StaticDirTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="meddata-static-")
        root = Path(cls.tmp)
        (root / "index.html").write_text("<!doctype html><title>前端</title>", encoding="utf-8")
        (root / "main.js").write_text("console.log(1)", encoding="utf-8")
        cls.client = make_client(cls.tmp)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        make_client(None)

    def test_index_is_served_at_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("前端", r.text)

    def test_real_files_are_served(self):
        r = self.client.get("/main.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn("console.log", r.text)

    def test_spa_routes_fall_back_to_index(self):
        """Angular 是 history 模式，/cohorts 在服务端没有对应文件，
        直接刷新不能 404。"""
        for path in ("/cohorts", "/workbench", "/reports", "/datasets"):
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 200)
                self.assertIn("前端", r.text)

    def test_api_is_not_shadowed_by_the_catch_all(self):
        """兜底路由不能把 API 吃掉。"""
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_unknown_api_path_does_not_return_html(self):
        r = self.client.get("/api/definitely-not-a-route")
        self.assertNotIn("<!doctype", r.text.lower())

    def test_path_traversal_is_refused(self):
        for path in ("/../../../../etc/passwd", "/..%2f..%2fetc%2fpasswd",
                     "/static/../../../etc/hosts"):
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertNotIn("root:", r.text)
                self.assertIn("前端", r.text)   # 回退到 index，不是泄露文件


if __name__ == "__main__":
    unittest.main()
