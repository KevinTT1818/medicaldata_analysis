"""测试环境。**每个测试模块都必须先导入本模块，再导入 app.***

DuckDB 是单写入者，测试要用独立的库文件才能和开发服务器并行跑，所以要覆盖
MEDDATA_DATA_DIR。但 app.config 在整个进程里只导入一次 —— 各测试文件各设各的
环境变量的话，只有第一个被导入的文件说了算，其余的 _TMP 指向的目录根本没被用到。
（这条曾让一个缓存测试单独跑通过、全量跑失败：它按自己的 _TMP 去找缓存文件，
自然一个也找不到。）

所以在这里设一次，全部测试共用，并把真实路径导出去让测试直接用。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDDATA_DATA_DIR", tempfile.mkdtemp(prefix="meddata-tests-"))
os.environ.setdefault("MEDDATA_RAW_DIR", str(ROOT / "data" / "raw"))

from app.config import CACHE_DIR, DATA_DIR, RAW_DIR  # noqa: E402

__all__ = ["ROOT", "DATA_DIR", "RAW_DIR", "CACHE_DIR"]
