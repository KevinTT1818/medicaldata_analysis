"""DuckDB 连接管理。

DuckDB 是单写入者模型，但 `conn.cursor()` 派生的连接共享同一个数据库文件，
可以安全地并发读。写入（ETL）走全局锁串行化。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import duckdb

from .config import WAREHOUSE

_conn: duckdb.DuckDBPyConnection | None = None
_write_lock = threading.Lock()


def _root() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(str(WAREHOUSE))
    return _conn


@contextmanager
def read():
    """并发安全的只读游标。"""
    cur = _root().cursor()
    try:
        yield cur
    finally:
        cur.close()


@contextmanager
def write():
    """串行化的写入连接。ETL 用。"""
    with _write_lock:
        yield _root()


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
