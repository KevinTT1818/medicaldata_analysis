"""DuckDB 连接管理。

DuckDB 是单写入者模型，但 `conn.cursor()` 派生的连接共享同一个数据库文件，
可以安全地并发读。写入（ETL）走全局锁串行化。

`read()` 与 `write()` 用的是不同的连接对象，因此各有独立的事务上下文，
DuckDB 的快照隔离在两者之间生效：写入方在 `transaction()` 里做的删改，
在 COMMIT 之前对并发的读完全不可见。重新导入数据集必须依赖这一点，
否则并发的分析会读到被清空的表并"正常"返回一个基于空数据的结果。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import duckdb

from .config import WAREHOUSE

_conn: duckdb.DuckDBPyConnection | None = None
# 可重入：transaction() 内部再调用 write() 的路径不该死锁。
_write_lock = threading.RLock()


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
    """串行化的写入连接。单条语句自动提交。"""
    with _write_lock:
        yield _root()


@contextmanager
def transaction():
    """串行化 + 原子的写入连接。

    块内的所有写要么全部生效、要么全部不生效，中途对并发读者始终不可见。
    任何异常都会回滚并继续向上抛。
    """
    with _write_lock:
        conn = _root()
        conn.execute("BEGIN TRANSACTION")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
