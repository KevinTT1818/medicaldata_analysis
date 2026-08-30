"""日志。

审计发现全部代码里的日志设施只有一个 traceback.print_exc() 和一个 print()：
没有请求日志、没有慢查询记录、没有错误上下文。出了问题事后完全无从追查
是哪次分析、读到了什么。

这里配一套标准 logging，并把任务的提交、耗时、失败原因记下来。
"""
from __future__ import annotations

import logging
import sys

from .config import LOG_LEVEL

#: 单次请求超过这个秒数就单独记一条 —— 慢查询要能事后翻出来
SLOW_REQUEST_SECONDS = 2.0

_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger("app")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
