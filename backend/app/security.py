"""API 访问控制。

审计发现整套 API 一个 Depends 都没有：任何能连上这个端口的人都可以删掉
全部报告与队列、重新导入数据集、读走所有结果。

这里是最简单的一层：设了 MEDDATA_API_KEY 就校验 X-API-Key 请求头，
没设就放行但在启动时打一条醒目的告警。

为什么不默认强制：这套东西现在的用法是本机单人跑 uvicorn + ng serve，
默认要密钥会让开箱即用的流程直接断掉，而本机场景下它也挡不住什么。
只要要给第二个人用、或者绑到 0.0.0.0 之外，就必须设上这个变量。
"""
from __future__ import annotations

import logging
import secrets

from fastapi import Header, HTTPException, status

from .config import API_KEY

log = logging.getLogger(__name__)

HEADER_NAME = "X-API-Key"


def enabled() -> bool:
    return bool(API_KEY)


def require_api_key(x_api_key: str | None = Header(None, alias=HEADER_NAME)) -> None:
    """校验请求头里的密钥。未配置密钥时直接放行。"""
    if not API_KEY:
        return
    # compare_digest：避免按字符比较带来的计时差异泄露密钥前缀
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=f"缺少或错误的 {HEADER_NAME}",
            headers={"WWW-Authenticate": HEADER_NAME},
        )


def warn_if_open() -> None:
    """启动时提醒当前是无保护状态。"""
    if API_KEY:
        log.info("API 密钥校验已启用（请求头 %s）", HEADER_NAME)
        return
    log.warning(
        "未设置 MEDDATA_API_KEY —— 任何能连上本端口的人都可以删除队列与报告、"
        "重新导入数据集。仅在本机单人使用时可以这样；要给别人用请设置该环境变量。"
    )
