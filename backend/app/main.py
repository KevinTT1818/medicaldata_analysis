from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import logging_setup, security
from .api import analyses, cohorts, datasets, jobs, reports
from .cdm import schema

log = logging.getLogger("app.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_setup.configure()
    schema.init()
    security.warn_if_open()
    yield


app = FastAPI(
    title="医疗数据分析平台",
    description="基于公开医疗数据集的分析系统。统一数据模型 + 可插拔分析算子。",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_methods=["*"],
    # 浏览器要能带上密钥请求头，预检才会放行
    allow_headers=["*", security.HEADER_NAME],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每次请求的方法、路径、状态与耗时。慢请求单独抬一级。"""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started

    line = "%s %s -> %d  %.0fms"
    args = (request.method, request.url.path, response.status_code, elapsed * 1000)
    if elapsed >= logging_setup.SLOW_REQUEST_SECONDS:
        log.warning(line + "  （慢）", *args)
    elif response.status_code >= 500:
        log.error(line, *args)
    elif response.status_code >= 400:
        log.warning(line, *args)
    else:
        log.info(line, *args)
    return response


# 所有业务路由都要过密钥校验；/api/health 故意留在外面，
# 好让前端在没有密钥时也能问出「这台服务器要不要密钥」。
_guard = [Depends(security.require_api_key)]

app.include_router(datasets.router, dependencies=_guard)
app.include_router(cohorts.router, dependencies=_guard)
app.include_router(analyses.router, dependencies=_guard)
app.include_router(jobs.router, dependencies=_guard)
app.include_router(reports.router, dependencies=_guard)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "auth_required": security.enabled()}
