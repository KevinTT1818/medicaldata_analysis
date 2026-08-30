from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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


# --- 前端静态资源（可选）---
#
# 只在 MEDDATA_STATIC_DIR 指向一个存在的目录时挂载。开发时不设这个变量，
# 前端仍走 ng serve + proxy.conf.json，行为完全不变；容器里把 ng build 的
# 产物放进去，一个进程就能同时供 API 和界面，不用再拉一个 nginx。
_static = os.environ.get("MEDDATA_STATIC_DIR", "").strip()
STATIC_DIR = Path(_static) if _static else None

if STATIC_DIR and STATIC_DIR.is_dir():
    _index = STATIC_DIR / "index.html"

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """把非 API 路径交给前端路由。

        Angular 用的是 history 模式，/cohorts 这类地址在服务端并不存在文件，
        直接刷新会 404，所以命不中静态文件就一律回 index.html。
        """
        # 打错的 API 路径必须还是 404，不能被兜底成一页 HTML —— 否则
        # 客户端拿到 200 + <!doctype，会把「路由写错了」看成「返回了空数据」
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"没有这个接口：/{path}")

        candidate = (STATIC_DIR / path).resolve()
        # 防目录穿越：解析后必须仍在静态目录之内
        if path and STATIC_DIR.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_index)

    logging.getLogger("app").info("前端静态资源挂载自 %s", STATIC_DIR)
