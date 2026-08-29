from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import analyses, cohorts, datasets, jobs, reports
from .cdm import schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    schema.init()
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
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(cohorts.router)
app.include_router(analyses.router)
app.include_router(jobs.router)
app.include_router(reports.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
