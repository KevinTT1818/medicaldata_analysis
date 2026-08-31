# 一个镜像同时供 API 与界面。
#
# 前端先在 node 阶段构建，产物交给后端用 StaticFiles 托管 —— 单人用的研究
# 工具没必要为了发静态文件再拉一个 nginx 进来。

# --- 阶段 1：构建前端 ---
FROM node:20-slim AS web

WORKDIR /src
# 先只拷依赖清单，依赖没变时这一层能命中缓存
COPY package.json package-lock.json ./
RUN npm ci

COPY angular.json tsconfig*.json ./
COPY src ./src
RUN npx ng build --configuration production

# --- 阶段 2：运行 ---
FROM python:3.12-slim

# lifelines / statsmodels 装的是 wheel，不需要编译器；
# curl 只为 HEALTHCHECK 留着
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=web /src/dist/medicaldata_analysis/browser ./static

# 原始数据与 DuckDB 库都落在这里，用卷挂出来 —— 镜像里不带数据
ENV MEDDATA_DATA_DIR=/data \
    MEDDATA_RAW_DIR=/data/raw \
    MEDDATA_STATIC_DIR=/app/static \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]

# 不以 root 跑
RUN useradd --create-home --uid 10001 meddata \
 && mkdir -p /data && chown -R meddata:meddata /data /app
USER meddata

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# DuckDB 是单写入者，多 worker 会抢锁 —— 只能单进程
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
