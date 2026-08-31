"""路径与运行期配置。

数据目录可用环境变量 MEDDATA_DATA_DIR 覆盖 —— DuckDB 是单写入者，
跑测试时若和开发服务器共用同一个库文件会抢锁失败，测试因此指向独立目录。
原始文件目录单独用 MEDDATA_RAW_DIR 覆盖，好让测试复用已下载的数据。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MEDDATA_DATA_DIR") or BASE_DIR / "data")

RAW_DIR = Path(os.environ.get("MEDDATA_RAW_DIR") or DATA_DIR / "raw")
STAGING_DIR = DATA_DIR / "staging"  # 解析成 Parquet 的中间层
CDM_DIR = DATA_DIR / "cdm"          # 6 张标准表，按 dataset 分区
CACHE_DIR = DATA_DIR / "cache"      # 分析结果缓存

WAREHOUSE = DATA_DIR / "warehouse.duckdb"

for _d in (RAW_DIR, STAGING_DIR, CDM_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 判定连续/分类变量的阈值：唯一值个数 <= 此值且为整数型时按分类处理
CATEGORICAL_MAX_LEVELS = 10

#: 一次分析最多纳入多少个变量。不是功能限制 —— MIMIC 的 189 个变量「全选」
#: 是正常用法，这条线只是挡住畸形请求，免得一个请求就能撑爆内存。
MAX_VARIABLES_PER_ANALYSIS = 500

# 变量目录的最低覆盖率。MIMIC 有几百个 ICD 码，多数只出现在一两个人身上，
# 全列出来变量选择器就没法用了，而且只覆盖一两个人的变量也做不了统计。
# 被隐藏的数量会在 schema 响应里报出来。
MIN_VARIABLE_COVERAGE = 0.05
MIN_VARIABLE_PATIENTS = 2

#: API 密钥。设了就对 /api/* 强制校验（/api/health 除外）。
#: 留空时不校验，但启动会打一条醒目的告警 —— 只在本机单人用时才可以留空，
#: 任何能连上这个端口的人都能删掉全部队列与报告、重新导入数据集。
API_KEY = os.environ.get("MEDDATA_API_KEY", "").strip()

#: 内存任务表最多保留多少条。超出后淘汰最早结束的任务（运行中的绝不淘汰）。
#: 每条任务都挂着完整的结果 payload —— KM 曲线的 t / survival 数组可达数千点，
#: 只增不减的话长期运行内存只会往上走。结果本来就有落盘缓存，
#: 丢掉内存副本不影响可复现。
MAX_RETAINED_JOBS = int(os.environ.get("MEDDATA_MAX_JOBS", "200"))

#: 日志级别。
LOG_LEVEL = os.environ.get("MEDDATA_LOG_LEVEL", "INFO").upper()
