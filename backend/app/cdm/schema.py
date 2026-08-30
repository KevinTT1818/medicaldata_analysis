"""统一数据模型（简化版 OMOP CDM）。

6 张表是全系统唯一的契约：adapter 往这里写，分析算子从这里读。
所有表都带 `dataset` 列，多个数据集共存于同一份表中，查询时按 dataset 过滤。
"""
from __future__ import annotations

import re

from .. import store

TABLES = ("person", "visit", "condition", "measurement", "drug", "outcome")

DDL = """
CREATE TABLE IF NOT EXISTS person (
    dataset       VARCHAR NOT NULL,
    person_id     VARCHAR NOT NULL,
    gender        VARCHAR,          -- 'M' | 'F' | NULL
    birth_year    INTEGER,
    age_at_index  DOUBLE,           -- 横断面数据集只给年龄不给出生年，两列并存
    race          VARCHAR,
    ethnicity     VARCHAR,
    -- 复杂抽样设计。NHANES 这类调查按不同概率抽样，直接求均值得到的是样本均值
    -- 而不是人群估计，必须加权。没有抽样设计的数据集这三列留空。
    sample_weight DOUBLE,
    psu           VARCHAR,          -- 初级抽样单元
    stratum       VARCHAR           -- 分层
);

CREATE TABLE IF NOT EXISTS visit (
    dataset          VARCHAR NOT NULL,
    visit_id         VARCHAR NOT NULL,
    person_id        VARCHAR NOT NULL,
    start_ts         TIMESTAMP,
    end_ts           TIMESTAMP,
    visit_type       VARCHAR,       -- outpatient | inpatient | icu | emergency
    discharge_status VARCHAR
);

CREATE TABLE IF NOT EXISTS condition (
    dataset      VARCHAR NOT NULL,
    condition_id VARCHAR NOT NULL,
    person_id    VARCHAR NOT NULL,
    visit_id     VARCHAR,
    code         VARCHAR NOT NULL,
    code_system  VARCHAR,           -- ICD-9-CM | ICD-10 | UCI | ...
    code_text    VARCHAR,
    onset_ts     TIMESTAMP,
    is_primary   BOOLEAN
);

CREATE TABLE IF NOT EXISTS measurement (
    dataset        VARCHAR NOT NULL,
    measurement_id VARCHAR NOT NULL,
    person_id      VARCHAR NOT NULL,
    visit_id       VARCHAR,
    code           VARCHAR NOT NULL,
    code_system    VARCHAR,         -- LOINC | UCI | ...
    code_text      VARCHAR,
    value_num      DOUBLE,          -- 归一化后的数值
    value_text     VARCHAR,         -- 分类型结果（如 thal = '可逆缺损'）
    unit           VARCHAR,         -- 归一化后的单位
    value_raw      DOUBLE,          -- 数据集原始值，用于追溯
    unit_raw       VARCHAR,         -- 数据集原始单位
    ts             TIMESTAMP,
    ref_low        DOUBLE,
    ref_high       DOUBLE
);

CREATE TABLE IF NOT EXISTS drug (
    dataset     VARCHAR NOT NULL,
    drug_id     VARCHAR NOT NULL,
    person_id   VARCHAR NOT NULL,
    visit_id    VARCHAR,
    code        VARCHAR,
    code_system VARCHAR,            -- RxNorm | ATC | ...
    name        VARCHAR,
    start_ts    TIMESTAMP,
    end_ts      TIMESTAMP,
    dose        DOUBLE,
    dose_unit   VARCHAR,
    route       VARCHAR
);

CREATE TABLE IF NOT EXISTS outcome (
    dataset       VARCHAR NOT NULL,
    person_id     VARCHAR NOT NULL,
    event_type    VARCHAR NOT NULL, -- death | readmission | heart_disease | ...
    event_ts      TIMESTAMP,
    censor_ts     TIMESTAMP,
    -- 生存分析要的是「随访时长 + 是否删失」这对语义。有真实日期的数据集
    -- （MIMIC）填 event_ts/censor_ts，只给随访天数的（多数公开队列）直接填这列。
    followup_days DOUBLE,
    is_event      BOOLEAN
);

-- 队列定义（元数据，不属于 CDM 本身）
CREATE TABLE IF NOT EXISTS cohort_definition (
    cohort_id   VARCHAR NOT NULL,
    dataset     VARCHAR NOT NULL,
    name        VARCHAR NOT NULL,
    description VARCHAR,
    definition  VARCHAR NOT NULL,  -- 条件树的 JSON
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP
);

-- 报告定义（元数据，不属于 CDM 本身）
CREATE TABLE IF NOT EXISTS report_definition (
    report_id   VARCHAR NOT NULL,
    title       VARCHAR NOT NULL,
    description VARCHAR,
    sections    VARCHAR NOT NULL,  -- 各小节的分析定义，JSON
    snapshot    VARCHAR,           -- 保存时各小节的结果指纹，重跑时用来比对
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP
);

-- 数据集注册表（元数据，不属于 CDM 本身）
CREATE TABLE IF NOT EXISTS dataset_registry (
    dataset      VARCHAR PRIMARY KEY,
    label        VARCHAR,
    description  VARCHAR,
    source_url   VARCHAR,
    version      VARCHAR,
    n_person     INTEGER,
    imported_at  TIMESTAMP
);
"""


#: 从 DDL 里抽列定义用。DDL 是我们自己写的受控格式，正则足够可靠。
_TABLE_RE = re.compile(
    r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", re.DOTALL
)
_COLUMN_RE = re.compile(r"^\s*(\w+)\s+([A-Z][A-Z0-9_ ]*)")


def _declared_columns() -> dict[str, list[tuple[str, str]]]:
    """解析 DDL，得到 {表名: [(列名, 类型), ...]}。"""
    out: dict[str, list[tuple[str, str]]] = {}
    for table, body in _TABLE_RE.findall(DDL):
        columns: list[tuple[str, str]] = []
        for line in body.splitlines():
            line = line.split("--")[0].strip().rstrip(",")
            if not line or line.upper().startswith(("PRIMARY KEY", "UNIQUE", "CHECK")):
                continue
            m = _COLUMN_RE.match(line)
            if m:
                columns.append((m.group(1), m.group(2).strip()))
        out[table] = columns
    return out


def migrate() -> list[str]:
    """给已存在的表补上 DDL 里新增的列。

    CREATE TABLE IF NOT EXISTS 对已存在的表是空操作，CDM 一旦演进
    （比如 outcome 加 followup_days）老库就会在插入时报错。
    这里做一次列级对账，只加不删 —— 删列会丢数据，交给人决定。
    """
    applied: list[str] = []
    with store.write() as conn:
        existing_tables = {
            row[0] for row in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        for table, columns in _declared_columns().items():
            if table not in existing_tables:
                continue
            present = {
                row[0] for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'main' AND table_name = ?", [table]
                ).fetchall()
            }
            for name, sql_type in columns:
                if name not in present:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
                    applied.append(f"{table}.{name}")
    return applied


def init() -> None:
    """建表并补齐新增列。幂等。"""
    with store.write() as conn:
        conn.execute(DDL)
    added = migrate()
    if added:
        print(f"[cdm] 已补列：{', '.join(added)}")


def clear_dataset(dataset: str, conn=None) -> None:
    """删除某个数据集在 CDM 各表中的全部行，用于重新导入。

    传入 `conn` 可以让这次删除并入调用方的事务——重新导入必须这么做，
    否则"删除"会先单独提交，并发的读就会看到一个空数据集。
    """
    def run(c) -> None:
        for t in TABLES:
            c.execute(f"DELETE FROM {t} WHERE dataset = ?", [dataset])
        c.execute("DELETE FROM dataset_registry WHERE dataset = ?", [dataset])

    if conn is not None:
        run(conn)
        return
    with store.write() as own:
        run(own)
