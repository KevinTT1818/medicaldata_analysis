"""UCI Heart Disease (Cleveland) 适配器。

原始文件：data/raw/uci_heart/processed.cleveland.data
303 行 x 14 列，无表头，缺失值记为 '?'（ca 列 4 个、thal 列 2 个）。
列定义来自同目录的 heart-disease.names。
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from ..cdm import units
from ..config import RAW_DIR
from .base import Adapter, CdmTables, register

SOURCE = RAW_DIR / "uci_heart" / "processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# 连续型测量：列名 -> (code, code_system, 中文名, 原始单位)
CONTINUOUS = {
    "trestbps": ("LOINC:8480-6", "LOINC", "静息收缩压", "mmHg"),
    "chol":     ("LOINC:2093-3", "LOINC", "总胆固醇", "mg/dL"),
    "thalach":  ("UCI:thalach", "UCI", "运动最大心率", "bpm"),
    "oldpeak":  ("UCI:oldpeak", "UCI", "运动诱发 ST 段压低", "mm"),
    "ca":       ("UCI:ca", "UCI", "荧光造影显示的主要血管数", "支"),
}

# 分类型测量：列名 -> (code, 中文名, {原始编码: 文字})
CATEGORICAL = {
    "cp": ("UCI:cp", "胸痛类型", {
        1.0: "典型心绞痛", 2.0: "非典型心绞痛",
        3.0: "非心绞痛性疼痛", 4.0: "无症状",
    }),
    "restecg": ("UCI:restecg", "静息心电图", {
        0.0: "正常", 1.0: "ST-T 波异常", 2.0: "左室肥厚",
    }),
    "slope": ("UCI:slope", "运动峰值 ST 段斜率", {
        1.0: "上斜", 2.0: "平坦", 3.0: "下斜",
    }),
    "thal": ("UCI:thal", "铊显像", {
        3.0: "正常", 6.0: "固定缺损", 7.0: "可逆缺损",
    }),
}

# 诊断：列名 -> (code, 中文名)，值为 1 时记一条 condition
CONDITIONS = {
    "fbs":   ("UCI:fbs_high", "空腹血糖 > 120 mg/dL"),
    "exang": ("UCI:exang", "运动诱发心绞痛"),
}


class UciHeartAdapter(Adapter):
    dataset_id = "uci_heart"
    label = "UCI Heart Disease (Cleveland)"
    description = "克利夫兰诊所冠心病数据集，303 例，14 个变量。结局为冠脉造影狭窄 > 50%。"
    source_url = "https://archive.ics.uci.edu/dataset/45/heart+disease"
    version = "1988-cleveland"

    def is_available(self) -> bool:
        return SOURCE.exists()

    def build(self) -> CdmTables:
        raw = pl.read_csv(
            SOURCE,
            has_header=False,
            new_columns=COLUMNS,
            null_values=["?"],
            schema_overrides={c: pl.Float64 for c in COLUMNS},
        )
        raw = raw.with_row_index("row")

        ds = self.dataset_id
        pid = pl.format("uci-{}", pl.col("row").cast(pl.Utf8).str.pad_start(4, "0"))

        person = raw.select(
            dataset=pl.lit(ds),
            person_id=pid,
            gender=pl.when(pl.col("sex") == 1.0).then(pl.lit("M"))
                     .when(pl.col("sex") == 0.0).then(pl.lit("F"))
                     .otherwise(None),
            birth_year=pl.lit(None, dtype=pl.Int32),
            age_at_index=pl.col("age"),
            race=pl.lit(None, dtype=pl.Utf8),
            ethnicity=pl.lit(None, dtype=pl.Utf8),
        )

        measurements: list[pl.DataFrame] = []

        for col, (code, system, text, unit_raw) in CONTINUOUS.items():
            conv_factor, target_unit = units.factor(unit_raw, code)
            frame = raw.select(
                dataset=pl.lit(ds),
                measurement_id=pl.format("{}-{}", pl.lit(code), pid),
                person_id=pid,
                visit_id=pl.lit(None, dtype=pl.Utf8),
                code=pl.lit(code),
                code_system=pl.lit(system),
                code_text=pl.lit(text),
                value_num=pl.col(col),          # 下面统一做单位换算
                value_text=pl.lit(None, dtype=pl.Utf8),
                unit=pl.lit(target_unit),
                value_raw=pl.col(col),
                unit_raw=pl.lit(unit_raw),
                ts=pl.lit(None, dtype=pl.Datetime),
                ref_low=pl.lit(None, dtype=pl.Float64),
                ref_high=pl.lit(None, dtype=pl.Float64),
            ).filter(pl.col("value_raw").is_not_null())

            if conv_factor != 1.0:
                frame = frame.with_columns(
                    value_num=(pl.col("value_raw") * conv_factor).round(4)
                )
            measurements.append(frame)

        for col, (code, text, levels) in CATEGORICAL.items():
            mapping = pl.when(pl.col(col) == list(levels)[0]).then(pl.lit(levels[list(levels)[0]]))
            for lv in list(levels)[1:]:
                mapping = mapping.when(pl.col(col) == lv).then(pl.lit(levels[lv]))
            mapping = mapping.otherwise(None)

            measurements.append(raw.select(
                dataset=pl.lit(ds),
                measurement_id=pl.format("{}-{}", pl.lit(code), pid),
                person_id=pid,
                visit_id=pl.lit(None, dtype=pl.Utf8),
                code=pl.lit(code),
                code_system=pl.lit("UCI"),
                code_text=pl.lit(text),
                value_num=pl.lit(None, dtype=pl.Float64),
                value_text=mapping,
                unit=pl.lit(None, dtype=pl.Utf8),
                value_raw=pl.col(col),
                unit_raw=pl.lit(None, dtype=pl.Utf8),
                ts=pl.lit(None, dtype=pl.Datetime),
                ref_low=pl.lit(None, dtype=pl.Float64),
                ref_high=pl.lit(None, dtype=pl.Float64),
            ).filter(pl.col("value_raw").is_not_null()))

        measurement = pl.concat(measurements, how="vertical")

        conditions = [
            raw.filter(pl.col(col) == 1.0).select(
                dataset=pl.lit(ds),
                condition_id=pl.format("{}-{}", pl.lit(code), pid),
                person_id=pid,
                visit_id=pl.lit(None, dtype=pl.Utf8),
                code=pl.lit(code),
                code_system=pl.lit("UCI"),
                code_text=pl.lit(text),
                onset_ts=pl.lit(None, dtype=pl.Datetime),
                is_primary=pl.lit(False),
            )
            for col, (code, text) in CONDITIONS.items()
        ]
        condition = pl.concat(conditions, how="vertical")

        # num: 0 = 狭窄 < 50%，1-4 = 狭窄 > 50%。二值化为结局事件。
        outcome = raw.filter(pl.col("num").is_not_null()).select(
            dataset=pl.lit(ds),
            person_id=pid,
            event_type=pl.lit("heart_disease"),
            event_ts=pl.lit(None, dtype=pl.Datetime),
            censor_ts=pl.lit(None, dtype=pl.Datetime),
            is_event=pl.col("num") > 0,
        )

        empty_visit = pl.DataFrame(schema={
            "dataset": pl.Utf8, "visit_id": pl.Utf8, "person_id": pl.Utf8,
            "start_ts": pl.Datetime, "end_ts": pl.Datetime,
            "visit_type": pl.Utf8, "discharge_status": pl.Utf8,
        })
        empty_drug = pl.DataFrame(schema={
            "dataset": pl.Utf8, "drug_id": pl.Utf8, "person_id": pl.Utf8,
            "visit_id": pl.Utf8, "code": pl.Utf8, "code_system": pl.Utf8,
            "name": pl.Utf8, "start_ts": pl.Datetime, "end_ts": pl.Datetime,
            "dose": pl.Float64, "dose_unit": pl.Utf8, "route": pl.Utf8,
        })

        return {
            "person": person,
            "visit": empty_visit,
            "condition": condition,
            "measurement": measurement,
            "drug": empty_drug,
            "outcome": outcome,
        }


register(UciHeartAdapter())
