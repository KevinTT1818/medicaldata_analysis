"""UCI Heart Failure Clinical Records 适配器。

原始文件：data/raw/heart_failure/heart_failure_clinical_records_dataset.csv
299 行 x 13 列，有表头，无缺失值。带随访天数（time）与死亡事件（DEATH_EVENT），
是本项目里第一个能做生存分析的数据集。

这个适配器是对「新增数据集只写一个 adapter」这条架构主张的检验：
写完之后，Table 1、缺失分析、分布、组间对比全部立刻可用，未改一行分析代码。
"""
from __future__ import annotations

import polars as pl

from ..cdm import units
from ..config import RAW_DIR
from .base import Adapter, CdmTables, register

SOURCE = RAW_DIR / "heart_failure" / "heart_failure_clinical_records_dataset.csv"

# 连续型测量：列名 -> (code, code_system, 中文名, 原始单位)
CONTINUOUS = {
    "creatinine_phosphokinase": ("LOINC:2157-6", "LOINC", "肌酸激酶 CPK", "U/L"),
    "ejection_fraction":        ("LOINC:10230-1", "LOINC", "左室射血分数", "%"),
    "platelets":                ("LOINC:26515-7", "LOINC", "血小板计数", "/mL"),
    "serum_creatinine":         ("LOINC:2160-0", "LOINC", "血清肌酐", "mg/dL"),
    "serum_sodium":             ("LOINC:2951-2", "LOINC", "血清钠", "mEq/L"),
}

# 诊断/暴露：列名 -> (code, 中文名)，值为 1 时记一条 condition
CONDITIONS = {
    "anaemia":             ("HF:anaemia", "贫血"),
    "diabetes":            ("HF:diabetes", "糖尿病"),
    "high_blood_pressure": ("HF:hypertension", "高血压"),
    "smoking":             ("HF:smoking", "吸烟"),
}


class HeartFailureAdapter(Adapter):
    dataset_id = "heart_failure"
    label = "UCI Heart Failure Clinical Records"
    description = (
        "心衰患者随访队列，299 例，随访 4–285 天，96 例死亡。"
        "带随访时长与事件标志，可做生存分析。"
    )
    source_url = "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"
    version = "2020-chicco-jurman"

    def is_available(self) -> bool:
        return SOURCE.exists()

    def build(self) -> CdmTables:
        # 全列强制 Float64：age 在文件后段有小数（60.667），
        # 靠前 100 行推断会得到 i64 并静默截断。
        raw = pl.read_csv(
            SOURCE, infer_schema=False,
        ).with_columns(pl.all().cast(pl.Float64)).with_row_index("row")
        ds = self.dataset_id
        pid = pl.format("hf-{}", pl.col("row").cast(pl.Utf8).str.pad_start(4, "0"))

        person = raw.select(
            dataset=pl.lit(ds),
            person_id=pid,
            # 该数据集 sex: 1 = 男, 0 = 女
            gender=pl.when(pl.col("sex") == 1).then(pl.lit("M"))
                     .when(pl.col("sex") == 0).then(pl.lit("F"))
                     .otherwise(None),
            birth_year=pl.lit(None, dtype=pl.Int32),
            age_at_index=pl.col("age").cast(pl.Float64),
            race=pl.lit(None, dtype=pl.Utf8),
            ethnicity=pl.lit(None, dtype=pl.Utf8),
        )

        measurements = []
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
                value_num=pl.col(col).cast(pl.Float64),
                value_text=pl.lit(None, dtype=pl.Utf8),
                unit=pl.lit(target_unit),
                value_raw=pl.col(col).cast(pl.Float64),
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

        condition = pl.concat([
            raw.filter(pl.col(col) == 1).select(
                dataset=pl.lit(ds),
                condition_id=pl.format("{}-{}", pl.lit(code), pid),
                person_id=pid,
                visit_id=pl.lit(None, dtype=pl.Utf8),
                code=pl.lit(code),
                code_system=pl.lit("HF"),
                code_text=pl.lit(text),
                onset_ts=pl.lit(None, dtype=pl.Datetime),
                is_primary=pl.lit(False),
            )
            for col, (code, text) in CONDITIONS.items()
        ], how="vertical")

        outcome = raw.select(
            dataset=pl.lit(ds),
            person_id=pid,
            event_type=pl.lit("death"),
            event_ts=pl.lit(None, dtype=pl.Datetime),
            censor_ts=pl.lit(None, dtype=pl.Datetime),
            followup_days=pl.col("time").cast(pl.Float64),
            is_event=pl.col("DEATH_EVENT") == 1,
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
            "measurement": pl.concat(measurements, how="vertical"),
            "drug": empty_drug,
            "outcome": outcome,
        }


register(HeartFailureAdapter())
