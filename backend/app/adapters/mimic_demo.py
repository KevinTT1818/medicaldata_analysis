"""MIMIC-IV Clinical Database Demo 2.2 适配器。

原始文件：data/raw/mimic_demo/{hosp,icu}/*.csv.gz（免认证公开版，100 名患者）
用 DuckDB 直接读 gzip CSV，不经 pandas —— labevents 有 10.7 万行、
chartevents 66.9 万行，让查询引擎在源文件上做筛选和聚合更划算。

这是本项目第一个**时序住院**数据集，前三个都是一人一行的横断面或队列结构。
它第一次填上了 CDM 的 visit 与 drug 两张表，也第一次出现「一个人一个化验项有
几十上百个值」的情况 —— 取哪一个值（首次 / 末次 / 均值）会改变结论，
所以宽表的聚合策略必须由调用方显式指定，不能藏在实现里。

三个 MIMIC 特有的坑：

1. **日期被平移到 2100 年之后**做去标识化。`anchor_year` 与 `anchor_age` 给出
   「此人在 anchor_year 那年多少岁」，据此才能算出各次住院时的年龄。

2. **89 岁以上的患者年龄被统一记为 91 岁**（HIPAA 要求）。把 91 当真实年龄用会
   低估这批人的年龄，做年龄分层时要意识到这一点。

3. **化验项有 498 种**，全导进来变量目录就没法用了。这里按患者覆盖率取阈值，
   只导入覆盖多数患者的常规化验，并映射到 LOINC —— 映射之后，
   MIMIC 的肌酐和心衰队列的肌酐是同一个变量，单位也统一到 µmol/L。
"""
from __future__ import annotations

import duckdb
import polars as pl

from ..cdm import units
from ..config import RAW_DIR
from .base import Adapter, CdmTables, register

SOURCE = RAW_DIR / "mimic_demo"

#: 化验项至少要覆盖这么大比例的患者才导入，避免变量目录被 498 项淹没
LAB_COVERAGE_THRESHOLD = 0.5

#: itemid -> (LOINC code, 中文名)。映射到 LOINC 才能跨数据集比较。
LAB_LOINC = {
    50862: ("LOINC:1751-6", "白蛋白"),
    50863: ("LOINC:6768-6", "碱性磷酸酶"),
    50868: ("LOINC:33037-3", "阴离子间隙"),
    50882: ("LOINC:1963-8", "碳酸氢盐"),
    50885: ("LOINC:1975-2", "总胆红素"),
    50893: ("LOINC:17861-6", "总钙"),
    50902: ("LOINC:2075-0", "氯"),
    50912: ("LOINC:2160-0", "肌酐"),
    50931: ("LOINC:2345-7", "葡萄糖"),
    50960: ("LOINC:2601-3", "镁"),
    50970: ("LOINC:2777-1", "磷酸盐"),
    50971: ("LOINC:2823-3", "钾"),
    50983: ("LOINC:2951-2", "钠"),
    51006: ("LOINC:3094-0", "尿素氮"),
    51221: ("LOINC:4544-3", "红细胞压积"),
    51222: ("LOINC:718-7", "血红蛋白"),
    51237: ("LOINC:34714-6", "INR"),
    51248: ("LOINC:785-6", "平均红细胞血红蛋白量"),
    51249: ("LOINC:786-4", "平均红细胞血红蛋白浓度"),
    51250: ("LOINC:787-2", "平均红细胞体积"),
    51265: ("LOINC:777-3", "血小板计数"),
    51274: ("LOINC:5902-2", "凝血酶原时间"),
    51275: ("LOINC:14979-9", "部分凝血活酶时间"),
    51277: ("LOINC:788-0", "红细胞分布宽度"),
    51279: ("LOINC:789-8", "红细胞计数"),
    51301: ("LOINC:6690-2", "白细胞计数"),
}

#: ICU 生命体征。chartevents 有 1318 项，只取这几个标准体征。
#: 收缩压 / 舒张压刻意用与 NHANES、UCI 相同的 LOINC 编码 —— 同一个变量。
VITALS = {
    220045: ("LOINC:8867-4", "心率", "bpm"),
    220179: ("LOINC:8480-6", "收缩压", "mmHg"),
    220180: ("LOINC:8462-4", "舒张压", "mmHg"),
    220050: ("LOINC:8480-6", "收缩压", "mmHg"),   # 有创动脉压，同一指标
    220051: ("LOINC:8462-4", "舒张压", "mmHg"),
    220210: ("LOINC:9279-1", "呼吸频率", "次/分"),
    220277: ("LOINC:59408-5", "脉搏血氧饱和度", "%"),
    223762: ("LOINC:8310-5", "体温", "C"),
    223761: ("LOINC:8310-5", "体温", "F"),        # 华氏，需偏移换算，下面单独处理
}


#: 生命体征的生理合理范围（换算到目标单位之后）。超出范围的按录入错误剔除。
#: MIMIC 里确实存在把华氏值录进摄氏字段这类错误 —— 99 摄氏度不是体温。
PLAUSIBLE_RANGE = {
    "LOINC:8867-4": (20.0, 300.0),    # 心率 bpm
    "LOINC:8480-6": (30.0, 300.0),    # 收缩压 mmHg
    "LOINC:8462-4": (10.0, 200.0),    # 舒张压 mmHg
    "LOINC:9279-1": (3.0, 60.0),      # 呼吸频率 次/分
    "LOINC:59408-5": (50.0, 100.0),   # SpO2 %
    "LOINC:8310-5": (25.0, 45.0),     # 体温 摄氏
}


def _csv(relative: str) -> str:
    return f"read_csv_auto('{SOURCE / relative}')"


class MimicDemoAdapter(Adapter):
    dataset_id = "mimic_demo"
    label = "MIMIC-IV Demo 2.2"
    description = (
        "MIMIC-IV 免认证演示版，100 名 ICU 患者、275 次住院、140 次 ICU 停留。"
        "带完整的就诊、诊断、化验时序与用药记录。日期已平移去标识化。"
    )
    source_url = "https://physionet.org/content/mimic-iv-demo/2.2/"
    version = "2.2-demo"

    #: 因超出生理范围被剔除的生命体征条数，导入后可查
    dropped_implausible: int = 0

    def is_available(self) -> bool:
        required = [
            "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "icu/icustays.csv.gz",
            "hosp/diagnoses_icd.csv.gz", "hosp/d_icd_diagnoses.csv.gz",
            "hosp/labevents.csv.gz", "hosp/d_labitems.csv.gz",
            "hosp/prescriptions.csv.gz", "icu/chartevents.csv.gz",
        ]
        return all((SOURCE / r).exists() for r in required)

    def build(self) -> CdmTables:
        ds = self.dataset_id
        con = duckdb.connect()

        # ---- person ----
        # age_at_index 取「首次住院时的年龄」= anchor_age + (首次入院年 - anchor_year)
        person = con.execute(f"""
            WITH first_admit AS (
                SELECT subject_id, min(admittime) AS t0
                FROM {_csv('hosp/admissions.csv.gz')} GROUP BY 1
            )
            SELECT
                '{ds}'                                        AS dataset,
                'mimic-' || p.subject_id                      AS person_id,
                p.gender                                      AS gender,
                CAST(p.anchor_year - p.anchor_age AS INTEGER) AS birth_year,
                CAST(p.anchor_age + coalesce(
                    year(f.t0) - p.anchor_year, 0) AS DOUBLE) AS age_at_index,
                CAST(NULL AS VARCHAR)                         AS race,
                CAST(NULL AS VARCHAR)                         AS ethnicity,
                CAST(NULL AS DOUBLE)                          AS sample_weight,
                CAST(NULL AS VARCHAR)                         AS psu,
                CAST(NULL AS VARCHAR)                         AS stratum
            FROM {_csv('hosp/patients.csv.gz')} p
            LEFT JOIN first_admit f USING (subject_id)
        """).pl()

        # ---- visit：住院 + ICU 停留，第一次真正用上这张表 ----
        visit = con.execute(f"""
            SELECT
                '{ds}'                          AS dataset,
                'hadm-' || hadm_id              AS visit_id,
                'mimic-' || subject_id          AS person_id,
                admittime                       AS start_ts,
                dischtime                       AS end_ts,
                'inpatient'                     AS visit_type,
                discharge_location              AS discharge_status
            FROM {_csv('hosp/admissions.csv.gz')}
            UNION ALL
            SELECT
                '{ds}', 'icu-' || stay_id, 'mimic-' || subject_id,
                intime, outtime, 'icu', last_careunit
            FROM {_csv('icu/icustays.csv.gz')}
        """).pl()

        # ---- condition：ICD 诊断 ----
        condition = con.execute(f"""
            SELECT
                '{ds}'                                          AS dataset,
                'dx-' || d.hadm_id || '-' || d.seq_num          AS condition_id,
                'mimic-' || d.subject_id                        AS person_id,
                'hadm-' || d.hadm_id                            AS visit_id,
                d.icd_code                                      AS code,
                'ICD-' || d.icd_version || '-CM'                AS code_system,
                coalesce(t.long_title, d.icd_code)              AS code_text,
                a.admittime                                     AS onset_ts,
                d.seq_num = 1                                   AS is_primary
            FROM {_csv('hosp/diagnoses_icd.csv.gz')} d
            LEFT JOIN {_csv('hosp/d_icd_diagnoses.csv.gz')} t
                   ON t.icd_code = d.icd_code AND t.icd_version = d.icd_version
            LEFT JOIN {_csv('hosp/admissions.csv.gz')} a USING (hadm_id)
        """).pl()

        # ---- measurement：化验（按覆盖率筛项）+ ICU 生命体征 ----
        n_patients = con.execute(
            f"SELECT count(*) FROM {_csv('hosp/patients.csv.gz')}").fetchone()[0]
        min_patients = int(n_patients * LAB_COVERAGE_THRESHOLD)

        kept = con.execute(f"""
            SELECT itemid FROM {_csv('hosp/labevents.csv.gz')}
            WHERE valuenum IS NOT NULL
            GROUP BY itemid HAVING count(DISTINCT subject_id) >= {min_patients}
        """).fetchall()
        kept_ids = {row[0] for row in kept} & set(LAB_LOINC)
        id_list = ", ".join(str(i) for i in sorted(kept_ids)) or "NULL"

        # LOINC 与中文名通过 VALUES 映射进 SQL
        mapping = ", ".join(
            f"({i}, '{LAB_LOINC[i][0]}', '{LAB_LOINC[i][1]}')" for i in sorted(kept_ids)
        ) or "(NULL, NULL, NULL)"

        labs = con.execute(f"""
            WITH m(itemid, loinc, label) AS (VALUES {mapping})
            SELECT
                '{ds}'                                  AS dataset,
                'lab-' || l.labevent_id                 AS measurement_id,
                'mimic-' || l.subject_id                AS person_id,
                CASE WHEN l.hadm_id IS NULL THEN NULL
                     ELSE 'hadm-' || l.hadm_id END      AS visit_id,
                m.loinc                                 AS code,
                'LOINC'                                 AS code_system,
                m.label                                 AS code_text,
                l.valuenum                              AS value_num,
                CAST(NULL AS VARCHAR)                   AS value_text,
                l.valueuom                              AS unit,
                l.valuenum                              AS value_raw,
                l.valueuom                              AS unit_raw,
                l.charttime                             AS ts,
                l.ref_range_lower                       AS ref_low,
                l.ref_range_upper                       AS ref_high
            FROM {_csv('hosp/labevents.csv.gz')} l
            JOIN m USING (itemid)
            WHERE l.valuenum IS NOT NULL AND l.itemid IN ({id_list})
        """).pl()

        vital_map = ", ".join(
            f"({i}, '{c}', '{label}', '{unit}')" for i, (c, label, unit) in VITALS.items()
        )
        vitals = con.execute(f"""
            WITH v(itemid, loinc, label, unit) AS (VALUES {vital_map})
            SELECT
                '{ds}'                                  AS dataset,
                'vital-' || c.stay_id || '-' || c.itemid || '-' ||
                    CAST(epoch(c.charttime) AS BIGINT)  AS measurement_id,
                'mimic-' || c.subject_id                AS person_id,
                'icu-' || c.stay_id                     AS visit_id,
                v.loinc                                 AS code,
                'LOINC'                                 AS code_system,
                v.label                                 AS code_text,
                -- 华氏体温要偏移换算，units.py 只支持乘法，这里显式处理
                CASE WHEN v.unit = 'F' THEN round((c.valuenum - 32) * 5.0 / 9.0, 4)
                     ELSE c.valuenum END                AS value_num,
                CAST(NULL AS VARCHAR)                   AS value_text,
                CASE WHEN v.unit = 'F' THEN 'C' ELSE v.unit END AS unit,
                c.valuenum                              AS value_raw,
                v.unit                                  AS unit_raw,
                c.charttime                             AS ts,
                CAST(NULL AS DOUBLE)                    AS ref_low,
                CAST(NULL AS DOUBLE)                    AS ref_high
            FROM {_csv('icu/chartevents.csv.gz')} c
            JOIN v USING (itemid)
            WHERE c.valuenum IS NOT NULL
        """).pl()

        # 剔除生理上不可能的值。数量记进 self.dropped_implausible，导入时报出来。
        before = vitals.height
        keep = pl.lit(True)
        for code, (low, high) in PLAUSIBLE_RANGE.items():
            keep = keep & (
                (pl.col("code") != code)
                | (pl.col("value_num").is_between(low, high))
            )
        vitals = vitals.filter(keep)
        self.dropped_implausible = before - vitals.height

        measurement = pl.concat([labs, vitals], how="vertical_relaxed")

        # 化验的单位归一（肌酐 mg/dL -> µmol/L 等），与其他数据集对齐
        for code, (target_unit, _kind) in units.TARGETS.items():
            source_units = (
                measurement.filter(pl.col("code") == code)["unit_raw"].unique().to_list()
            )
            for source_unit in source_units:
                if source_unit is None or source_unit == target_unit:
                    continue
                factor, _ = units.factor(source_unit, code)
                if factor == 1.0:
                    continue
                measurement = measurement.with_columns(
                    value_num=pl.when(
                        (pl.col("code") == code) & (pl.col("unit_raw") == source_unit)
                    ).then((pl.col("value_raw") * factor).round(4))
                     .otherwise(pl.col("value_num")),
                    unit=pl.when(
                        (pl.col("code") == code) & (pl.col("unit_raw") == source_unit)
                    ).then(pl.lit(target_unit)).otherwise(pl.col("unit")),
                )

        # ---- drug：处方，第一次用上这张表 ----
        drug = con.execute(f"""
            SELECT
                '{ds}'                                  AS dataset,
                'rx-' || pharmacy_id || '-' || row_number() OVER () AS drug_id,
                'mimic-' || subject_id                  AS person_id,
                'hadm-' || hadm_id                      AS visit_id,
                coalesce(gsn, formulary_drug_cd)        AS code,
                CASE WHEN gsn IS NOT NULL THEN 'GSN' ELSE 'MIMIC-formulary' END AS code_system,
                drug                                    AS name,
                starttime                               AS start_ts,
                stoptime                                AS end_ts,
                TRY_CAST(dose_val_rx AS DOUBLE)         AS dose,
                dose_unit_rx                            AS dose_unit,
                route                                   AS route
            FROM {_csv('hosp/prescriptions.csv.gz')}
        """).pl()

        # ---- outcome ----
        # 死亡：随访起点是首次入院，终点是死亡日期或末次出院（删失）。
        # 这是第一个同时填 event_ts / censor_ts 和 followup_days 的数据集。
        death = con.execute(f"""
            WITH span AS (
                SELECT subject_id, min(admittime) AS t0, max(dischtime) AS t_last
                FROM {_csv('hosp/admissions.csv.gz')} GROUP BY 1
            )
            SELECT
                '{ds}'                                          AS dataset,
                'mimic-' || p.subject_id                        AS person_id,
                'death'                                         AS event_type,
                CAST(p.dod AS TIMESTAMP)                        AS event_ts,
                CASE WHEN p.dod IS NULL THEN s.t_last END       AS censor_ts,
                CAST(date_diff('day', s.t0,
                    coalesce(CAST(p.dod AS TIMESTAMP), s.t_last)) AS DOUBLE) AS followup_days,
                p.dod IS NOT NULL                               AS is_event
            FROM {_csv('hosp/patients.csv.gz')} p
            JOIN span s USING (subject_id)
        """).pl()

        in_hospital = con.execute(f"""
            SELECT
                '{ds}'                              AS dataset,
                'mimic-' || subject_id              AS person_id,
                'in_hospital_death'                 AS event_type,
                -- NULL 必须带类型，否则拼接时会被推断成 BIGINT 与 death 段冲突
                CAST(NULL AS TIMESTAMP)             AS event_ts,
                CAST(NULL AS TIMESTAMP)             AS censor_ts,
                CAST(NULL AS DOUBLE)                AS followup_days,
                max(hospital_expire_flag) = 1       AS is_event
            FROM {_csv('hosp/admissions.csv.gz')}
            GROUP BY 1, 2, 3
        """).pl()

        con.close()

        return {
            "person": person,
            "visit": visit,
            "condition": condition,
            "measurement": measurement,
            "drug": drug,
            "outcome": pl.concat([death, in_hospital], how="vertical_relaxed"),
        }


register(MimicDemoAdapter())
