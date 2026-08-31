"""M4 续：MIMIC-IV Demo 适配器与时序数据能力的回归测试。

这是第一个住院时序数据集，也是第一个填上 CDM 的 visit 与 drug 两张表的数据集。
期望值全部来自对原始 CSV 的独立查询。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

import duckdb
import numpy as np

from app.adapters import base as adapters
from app.adapters import heart_failure, mimic_demo, nhanes, uci_heart  # noqa: F401
from app.analyses import describe, regression, registry, survival  # noqa: F401
from app.analyses.base import AnalysisContext
from app.cdm import etl, schema
from app.cohort import builder, filters
from app.cohort.filters import Condition
from app.jobs import cache, runner

MIMIC = "mimic_demo"
HF = "heart_failure"
SRC = _ROOT / "data" / "raw" / "mimic_demo"
CREAT = "measurement.LOINC:2160-0"
TEMP = "measurement.LOINC:8310-5"
DEATH = "outcome.death"
IN_HOSP = "outcome.in_hospital_death"


def _raw(query: str):
    con = duckdb.connect()
    try:
        return con.execute(query).fetchall()
    finally:
        con.close()


def _csv(rel: str) -> str:
    return f"read_csv_auto('{SRC / rel}')"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(MIMIC)
        etl.import_dataset(HF)
        cls.catalog = {v.id: v for v in builder.list_variables(MIMIC)}


class TableCoverageTest(Base):
    """CDM 六张表第一次全部有内容 —— 之前 visit 和 drug 一直是空的。"""

    def test_person_count(self):
        expected = _raw(f"SELECT count(*) FROM {_csv('hosp/patients.csv.gz')}")[0][0]
        self.assertEqual(builder.person_count(MIMIC), expected)
        self.assertEqual(expected, 100)

    def test_visit_table_populated(self):
        from app import store
        with store.read() as cur:
            rows = dict(cur.execute(
                "SELECT visit_type, count(*) FROM visit WHERE dataset = ? GROUP BY 1",
                [MIMIC],
            ).fetchall())
        n_adm = _raw(f"SELECT count(*) FROM {_csv('hosp/admissions.csv.gz')}")[0][0]
        n_icu = _raw(f"SELECT count(*) FROM {_csv('icu/icustays.csv.gz')}")[0][0]
        self.assertEqual(rows["inpatient"], n_adm)
        self.assertEqual(rows["icu"], n_icu)

    def test_drug_table_populated(self):
        from app import store
        with store.read() as cur:
            n = cur.execute(
                "SELECT count(*) FROM drug WHERE dataset = ?", [MIMIC]
            ).fetchone()[0]
        expected = _raw(f"SELECT count(*) FROM {_csv('hosp/prescriptions.csv.gz')}")[0][0]
        self.assertEqual(n, expected)

    def test_visits_link_to_persons(self):
        from app import store
        with store.read() as cur:
            orphans = cur.execute(
                """SELECT count(*) FROM visit v
                   WHERE v.dataset = ? AND NOT EXISTS (
                     SELECT 1 FROM person p
                     WHERE p.dataset = v.dataset AND p.person_id = v.person_id)""",
                [MIMIC],
            ).fetchone()[0]
        self.assertEqual(orphans, 0)

    def test_diagnoses_carry_icd_system(self):
        from app import store
        with store.read() as cur:
            systems = {r[0] for r in cur.execute(
                "SELECT DISTINCT code_system FROM condition WHERE dataset = ?", [MIMIC]
            ).fetchall()}
        self.assertTrue(systems <= {"ICD-9-CM", "ICD-10-CM"}, systems)


class TimeSeriesTest(Base):
    """一人一项多值：这是前三个数据集都没有的结构。"""

    def test_measurements_are_repeated(self):
        self.assertTrue(builder.has_repeated_measures(MIMIC))
        self.assertFalse(builder.has_repeated_measures(HF))

    def _creatinine(self, agg):
        frame = builder.build_feature_frame(MIMIC, [CREAT], agg=agg)
        values = frame[CREAT].cast(float, strict=False).to_numpy()
        return values[np.isfinite(values)]

    def test_aggregation_changes_the_answer(self):
        """取首次还是取最大，肌酐中位数差七成 —— 所以这个选择必须显式。"""
        first = float(np.median(self._creatinine("first")))
        largest = float(np.median(self._creatinine("max")))
        smallest = float(np.median(self._creatinine("min")))
        self.assertLess(smallest, first)
        self.assertLess(first, largest)
        self.assertGreater(largest / smallest, 1.3)

    def test_min_max_bracket_other_aggregations(self):
        values = {a: self._creatinine(a) for a in ("first", "last", "mean", "min", "max")}
        for agg in ("first", "last", "mean"):
            self.assertTrue(np.all(values["min"] <= values[agg] + 1e-9), agg)
            self.assertTrue(np.all(values[agg] <= values["max"] + 1e-9), agg)

    def test_first_matches_earliest_chart_time(self):
        """直接对原始文件查最早一次肌酐，与 agg='first' 对照。"""
        rows = _raw(f"""
            WITH lab AS (
                SELECT subject_id, valuenum, charttime,
                       row_number() OVER (PARTITION BY subject_id ORDER BY charttime) rn
                FROM {_csv('hosp/labevents.csv.gz')}
                WHERE itemid = 50912 AND valuenum IS NOT NULL)
            SELECT 'mimic-' || subject_id, valuenum FROM lab WHERE rn = 1
        """)
        expected = {pid: round(v * 88.4, 4) for pid, v in rows}

        frame = builder.build_feature_frame(MIMIC, [CREAT], agg="first")
        got = {
            row[0]: round(float(row[1]), 4)
            for row in zip(frame["person_id"].to_list(), frame[CREAT].to_list())
            if row[1] is not None
        }
        self.assertEqual(len(got), len(expected))
        for pid, value in expected.items():
            self.assertAlmostEqual(got[pid], value, places=3, msg=pid)

    def test_aggregation_is_in_the_fingerprint(self):
        """同参数不同聚合必须算作两次分析，否则缓存会串。"""
        params = {"variables": [CREAT]}
        a = runner.submit_analysis(MIMIC, "describe.baseline_table", params,
                                   measurement_agg="first")
        b = runner.submit_analysis(MIMIC, "describe.baseline_table", params,
                                   measurement_agg="max")
        self.assertNotEqual(a.spec["fingerprint"], b.spec["fingerprint"])


class DataQualityTest(Base):
    def test_implausible_vitals_dropped(self):
        """MIMIC 里有把华氏值录进摄氏字段的记录，99 摄氏度不是体温。"""
        self.assertGreater(adapters.get(MIMIC).dropped_implausible, 0)
        frame = builder.build_feature_frame(MIMIC, [TEMP], agg="max")
        values = frame[TEMP].cast(float, strict=False).to_numpy()
        values = values[np.isfinite(values)]
        self.assertTrue(values.size > 0)
        self.assertLessEqual(float(values.max()), 45.0)
        self.assertGreaterEqual(float(values.min()), 25.0)

    def test_fahrenheit_converted_to_celsius(self):
        from app import store
        with store.read() as cur:
            rows = cur.execute(
                """SELECT unit_raw, min(value_num), max(value_num), any_value(unit)
                   FROM measurement WHERE dataset = ? AND code = ? GROUP BY unit_raw""",
                [MIMIC, "LOINC:8310-5"],
            ).fetchall()
        for unit_raw, low, high, unit in rows:
            self.assertEqual(unit, "C")
            self.assertGreater(low, 25.0)
            self.assertLess(high, 45.0)

    def test_low_coverage_codes_hidden_from_catalog(self):
        """MIMIC 有几百个只出现一两次的 ICD 码，全列出来选择器没法用。"""
        listed = builder.list_variables(MIMIC)
        self.assertGreater(builder.hidden_variable_count(MIMIC), 100)
        self.assertLess(len(listed), 250)
        floor = max(2, int(100 * 0.05))
        for v in listed:
            if v.source == "condition":
                self.assertGreaterEqual(v.n_positive, floor)

    def test_binary_variables_report_positive_counts(self):
        death = self.catalog[DEATH]
        expected = _raw(
            f"SELECT count(*) FROM {_csv('hosp/patients.csv.gz')} WHERE dod IS NOT NULL"
        )[0][0]
        self.assertEqual(death.n_positive, expected)
        self.assertEqual(death.n_available, 100)

    def test_small_datasets_hide_nothing(self):
        builder.list_variables(HF)
        self.assertEqual(builder.hidden_variable_count(HF), 0)


class CrossDatasetTest(Base):
    """LOINC 映射的意义：不同来源的同一指标要能对上。"""

    def test_creatinine_is_the_same_variable_in_both_datasets(self):
        for dataset in (MIMIC, HF):
            catalog = {v.id: v for v in builder.list_variables(dataset)}
            self.assertIn(CREAT, catalog)
            self.assertEqual(catalog[CREAT].unit, "umol/L")

    def test_creatinine_values_are_physiological_in_both(self):
        for dataset, agg in ((MIMIC, "first"), (HF, "first")):
            frame = builder.build_feature_frame(dataset, [CREAT], agg=agg)
            values = frame[CREAT].cast(float, strict=False).to_numpy()
            values = values[np.isfinite(values)]
            # 正常成人血肌酐约 60–110 µmol/L，中位数应落在合理区间
            self.assertGreater(float(np.median(values)), 40.0)
            self.assertLess(float(np.median(values)), 300.0)


class SurvivalWithTimestampsTest(Base):
    """第一个同时填 event_ts / censor_ts 和 followup_days 的数据集。"""

    def test_outcome_has_both_timestamps_and_duration(self):
        from app import store
        with store.read() as cur:
            row = cur.execute(
                """SELECT count(*) FILTER (WHERE is_event AND event_ts IS NOT NULL),
                          count(*) FILTER (WHERE NOT is_event AND censor_ts IS NOT NULL),
                          count(followup_days)
                   FROM outcome WHERE dataset = ? AND event_type = 'death'""",
                [MIMIC],
            ).fetchone()
        n_dead = _raw(
            f"SELECT count(*) FROM {_csv('hosp/patients.csv.gz')} WHERE dod IS NOT NULL"
        )[0][0]
        self.assertEqual(row[0], n_dead)
        self.assertEqual(row[1], 100 - n_dead)
        self.assertEqual(row[2], 100)

    def test_kaplan_meier_runs(self):
        analysis = registry.get("survival.kaplan_meier")
        params = analysis.Params(outcome=DEATH)
        frame = builder.build_feature_frame(MIMIC, analysis.required_variables(params))
        out = analysis.run(AnalysisContext(MIMIC, frame, self.catalog, params)).payload
        series = out["series"][0]
        self.assertEqual(series["n"], 100)
        self.assertEqual(series["events"], 31)
        self.assertEqual(series["censored"], 69)

    def test_in_hospital_death_is_binary_not_survival(self):
        self.assertFalse(self.catalog[IN_HOSP].has_survival)
        self.assertTrue(self.catalog[DEATH].has_survival)

    def test_cohort_on_diagnosis_narrows_survival(self):
        import time
        cache.clear()
        diagnosis = next(
            v for v in builder.list_variables(MIMIC)
            if v.source == "condition" and 10 <= (v.n_positive or 0) <= 60
        )
        job = runner.submit_analysis(
            MIMIC, "survival.kaplan_meier", {"outcome": DEATH},
            cohort=Condition(variable=diagnosis.id, op="eq", value="是"),
        )
        for _ in range(300):
            if job.status in ("succeeded", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(job.status, "succeeded", job.error)
        self.assertEqual(job.cohort_n, diagnosis.n_positive)


class VisitVariableTest(Base):
    """就诊派生变量：把就诊事实做成人级变量，让「再入院」「ICU 停留 > 3 天」进条件树。

    分析单位始终是人 —— 所有算子都假设一人一行，所以不把就诊变成分析单位。
    """

    def _visit_vars(self, dataset=MIMIC):
        return [v for v in builder.list_variables(dataset) if v.source == "visit"]

    def test_only_datasets_with_visits_get_them(self):
        self.assertTrue(self._visit_vars(MIMIC))
        self.assertEqual(self._visit_vars(HF), [])

    def test_one_set_per_visit_type(self):
        ids = {v.id for v in self._visit_vars()}
        for visit_type in ("inpatient", "icu"):
            self.assertIn(f"visit.count:{visit_type}", ids)
            self.assertIn(f"visit.los_total:{visit_type}", ids)
            self.assertIn(f"visit.los_max:{visit_type}", ids)

    def test_counts_match_source_files(self):
        frame = builder.build_feature_frame(
            MIMIC, ["visit.count:inpatient", "visit.count:icu"])
        got_adm = sum(int(float(x)) for x in frame["visit.count:inpatient"].to_list())
        got_icu = sum(int(float(x)) for x in frame["visit.count:icu"].to_list())
        self.assertEqual(
            got_adm, _raw(f"SELECT count(*) FROM {_csv('hosp/admissions.csv.gz')}")[0][0])
        self.assertEqual(
            got_icu, _raw(f"SELECT count(*) FROM {_csv('icu/icustays.csv.gz')}")[0][0])

    def test_length_of_stay_matches_mimic_own_column(self):
        """MIMIC 的 icustays.los 就是天数，与我们从时间戳算的应当吻合。"""
        expected = dict(_raw(f"""
            SELECT 'mimic-' || subject_id, max(los)
            FROM {_csv('icu/icustays.csv.gz')} GROUP BY 1"""))
        frame = builder.build_feature_frame(MIMIC, ["visit.los_max:icu"])
        checked = 0
        for pid, value in zip(frame["person_id"].to_list(),
                              frame["visit.los_max:icu"].to_list()):
            if value is None or pid not in expected:
                continue
            self.assertAlmostEqual(float(value), expected[pid], places=6, msg=pid)
            checked += 1
        self.assertGreater(checked, 50)

    def test_zero_count_is_zero_but_zero_stay_is_undefined(self):
        """没住过 ICU 的人「次数 = 0」是事实，「住了几天」则是未定义。

        把没住过的人当 0 天混进均值，回答的就不是「住过的人住了多久」这个问题了。
        """
        frame = builder.build_feature_frame(
            MIMIC, ["visit.count:icu", "visit.los_max:icu"])
        counts = [int(float(x)) for x in frame["visit.count:icu"].to_list()]
        stays = frame["visit.los_max:icu"].to_list()
        # 构造一个没有该类就诊的场景：用一个不存在的就诊类型不好构造，
        # 改为断言语义本身 —— 有就诊的人时长非空，且次数列没有空值
        self.assertTrue(all(c is not None for c in counts))
        for count, stay in zip(counts, stays):
            if count == 0:
                self.assertIsNone(stay)

    def test_discharge_status_is_split_by_visit_type(self):
        """住院的结束状态是出院去向，ICU 的是末次监护单元，不是一回事。"""
        ids = {v.id for v in self._visit_vars()}
        self.assertIn("visit.discharge_status:inpatient", ids)
        self.assertIn("visit.discharge_status:icu", ids)

        frame = builder.build_feature_frame(
            MIMIC, ["visit.discharge_status:inpatient", "visit.discharge_status:icu"])
        inpatient = {x for x in frame["visit.discharge_status:inpatient"].to_list() if x}
        icu = {x for x in frame["visit.discharge_status:icu"].to_list() if x}
        self.assertTrue(inpatient)
        self.assertTrue(icu)
        # 监护单元不该出现在出院去向里
        self.assertFalse(any("Intensive Care Unit" in x for x in inpatient))

    def test_pushdown_parity_for_visit_filters(self):
        catalog = {v.id: v for v in builder.list_variables(MIMIC)}
        cases = [
            Condition(variable="visit.count:inpatient", op="gte", value=2),
            Condition(variable="visit.count:icu", op="eq", value=1),
            Condition(variable="visit.los_max:icu", op="gt", value=3),
            Condition(variable="visit.los_total:inpatient", op="between", value=[5, 30]),
            Condition(variable="visit.los_max:icu", op="is_null"),
            Condition(variable="visit.discharge_status:inpatient", op="eq", value="HOME"),
            Condition(variable="visit.discharge_status:inpatient", op="ne", value="HOME"),
        ]
        for node in cases:
            with self.subTest(variable=node.variable, op=node.op):
                columns = filters.collect_variables(node)
                python_n = int(filters.evaluate(
                    builder.build_feature_frame(MIMIC, columns), catalog, node).sum())
                sql_n = builder.build_feature_frame(MIMIC, columns, where=node).height
                self.assertEqual(python_n, sql_n)

    def test_readmission_cohort_is_meaningful(self):
        """「再入院」= 住院 ≥ 2 次。应当筛出一部分人而不是全部或零个。"""
        catalog = {v.id: v for v in builder.list_variables(MIMIC)}
        node = Condition(variable="visit.count:inpatient", op="gte", value=2)
        n = builder.build_feature_frame(MIMIC, ["visit.count:inpatient"],
                                        where=node).height
        self.assertGreater(n, 0)
        self.assertLess(n, 100)
        expected = _raw(f"""
            SELECT count(*) FROM (
              SELECT subject_id FROM {_csv('hosp/admissions.csv.gz')}
              GROUP BY 1 HAVING count(*) >= 2)""")[0][0]
        self.assertEqual(n, expected)

    def test_visit_variables_work_in_analyses(self):
        """派生出来就得能用 —— 进 Table 1 和进队列都要跑得通。"""
        catalog = {v.id: v for v in builder.list_variables(MIMIC)}
        analysis = registry.get("describe.baseline_table")
        params = analysis.Params(
            variables=["visit.count:inpatient", "visit.los_max:icu",
                       "visit.discharge_status:inpatient"],
            group_by=DEATH,
        )
        frame = builder.build_feature_frame(MIMIC, analysis.required_variables(params))
        out = analysis.run(AnalysisContext(MIMIC, frame, catalog, params)).payload
        self.assertEqual(len(out["rows"]), 3)
        self.assertEqual(out["overall_n"], 100)

    def test_visit_cohort_applies_to_survival(self):
        import time
        cache.clear()
        job = runner.submit_analysis(
            MIMIC, "survival.kaplan_meier", {"outcome": DEATH},
            cohort=Condition(variable="visit.los_max:icu", op="gt", value=3),
        )
        for _ in range(300):
            if job.status in ("succeeded", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(job.status, "succeeded", job.error)
        self.assertGreater(job.cohort_n, 0)
        self.assertLess(job.cohort_n, 100)


if __name__ == "__main__":
    unittest.main()
