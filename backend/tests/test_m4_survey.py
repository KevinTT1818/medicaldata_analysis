"""M4 回归测试：NHANES 适配器、抽样权重、SQL 下推、结果缓存。

加权估计用 NCHS 已发表的人群数字做外部校验 —— 这是最强的正确性证据：
不是自己跟自己比，而是跟官方统计比。
"""
from __future__ import annotations

import unittest

from _env import CACHE_DIR, ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

import numpy as np
import pandas as pd

from app.adapters import heart_failure, nhanes, uci_heart  # noqa: F401
from app.analyses import describe, regression, registry, survival  # noqa: F401
from app.analyses import weights as W
from app.analyses.base import AnalysisContext
from app.cdm import etl, schema
from app.cohort import builder, filters
from app.cohort.filters import Condition, Group
from app.jobs import cache, runner

NH, HF, UCI = "nhanes_2017", "heart_failure", "uci_heart"
AGE = "person.age_at_index"
SEX = "person.gender"
BMI = "measurement.LOINC:39156-5"
SBP = "measurement.LOINC:8480-6"
CHOL = "measurement.LOINC:2093-3"
DM_Q = "measurement.NHANES:diabetes"
RAW = _ROOT / "data" / "raw" / "nhanes"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        for ds in (NH, HF, UCI):
            etl.import_dataset(ds)
        cls.catalog = {
            ds: {v.id: v for v in builder.list_variables(ds)} for ds in (NH, HF, UCI)
        }

    def run_analysis(self, analysis_id, dataset=NH, **params):
        analysis = registry.get(analysis_id)
        parsed = analysis.Params(**params)
        frame = builder.build_feature_frame(dataset, analysis.required_variables(parsed))
        return analysis.run(AnalysisContext(dataset, frame, self.catalog[dataset], parsed))


class WeightedStatisticsTest(unittest.TestCase):
    """加权统计的数学性质。"""

    def test_equal_weights_degenerate_to_unweighted(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        w = np.ones(5)
        self.assertAlmostEqual(W.mean(x, w), float(np.mean(x)), places=10)
        self.assertAlmostEqual(W.sd(x, w), float(np.std(x)), places=10)
        self.assertAlmostEqual(W.quantile(x, w, 0.5), float(np.percentile(x, 50)), places=10)

    def test_zero_weight_rows_are_excluded(self):
        """权重为 0 的人（如只参加访谈未参加体检）不该影响估计。"""
        x = np.array([1.0, 2.0, 999.0])
        w = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(W.mean(x, w), 1.5, places=10)

    def test_weight_shifts_estimate(self):
        x = np.array([1.0, 100.0])
        self.assertLess(W.mean(x, np.array([9.0, 1.0])), W.mean(x, np.array([1.0, 9.0])))

    def test_effective_n_penalises_unequal_weights(self):
        equal = W.effective_n(np.ones(100))
        skewed = W.effective_n(np.concatenate([np.ones(99), [1000.0]]))
        self.assertAlmostEqual(equal, 100.0, places=6)
        self.assertLess(skewed, equal)

    def test_proportion_is_weighted(self):
        mask = np.array([True, False])
        self.assertAlmostEqual(W.proportion(mask, np.array([3.0, 1.0])), 0.75, places=10)

    def test_has_weights_returns_python_bool(self):
        """numpy.bool_ 过不了 JSON 序列化，会让整份结果发不出去。"""
        result = W.has_weights(np.array([1.0, 2.0]))
        self.assertIs(type(result), bool)
        self.assertIs(type(W.has_weights(None)), bool)
        self.assertIs(type(W.has_weights(np.array([0.0, 0.0]))), bool)


class NhanesAdapterTest(Base):
    def test_participant_count(self):
        self.assertEqual(builder.person_count(NH), 9254)

    def test_survey_design_columns(self):
        self.assertTrue(builder.has_sample_weights(NH))
        frame = builder.build_feature_frame(NH, [AGE])
        w = frame[builder.WEIGHT_COLUMN].to_numpy()
        self.assertEqual(len(w), 9254)
        self.assertGreater(float(np.nansum(w)), 0)

    def test_questionnaire_preserves_negatives_and_missing(self):
        """问卷存成 measurement 而非 condition，「否」和「没回答」必须分开。

        原始 DIQ010：是 893、否 7816、临界 184，其余是拒答/不知道/未问到。
        """
        frame = builder.build_feature_frame(NH, [DM_Q])
        values = frame[DM_Q].to_list()
        counts = {v: values.count(v) for v in ("是", "否", "临界")}
        self.assertEqual(counts, {"是": 893, "否": 7816, "临界": 184})
        answered = sum(counts.values())
        self.assertEqual(answered, 8893)
        # 361 人无有效作答，必须是缺失而不是「否」
        self.assertEqual(9254 - answered, 361)

    def test_refused_codes_are_not_data(self):
        """7 / 9（拒答 / 不知道）不能变成一个类别水平。"""
        levels = set(self.catalog[NH][DM_Q].levels or [])
        self.assertEqual(levels, {"是", "否", "临界"})

    def test_cholesterol_unit_normalised(self):
        raw = pd.read_sas(RAW / "TCHOL_J.xpt", format="xport")
        expected = round(float(raw["LBXTC"].dropna().iloc[0]) * 0.02586, 4)
        frame = builder.build_feature_frame(NH, [CHOL])
        values = [float(v) for v in frame[CHOL].to_list() if v is not None]
        self.assertIn(expected, values)
        self.assertEqual(self.catalog[NH][CHOL].unit, "mmol/L")

    def test_blood_pressure_zero_readings_treated_as_missing(self):
        """舒张压 0 表示未测得，不是真的 0。"""
        frame = builder.build_feature_frame(NH, ["measurement.LOINC:8462-4"])
        values = [float(v) for v in frame["measurement.LOINC:8462-4"].to_list() if v is not None]
        self.assertTrue(values)
        self.assertGreater(min(values), 0)

    def test_fasting_glucose_deliberately_absent(self):
        """GLU_J 用的是 WTSAF2YR 而非 WTMEC2YR，当前 CDM 装不下按变量选权重，
        所以刻意不导入 —— 导进来只会让人用错权重。"""
        codes = {v.id for v in builder.list_variables(NH)}
        self.assertNotIn("measurement.LOINC:2345-7", codes)


class PopulationEstimateTest(Base):
    """外部校验：与 NCHS 已发表的人群数字对照。"""

    def _adults(self):
        frame = builder.build_feature_frame(NH, [AGE, BMI])
        age = frame[AGE].cast(float).to_numpy()
        bmi = frame[BMI].cast(float, strict=False).to_numpy()
        w = frame[builder.WEIGHT_COLUMN].cast(float).to_numpy()
        mask = (age >= 20) & np.isfinite(bmi)
        return bmi[mask], w[mask]

    def test_obesity_prevalence_matches_nchs(self):
        """NCHS Data Brief 360：2017–2018 年美国成人肥胖率 42.4%。"""
        bmi, w = self._adults()
        estimate = W.proportion(bmi >= 30, w) * 100
        self.assertAlmostEqual(estimate, 42.4, delta=0.6)

    def test_severe_obesity_prevalence_matches_nchs(self):
        """同一报告：重度肥胖（BMI ≥ 40）9.2%。"""
        bmi, w = self._adults()
        estimate = W.proportion(bmi >= 40, w) * 100
        self.assertAlmostEqual(estimate, 9.2, delta=0.6)

    def test_weighting_actually_changes_the_answer(self):
        """如果加权和不加权结果一样，说明权重没生效，测试就没意义。"""
        bmi, w = self._adults()
        unweighted = float((bmi >= 30).mean()) * 100
        weighted = W.proportion(bmi >= 30, w) * 100
        self.assertNotAlmostEqual(unweighted, weighted, places=3)


class WeightingModeTest(Base):
    def test_auto_weights_survey_data(self):
        out = self.run_analysis("describe.baseline_table", variables=[AGE]).payload
        self.assertTrue(out["weighted"])
        self.assertTrue(out["weights_available"])
        self.assertIsNotNone(out["effective_n"])

    def test_auto_does_not_weight_plain_data(self):
        out = self.run_analysis(
            "describe.baseline_table", dataset=HF, variables=["person.age_at_index"]
        ).payload
        self.assertFalse(out["weighted"])
        self.assertFalse(out["weights_available"])

    def test_weighted_mode_suppresses_p_values(self):
        """加权点估计配未加权 p 值是自相矛盾的，宁可不出。"""
        out = self.run_analysis(
            "describe.baseline_table", variables=[AGE], group_by=SEX
        ).payload
        self.assertTrue(out["weighted"])
        self.assertTrue(all(r["p"] is None for r in out["rows"]))
        self.assertTrue(any("不出 p 值" in n for n in out["notes"]))

    def test_unweighted_mode_gives_p_values_with_a_warning(self):
        result = self.run_analysis(
            "describe.baseline_table", variables=[AGE], group_by=SEX, weighting="unweighted"
        )
        self.assertFalse(result.payload["weighted"])
        self.assertIsNotNone(result.payload["rows"][0]["p"])
        self.assertTrue(any("不能外推到人群" in w for w in result.warnings))

    def test_forced_weighting_on_plain_dataset_fails(self):
        with self.assertRaises(ValueError) as cm:
            self.run_analysis(
                "describe.baseline_table", dataset=HF,
                variables=["person.age_at_index"], weighting="weighted",
            )
        self.assertIn("没有抽样权重", str(cm.exception))

    def test_group_meta_carries_weighted_share(self):
        out = self.run_analysis(
            "describe.baseline_table", variables=[AGE], group_by=SEX
        ).payload
        shares = [g["weighted_pct"] for g in out["groups"]]
        self.assertTrue(all(s is not None for s in shares))
        self.assertAlmostEqual(sum(shares), 100.0, delta=0.2)

    def test_other_operators_warn_about_unhandled_weights(self):
        result = self.run_analysis("describe.distribution", variables=[BMI])
        self.assertTrue(any("未做加权" in w for w in result.warnings))

    def test_missingness_does_not_warn(self):
        """缺失率是样本事实，不涉及人群估计，不该刷警告。"""
        result = self.run_analysis("describe.missingness", variables=[BMI])
        self.assertEqual(result.warnings, [])


class PushdownParityTest(Base):
    """SQL 下推与 Python 求值必须给出同一批人。"""

    def _both(self, dataset, node):
        catalog = self.catalog[dataset]
        columns = filters.collect_variables(node)
        python_n = int(filters.evaluate(
            builder.build_feature_frame(dataset, columns), catalog, node
        ).sum())
        sql_n = builder.build_feature_frame(dataset, columns, where=node).height
        return python_n, sql_n

    def test_parity_across_operators(self):
        cases = [
            (NH, Condition(variable=AGE, op="between", value=[20, 80])),
            (NH, Condition(variable=AGE, op="gte", value=65)),
            (NH, Condition(variable=AGE, op="lt", value=18)),
            (NH, Condition(variable=BMI, op="gte", value=30)),
            (NH, Condition(variable=SEX, op="eq", value="F")),
            (NH, Condition(variable=SEX, op="ne", value="F")),
            (NH, Condition(variable=SEX, op="in", value=["F"])),
            (NH, Condition(variable=SEX, op="not_in", value=["F"])),
            (NH, Condition(variable=DM_Q, op="eq", value="是")),
            (NH, Condition(variable=DM_Q, op="ne", value="是")),
            (NH, Condition(variable=DM_Q, op="is_null")),
            (NH, Condition(variable=DM_Q, op="not_null")),
            (HF, Condition(variable="condition.HF:diabetes", op="eq", value="是")),
            (HF, Condition(variable="outcome.death", op="eq", value="是")),
            (HF, Condition(variable="condition.HF:diabetes", op="ne", value="是")),
            (UCI, Condition(variable="measurement.UCI:ca", op="is_null")),
            (UCI, Condition(variable="measurement.UCI:ca", op="ne", value="0.0")),
            (UCI, Condition(variable="measurement.UCI:thal", op="not_in", value=["正常"])),
        ]
        for dataset, node in cases:
            with self.subTest(dataset=dataset, op=node.op, var=node.variable):
                python_n, sql_n = self._both(dataset, node)
                self.assertEqual(python_n, sql_n)

    def test_parity_for_nested_trees(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="gte", value=20),
            Group(op="or", children=[
                Condition(variable=DM_Q, op="eq", value="是"),
                Condition(variable=BMI, op="gte", value=35),
            ]),
        ])
        python_n, sql_n = self._both(NH, node)
        self.assertEqual(python_n, sql_n)
        self.assertGreater(python_n, 0)

    def test_empty_group_pushes_nothing_down(self):
        frame = builder.build_feature_frame(NH, [AGE], where=Group(op="and", children=[]))
        self.assertEqual(frame.height, 9254)

    def test_pushdown_uses_bound_parameters(self):
        """取值必须走绑定参数，不能拼进 SQL 字符串。"""
        params: dict = {}
        clause = filters.to_sql(
            Condition(variable=SEX, op="eq", value="'; DROP TABLE person; --"),
            self.catalog[NH], params,
        )
        self.assertNotIn("DROP", clause)
        self.assertIn("'; DROP TABLE person; --", params.values())


class CacheTest(Base):
    def _wait(self, job):
        import time
        for _ in range(300):
            if job.status in ("succeeded", "failed"):
                return job
            time.sleep(0.02)
        raise TimeoutError

    def setUp(self):
        cache.clear()

    def test_second_run_hits_cache(self):
        params = {"variables": [AGE], "group_by": SEX}
        first = self._wait(runner.submit_analysis(NH, "describe.baseline_table", params))
        self.assertFalse(first.cached)
        second = self._wait(runner.submit_analysis(NH, "describe.baseline_table", params))
        self.assertTrue(second.cached)
        self.assertEqual(first.result, second.result)

    def test_cache_carries_cohort_counts(self):
        node = Condition(variable=AGE, op="gte", value=20)
        first = self._wait(runner.submit_analysis(
            NH, "describe.baseline_table", {"variables": [AGE]}, cohort=node))
        second = self._wait(runner.submit_analysis(
            NH, "describe.baseline_table", {"variables": [AGE]}, cohort=node))
        self.assertTrue(second.cached)
        self.assertEqual(first.cohort_n, second.cohort_n)
        self.assertEqual(second.cohort_n, 5569)

    def test_different_params_miss_cache(self):
        self._wait(runner.submit_analysis(NH, "describe.baseline_table", {"variables": [AGE]}))
        other = self._wait(runner.submit_analysis(
            NH, "describe.baseline_table", {"variables": [AGE, SEX]}))
        self.assertFalse(other.cached)

    def test_different_cohort_misses_cache(self):
        params = {"variables": [AGE]}
        self._wait(runner.submit_analysis(NH, "describe.baseline_table", params))
        filtered = self._wait(runner.submit_analysis(
            NH, "describe.baseline_table", params,
            cohort=Condition(variable=AGE, op="gte", value=20)))
        self.assertFalse(filtered.cached)

    def test_reimport_invalidates_cache(self):
        """数据重新导入后不能还拿着旧结论。"""
        params = {"variables": [AGE]}
        self._wait(runner.submit_analysis(HF, "describe.baseline_table", params))
        hit = self._wait(runner.submit_analysis(HF, "describe.baseline_table", params))
        self.assertTrue(hit.cached)

        etl.import_dataset(HF)  # 版本号（导入时间戳）变了
        after = self._wait(runner.submit_analysis(HF, "describe.baseline_table", params))
        self.assertFalse(after.cached)

    def test_corrupt_cache_file_is_recomputed(self):
        params = {"variables": [AGE]}
        self._wait(runner.submit_analysis(NH, "describe.baseline_table", params))
        for path in CACHE_DIR.glob("*.json"):
            path.write_text("{ 这不是合法 JSON", encoding="utf-8")
        again = self._wait(runner.submit_analysis(NH, "describe.baseline_table", params))
        self.assertEqual(again.status, "succeeded")
        self.assertFalse(again.cached)


class JobStateTest(Base):
    def test_cohort_counts_always_populated(self):
        """job 对象必须在任务开始前就传给工作线程，否则快任务会跑在回填之前。"""
        import time
        cache.clear()
        for i in range(5):
            job = runner.submit_analysis(
                NH, "describe.baseline_table",
                {"variables": [AGE], "normality_alpha": 0.05 - i * 0.001},
            )
            for _ in range(300):
                if job.status in ("succeeded", "failed"):
                    break
                time.sleep(0.02)
            self.assertEqual(job.status, "succeeded", job.error)
            self.assertEqual(job.cohort_n, 9254)
            self.assertEqual(job.cohort_total, 9254)


if __name__ == "__main__":
    unittest.main()
