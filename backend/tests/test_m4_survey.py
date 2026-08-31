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

    def test_weighted_mode_uses_design_based_test(self):
        """加权模式的 p 值来自设计校正 Wald 检验，不是未加权的 t 检验。

        （早先的实现在加权模式下不出 p 值，因为方差估计还没做。现在做了。）
        """
        out = self.run_analysis(
            "describe.baseline_table", variables=[AGE], group_by=SEX
        ).payload
        self.assertTrue(out["weighted"])
        row = out["rows"][0]
        self.assertIsNotNone(row["p"])
        self.assertEqual(row["test"], "设计校正 Wald 检验")
        self.assertTrue(any("Taylor 线性化" in n for n in out["notes"]))

    def test_weighted_mode_reports_design_parameters(self):
        out = self.run_analysis(
            "describe.baseline_table", variables=[AGE], group_by=SEX
        ).payload
        design = out["design"]
        # NHANES 2017–2018：15 层 × 2 PSU，自由度 15，与 NCHS 官方一致
        self.assertEqual(design["n_strata"], 15)
        self.assertEqual(design["n_psu"], 30)
        self.assertEqual(design["df"], 15)
        self.assertFalse(design["approximate"])

    def test_weighted_continuous_rows_report_means(self):
        """加权时一律报均值 —— 检验比的是均值，显示中位数读者对不上。"""
        out = self.run_analysis(
            "describe.baseline_table", variables=[AGE, BMI], group_by=SEX
        ).payload
        for row in out["rows"]:
            if row["kind"] == "continuous":
                self.assertEqual(row["stat"], "mean_sd")

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


class TaylorLinearisationTest(unittest.TestCase):
    """方差估计量本身的数学性质。不依赖具体数据集。"""

    def _simple(self, n=200, seed=0):
        import numpy as np
        from app.analyses import survey
        rng = np.random.default_rng(seed)
        y = rng.normal(10, 3, n)
        return y, survey.build_design(np.ones(n), None, None)

    def test_degenerates_to_ordinary_se_without_design(self):
        """等权、单层、每人一个 PSU —— 应当退化成普通标准误。"""
        import numpy as np
        from app.analyses import survey
        y, design = self._simple()
        est = survey.domain_mean(y, design)
        self.assertAlmostEqual(est.value, float(np.mean(y)), places=10)
        ordinary = float(np.std(y, ddof=1) / np.sqrt(y.size))
        self.assertAlmostEqual(est.se, ordinary, places=6)
        self.assertAlmostEqual(est.deff, 1.0, delta=0.02)

    def test_clustering_inflates_variance(self):
        """把观测塞进少数几个 PSU，方差必须变大 —— 这正是忽略聚类会低估的东西。"""
        import numpy as np
        from app.analyses import survey
        rng = np.random.default_rng(1)
        n = 400
        # 20 个 PSU，PSU 内有共同的随机效应 -> 组内相关
        psu = np.repeat(np.arange(20), n // 20)
        y = rng.normal(0, 1, n) + np.repeat(rng.normal(0, 2, 20), n // 20)
        w = np.ones(n)

        clustered = survey.domain_mean(
            y, survey.build_design(w, np.zeros(n), psu))
        independent = survey.domain_mean(y, survey.build_design(w, None, None))
        self.assertGreater(clustered.se, independent.se * 2)
        self.assertGreater(clustered.deff, 3.0)

    def test_degrees_of_freedom_is_psu_minus_strata(self):
        import numpy as np
        from app.analyses import survey
        strata = np.repeat(np.arange(5), 40)
        psu = np.tile(np.repeat(np.arange(2), 20), 5)
        design = survey.build_design(np.ones(200), strata, psu)
        self.assertEqual(design.n_strata, 5)
        self.assertEqual(design.n_psu, 10)
        self.assertEqual(design.df, 5)

    def test_singleton_strata_are_counted(self):
        import numpy as np
        from app.analyses import survey
        strata = np.array([0, 0, 1, 1, 2, 2])
        psu = np.array([0, 1, 0, 1, 0, 0])   # 第 2 层只有一个 PSU
        design = survey.build_design(np.ones(6), strata, psu)
        self.assertEqual(design.singleton_strata(), 1)

    def test_domain_estimation_keeps_all_psus(self):
        """域估计不能先切数据 —— 切掉会让某些 PSU 消失，方差就不对了。

        这里验证：域内均值等于把域外权重置零后的加权均值，
        但方差是在完整设计上算的（自由度不变）。
        """
        import numpy as np
        from app.analyses import survey
        rng = np.random.default_rng(2)
        n = 300
        strata = np.repeat(np.arange(3), 100)
        psu = np.tile(np.repeat(np.arange(2), 50), 3)
        y = rng.normal(5, 2, n)
        w = rng.uniform(1, 5, n)
        domain = rng.random(n) < 0.5

        design = survey.build_design(w, strata, psu)
        est = survey.domain_mean(y, design, domain)

        expected = float((w[domain] * y[domain]).sum() / w[domain].sum())
        self.assertAlmostEqual(est.value, expected, places=10)
        self.assertEqual(est.df, design.df)

    def test_wald_test_detects_a_real_difference(self):
        import numpy as np
        from app.analyses import survey
        rng = np.random.default_rng(3)
        n = 400
        strata = np.repeat(np.arange(10), 40)
        psu = np.tile(np.repeat(np.arange(2), 20), 10)
        group = rng.random(n) < 0.5
        y = rng.normal(0, 1, n) + group * 2.0        # 组间差 2 个标准差
        design = survey.build_design(np.ones(n), strata, psu)
        out = survey.wald_test(y, design, [group, ~group])
        self.assertIsNotNone(out["p"])
        self.assertLess(out["p"], 0.001)

    def test_wald_test_accepts_no_difference(self):
        import numpy as np
        from app.analyses import survey
        rng = np.random.default_rng(4)
        n = 400
        strata = np.repeat(np.arange(10), 40)
        psu = np.tile(np.repeat(np.arange(2), 20), 10)
        group = rng.random(n) < 0.5
        y = rng.normal(0, 1, n)                      # 组间无差异
        design = survey.build_design(np.ones(n), strata, psu)
        out = survey.wald_test(y, design, [group, ~group])
        self.assertGreater(out["p"], 0.05)

    def test_wald_refuses_when_contrasts_exceed_df(self):
        """对比数超过设计自由度时该检验不可估计，必须明说而不是给个数字。"""
        import numpy as np
        from app.analyses import survey
        n = 60
        strata = np.repeat(np.arange(2), 30)
        psu = np.tile(np.repeat(np.arange(2), 15), 2)   # df = 4 - 2 = 2
        design = survey.build_design(np.ones(n), strata, psu)
        categories = np.array(["a", "b", "c", "d"] * 15)
        groups = [np.arange(n) % 3 == k for k in range(3)]
        out = survey.categorical_wald_test(
            categories, ["a", "b", "c", "d"], design, groups)
        self.assertIsNone(out["p"])
        self.assertIn("超过设计自由度", out["detail"])


class NhanesStandardErrorTest(Base):
    """与 NCHS 官方口径对照。"""

    def _adult_design(self):
        import numpy as np
        from app.analyses import survey
        frame = builder.build_feature_frame(NH, [AGE, BMI])
        age = frame[AGE].cast(float).to_numpy()
        bmi = frame[BMI].cast(float, strict=False).to_numpy()
        design = survey.build_design(
            frame[builder.WEIGHT_COLUMN].cast(float).to_numpy(),
            frame[builder.STRATUM_COLUMN].to_numpy(),
            frame[builder.PSU_COLUMN].to_numpy(),
        )
        return bmi, design, (age >= 20) & np.isfinite(bmi)

    def test_design_parameters_match_nchs(self):
        _, design, _ = self._adult_design()
        self.assertEqual(design.n_strata, 15)
        self.assertEqual(design.n_psu, 30)
        self.assertEqual(design.df, 15)
        self.assertEqual(design.singleton_strata(), 0)

    def test_obesity_confidence_interval_covers_published_figure(self):
        """NCHS Data Brief 360：2017–2018 年美国成人肥胖率 42.4%。"""
        from app.analyses import survey
        bmi, design, adults = self._adult_design()
        est = survey.domain_proportion((bmi >= 30).astype(float), design, adults)
        self.assertAlmostEqual(est.value * 100, 42.4, delta=0.6)
        self.assertLess(est.ci_low * 100, 42.4)
        self.assertGreater(est.ci_high * 100, 42.4)
        self.assertEqual(est.df, 15)

    def test_design_se_is_much_larger_than_srs_se(self):
        """忽略聚类会低估标准误 —— 这正是必须做设计校正的理由。"""
        import numpy as np
        from app.analyses import survey
        bmi, design, adults = self._adult_design()
        est = survey.domain_proportion((bmi >= 30).astype(float), design, adults)

        n = int(adults.sum())
        srs_se = float(np.sqrt(est.value * (1 - est.value) / n))
        self.assertGreater(est.se, srs_se * 2)
        self.assertGreater(est.deff, 3.0)


class SurveyLogitTest(unittest.TestCase):
    """加权 logistic 的伪极大似然与三明治方差。"""

    @staticmethod
    def _sample(n=500, seed=7):
        import numpy as np
        rng = np.random.default_rng(seed)
        X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        eta = X @ np.array([-0.5, 1.2, -0.8])
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        return y, X

    def test_point_estimates_match_statsmodels_when_unweighted(self):
        """等权时伪极大似然就是普通极大似然，系数必须一致到机器精度。"""
        import numpy as np
        import statsmodels.api as sm
        from app.analyses import survey

        y, X = self._sample()
        fit = survey.weighted_logit(y, X, survey.build_design(np.ones(y.size), None, None))
        reference = sm.Logit(y, X).fit(disp=0)
        for i in range(X.shape[1]):
            self.assertAlmostEqual(fit.beta[i], reference.params[i], places=10)
        self.assertTrue(fit.converged)

    def test_sandwich_se_close_to_model_se_without_clustering(self):
        """无聚类时三明治是稳健估计量，与模型 SE 应当接近但不必相同。"""
        import numpy as np
        import statsmodels.api as sm
        from app.analyses import survey

        y, X = self._sample()
        fit = survey.weighted_logit(y, X, survey.build_design(np.ones(y.size), None, None))
        reference = sm.Logit(y, X).fit(disp=0)
        for i in range(X.shape[1]):
            self.assertAlmostEqual(fit.se[i] / reference.bse[i], 1.0, delta=0.15)

    def test_clustering_inflates_se_but_not_estimates(self):
        """簇内相关抬高的是方差，不是系数 —— 忽略它就会高估显著性。

        构造方式：自变量是个体级的，但结局上叠一个 PSU 层面的共同冲击。
        这是最典型的情形（同一社区的人共享未观测的环境因素）。
        簇级冲击结构上作用在截距，所以截距的方差膨胀是稳健的（这里 2 倍以上）；
        个体级协变量的斜率受影响小得多，且随样本随机性有正有负 ——
        所以这里只断言截距，不对斜率下结论。
        """
        import numpy as np
        from app.analyses import survey

        rng = np.random.default_rng(11)
        n, n_psu = 600, 30
        per_psu = n // n_psu
        psu = np.repeat(np.arange(n_psu), per_psu)

        x = rng.normal(0, 1, n)
        shock = np.repeat(rng.normal(0, 1.5, n_psu), per_psu)
        eta = -0.3 + 0.8 * x + shock
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
        X = np.column_stack([np.ones(n), x])
        w = np.ones(n)

        clustered = survey.weighted_logit(
            y, X, survey.build_design(w, np.zeros(n), psu))
        independent = survey.weighted_logit(
            y, X, survey.build_design(w, None, None))

        # 点估计只取决于权重，与聚类结构无关
        for i in range(2):
            self.assertAlmostEqual(clustered.beta[i], independent.beta[i], places=12)

        self.assertGreater(clustered.se[0], independent.se[0] * 2.0)

    def test_weights_change_the_point_estimate(self):
        """加权不只影响区间，点估计本身就会变。"""
        import numpy as np
        from app.analyses import survey

        rng = np.random.default_rng(13)
        n = 400
        x = rng.normal(0, 1, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x)))).astype(float)
        X = np.column_stack([np.ones(n), x])
        # 让 x 大的人权重高 -> 加权后的人群里 x 的分布不同
        w = np.exp(x)

        equal = survey.weighted_logit(y, X, survey.build_design(np.ones(n), None, None))
        skewed = survey.weighted_logit(y, X, survey.build_design(w, None, None))
        self.assertNotAlmostEqual(equal.beta[1], skewed.beta[1], places=3)

    def test_singular_information_matrix_is_reported(self):
        import numpy as np
        from app.analyses import survey

        n = 100
        x = np.linspace(-1, 1, n)
        X = np.column_stack([np.ones(n), x, x])   # 完全共线
        y = (x > 0).astype(float)
        with self.assertRaises(survey.LogitError):
            survey.weighted_logit(y, X, survey.build_design(np.ones(n), None, None))

    def test_weighted_auc_degenerates_to_unweighted(self):
        import numpy as np
        from app.analyses import regression, survey

        rng = np.random.default_rng(17)
        n = 300
        y = (rng.random(n) < 0.4).astype(float)
        score = rng.random(n) + y * 0.3
        weighted = survey.weighted_auc(y, score, np.ones(n))
        plain = regression._roc(y, score)
        self.assertAlmostEqual(weighted["auc"], plain["auc"], places=10)

    def test_weighted_auc_responds_to_weights(self):
        import numpy as np
        from app.analyses import survey

        rng = np.random.default_rng(19)
        n = 300
        y = (rng.random(n) < 0.5).astype(float)
        score = rng.random(n)
        equal = survey.weighted_auc(y, score, np.ones(n))
        skewed = survey.weighted_auc(y, score, 1 + 9 * score)
        self.assertNotAlmostEqual(equal["auc"], skewed["auc"], places=3)


class NhanesLogitTest(Base):
    def _run(self, dataset=NH, **params):
        analysis = registry.get("regression.logistic")
        parsed = analysis.Params(**params)
        frame = builder.build_feature_frame(dataset, analysis.required_variables(parsed))
        return analysis.run(AnalysisContext(dataset, frame, self.catalog[dataset], parsed))

    def test_weighted_path_reports_design(self):
        out = self._run(outcome="measurement.NHANES:hypertension",
                        covariates=[AGE, BMI, SEX]).payload
        self.assertTrue(out["weighted"])
        self.assertEqual(out["design"]["df"], 15)
        self.assertEqual(out["design"]["n_strata"], 15)
        self.assertFalse(out["design"]["approximate"])
        self.assertTrue(any("三明治方差" in n for n in out["notes"]))

    def test_pseudo_r2_omitted_when_weighted(self):
        """伪 R² 在加权拟合下没有对应的标准定义，不给数字比给个错的强。"""
        out = self._run(outcome="measurement.NHANES:hypertension",
                        covariates=[AGE, BMI]).payload
        self.assertIsNone(out["pseudo_r2"])

    def test_plain_dataset_keeps_model_based_path(self):
        out = self._run(dataset=HF, outcome="outcome.death",
                        covariates=["person.age_at_index"]).payload
        self.assertFalse(out["weighted"])
        self.assertIsNone(out["design"])
        self.assertIsNotNone(out["pseudo_r2"])

    def test_no_unweighted_warning_when_design_is_applied(self):
        """既然做了设计校正，就不该再提示「未做加权」。"""
        result = self._run(outcome="measurement.NHANES:hypertension", covariates=[AGE])
        self.assertFalse(any("未做加权" in w for w in result.warnings))

    def test_effect_directions_are_clinically_sensible(self):
        out = self._run(outcome="measurement.NHANES:hypertension",
                        covariates=[AGE, BMI]).payload
        by_variable = {r["variable"]: r for r in out["rows"]}
        # 年龄与 BMI 都是高血压的危险因素
        self.assertGreater(by_variable[AGE]["estimate"], 1.0)
        self.assertGreater(by_variable[BMI]["estimate"], 1.0)
        for row in out["rows"]:
            self.assertLess(row["ci_lower"], row["estimate"])
            self.assertGreater(row["ci_upper"], row["estimate"])
