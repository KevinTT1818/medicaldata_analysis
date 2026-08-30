"""多重插补（MICE）与 Rubin 合并规则的回归测试。

完全病例分析是设计文档里标为「结论会错」的坑，也是设计文档承诺过要提供
替代方案的地方。这里既测插补本身的性质，也测它接进 logistic 之后的行为。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

import numpy as np
import polars as pl

from app.adapters import heart_failure, mimic_demo, nhanes, uci_heart  # noqa: F401
from app.analyses import describe, imputation, regression, registry, survival  # noqa: F401
from app.analyses.base import AnalysisContext
from app.cdm import etl, schema
from app.cohort import builder
from app.cohort.filters import Condition

NH, UCI = "nhanes_2017", "uci_heart"
AGE = "person.age_at_index"
BMI = "measurement.LOINC:39156-5"
SBP = "measurement.LOINC:8480-6"
SEX = "person.gender"
HTN = "measurement.NHANES:hypertension"
CA = "measurement.UCI:ca"


class RubinPoolingTest(unittest.TestCase):
    """合并规则本身。不依赖数据集。"""

    def test_no_between_variance_degenerates_to_plain_inference(self):
        """m 次结果完全一致时，合并后就该等于单次结果。"""
        pooled = imputation.pool([1.0] * 5, [0.04] * 5)
        self.assertAlmostEqual(pooled.estimate, 1.0, places=12)
        self.assertAlmostEqual(pooled.se, 0.2, places=8)
        self.assertLess(pooled.fmi, 1e-6)

    def test_disagreement_widens_the_interval(self):
        """插补之间越不一致，区间越宽 —— 这正是它比单次填补诚实的地方。"""
        agree = imputation.pool([1.0] * 5, [0.04] * 5)
        disagree = imputation.pool([0.8, 1.0, 1.2, 0.9, 1.1], [0.04] * 5)
        self.assertGreater(disagree.se, agree.se)
        self.assertGreater(disagree.fmi, 0.1)
        self.assertGreater(disagree.ci_high - disagree.ci_low,
                           agree.ci_high - agree.ci_low)

    def test_fmi_grows_with_between_variance(self):
        variances = [0.04] * 5
        low = imputation.pool([1.0, 1.01, 0.99, 1.0, 1.0], variances)
        high = imputation.pool([0.5, 1.5, 0.7, 1.3, 1.0], variances)
        self.assertLess(low.fmi, high.fmi)

    def test_barnard_rubin_caps_degrees_of_freedom(self):
        """样本本身自由度有限时，合并不该给出虚高的自由度。"""
        estimates = [0.9, 1.0, 1.1, 0.95, 1.05]
        variances = [0.04] * 5
        unbounded = imputation.pool(estimates, variances)
        bounded = imputation.pool(estimates, variances, complete_df=10)
        self.assertLess(bounded.df, unbounded.df)
        self.assertLess(bounded.df, 10)

    def test_single_imputation_is_rejected(self):
        with self.assertRaises(imputation.ImputationError):
            imputation.pool([1.0], [0.04])


class MiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(NH)
        etl.import_dataset(UCI)
        cls.catalog = {
            ds: {v.id: v for v in builder.list_variables(ds)} for ds in (NH, UCI)
        }

    def _nhanes(self, variables):
        return builder.build_feature_frame(NH, variables)

    def test_no_missing_returns_original_frame(self):
        frame = self._nhanes([AGE])
        frames, report = imputation.impute(frame, self.catalog[NH], [AGE], m=5)
        self.assertTrue(report["skipped"])
        self.assertEqual(len(frames), 1)

    def test_imputation_fills_everything(self):
        variables = [AGE, BMI, SBP]
        frame = self._nhanes(variables)
        frames, _ = imputation.impute(frame, self.catalog[NH], variables, m=3, seed=0)
        for imputed in frames:
            for vid in variables:
                column = imputed[vid].cast(pl.Float64, strict=False).to_numpy()
                self.assertTrue(np.isfinite(column).all(), vid)

    def test_predictive_mean_matching_only_uses_real_observations(self):
        """PMM 的意义就在这里：补出来的一定是数据里出现过的值，不会补出负数年龄。"""
        variables = [AGE, BMI]
        frame = self._nhanes(variables)
        observed = frame[BMI].cast(pl.Float64, strict=False).to_numpy()
        missing = ~np.isfinite(observed)
        pool_of_values = set(observed[np.isfinite(observed)].tolist())

        frames, _ = imputation.impute(frame, self.catalog[NH], variables, m=2, seed=0)
        filled = frames[0][BMI].cast(pl.Float64, strict=False).to_numpy()[missing]
        self.assertTrue(filled.size > 0)
        for value in filled:
            self.assertIn(value, pool_of_values)

    def test_imputations_differ_from_each_other(self):
        """m 份完全相同就不是多重插补，也就算不出组间方差。"""
        variables = [AGE, BMI]
        frame = self._nhanes(variables)
        missing = ~np.isfinite(frame[BMI].cast(pl.Float64, strict=False).to_numpy())
        frames, _ = imputation.impute(frame, self.catalog[NH], variables, m=3, seed=0)
        filled = [f[BMI].cast(pl.Float64, strict=False).to_numpy()[missing] for f in frames]
        self.assertGreater(float(np.mean(filled[0] != filled[1])), 0.5)
        self.assertGreater(float(np.mean(filled[0] != filled[2])), 0.5)

    def test_imputation_conditions_on_other_variables(self):
        """插补值应随预测变量变化 —— 否则就只是按边缘分布瞎填。

        NHANES 的 BMI 随年龄单调上升，插补值也必须体现这一点。
        """
        variables = [AGE, BMI]
        frame = self._nhanes(variables)
        age = frame[AGE].cast(pl.Float64).to_numpy()
        missing = ~np.isfinite(frame[BMI].cast(pl.Float64, strict=False).to_numpy())

        frames, _ = imputation.impute(frame, self.catalog[NH], variables, m=1, seed=0)
        filled = frames[0][BMI].cast(pl.Float64, strict=False).to_numpy()

        children = missing & (age >= 2) & (age < 12)
        adults = missing & (age >= 20)
        self.assertTrue(children.any() and adults.any())
        self.assertLess(float(np.median(filled[children])),
                        float(np.median(filled[adults])))

    def test_categorical_imputation_keeps_level_set(self):
        variables = [AGE, HTN]
        frame = self._nhanes(variables)
        observed = {x for x in frame[HTN].to_list() if x is not None}
        frames, _ = imputation.impute(frame, self.catalog[NH], variables, m=2, seed=0)
        filled = set(frames[0][HTN].to_list())
        self.assertIsNotNone(observed)
        self.assertFalse(None in filled)
        self.assertTrue(filled <= observed)

    def test_reproducible_with_same_seed(self):
        variables = [AGE, BMI]
        frame = self._nhanes(variables)
        a, _ = imputation.impute(frame, self.catalog[NH], variables, m=2, seed=42)
        b, _ = imputation.impute(frame, self.catalog[NH], variables, m=2, seed=42)
        self.assertEqual(a[0][BMI].to_list(), b[0][BMI].to_list())

    def test_report_lists_imputed_columns(self):
        variables = [AGE, BMI, SBP]
        frame = self._nhanes(variables)
        _, report = imputation.impute(frame, self.catalog[NH], variables, m=2, seed=0)
        imputed = {c["variable"] for c in report["columns"]}
        self.assertIn(BMI, imputed)
        self.assertIn(SBP, imputed)
        self.assertNotIn(AGE, imputed)     # 年龄没有缺失
        # 缺失少的先补
        percentages = [c["pct"] for c in report["columns"]]
        self.assertEqual(percentages, sorted(percentages))


class LogisticWithImputationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(NH)
        cls.catalog = {v.id: v for v in builder.list_variables(NH)}

    def _run(self, **params):
        analysis = registry.get("regression.logistic")
        parsed = analysis.Params(**params)
        frame = builder.build_feature_frame(
            NH, analysis.required_variables(parsed),
            where=Condition(variable=AGE, op="gte", value=20),
        )
        return analysis.run(AnalysisContext(NH, frame, self.catalog, parsed))

    def test_imputation_keeps_more_rows_than_complete_case(self):
        covariates = [AGE, BMI, SBP, SEX]
        complete = self._run(outcome=HTN, covariates=covariates,
                             missing="complete_case").payload
        imputed = self._run(outcome=HTN, covariates=covariates,
                            missing="multiple_imputation", n_imputations=3).payload
        self.assertGreater(imputed["n_used"], complete["n_used"])
        self.assertLess(imputed["n_dropped"], complete["n_dropped"])

    def test_outcome_missing_rows_are_still_dropped(self):
        """插补协变量可以，插补结局不行 —— 那等于把答案编出来。"""
        imputed = self._run(outcome=HTN, covariates=[AGE, BMI],
                            missing="multiple_imputation", n_imputations=3).payload
        self.assertGreater(imputed["n_dropped"], 0)

    def test_fmi_tracks_missingness(self):
        """缺失越多的变量，其系数的缺失信息占比应当越高。"""
        out = self._run(outcome=HTN, covariates=[AGE, BMI, SBP],
                        missing="multiple_imputation", n_imputations=5).payload
        by_variable = {r["variable"]: r["fmi"] for r in out["rows"]}
        self.assertIsNotNone(by_variable[SBP])
        # SBP 缺 27%、BMI 缺 13.5%、年龄不缺
        self.assertGreater(by_variable[SBP], by_variable[BMI])
        self.assertGreater(by_variable[BMI], by_variable[AGE])

    def test_complete_case_reports_no_fmi(self):
        out = self._run(outcome=HTN, covariates=[AGE, BMI],
                        missing="complete_case").payload
        for row in out["rows"]:
            self.assertIsNone(row["fmi"])

    def test_imputation_notes_state_the_mar_assumption(self):
        """随机缺失是个假设，不是事实。用户必须看到它。"""
        out = self._run(outcome=HTN, covariates=[AGE, BMI],
                        missing="multiple_imputation", n_imputations=3).payload
        self.assertTrue(any("随机缺失" in n for n in out["notes"]))
        self.assertTrue(any("Rubin" in n for n in out["notes"]))

    def test_complete_case_warns_when_dropping_a_lot(self):
        result = self._run(outcome=HTN, covariates=[AGE, BMI, SBP, SEX],
                           missing="complete_case")
        self.assertTrue(any("选择偏倚" in w for w in result.warnings))

    def test_pseudo_r2_omitted_after_pooling(self):
        """合并后的伪 R² 没有标准定义，不给数字比给个错的强。"""
        out = self._run(outcome=HTN, covariates=[AGE, BMI],
                        missing="multiple_imputation", n_imputations=3).payload
        self.assertIsNone(out["pseudo_r2"])

    def test_estimates_stay_close_to_complete_case(self):
        """插补不该把结论翻过来 —— 缺失不算极端时两者应当接近。"""
        covariates = [AGE, BMI]
        complete = self._run(outcome=HTN, covariates=covariates,
                             missing="complete_case").payload
        imputed = self._run(outcome=HTN, covariates=covariates,
                            missing="multiple_imputation", n_imputations=5).payload
        for a, b in zip(complete["rows"], imputed["rows"]):
            self.assertAlmostEqual(a["estimate"], b["estimate"], delta=0.02)

    def test_no_missing_falls_back_to_single_fit(self):
        """本来就没有缺失时不该白跑 m 次。"""
        out = self._run(outcome=HTN, covariates=[AGE],
                        missing="multiple_imputation", n_imputations=5).payload
        self.assertIsNone(out["imputation"])
        for row in out["rows"]:
            self.assertIsNone(row["fmi"])


if __name__ == "__main__":
    unittest.main()
