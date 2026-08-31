"""回归测试。用 `.venv/bin/python -m unittest discover -s tests` 跑。

这里的期望值全部来自对原始文件 processed.cleveland.data 的独立手工核对，
不是从流水线输出反抄的 —— 否则测试只能证明代码没变，不能证明代码算对。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

from app.analyses import describe, registry  # noqa: F401
from app.analyses.base import AnalysisContext
from app.cdm import etl, schema, units
from app.adapters import uci_heart  # noqa: F401  触发注册
from app.cohort import builder

DS = "uci_heart"


class UnitConversionTest(unittest.TestCase):
    def test_cholesterol_factor_is_not_prerounded(self):
        """系数本身不能舍入 —— 0.02586 舍成 0.0259 会给每个值带来 0.15% 的偏差。"""
        f, unit = units.factor("mg/dL", "LOINC:2093-3")
        self.assertEqual(unit, "mmol/L")
        self.assertAlmostEqual(f, 0.02586, places=10)

    def test_known_conversion(self):
        value, unit = units.convert(233.0, "mg/dL", "LOINC:2093-3")
        self.assertEqual(unit, "mmol/L")
        self.assertAlmostEqual(value, 6.0254, places=4)

    def test_unregistered_code_passes_through(self):
        value, unit = units.convert(150.0, "bpm", "UCI:thalach")
        self.assertEqual((value, unit), (150.0, "bpm"))


class CdmImportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)
        cls.catalog = {v.id: v for v in builder.list_variables(DS)}

    def test_person_count(self):
        self.assertEqual(self.catalog["person.age_at_index"].n_available, 303)

    def test_missing_counts_match_source(self):
        """原始文件里 ca 有 4 个 '?'，thal 有 2 个。"""
        self.assertEqual(self.catalog["measurement.UCI:ca"].n_missing, 4)
        self.assertEqual(self.catalog["measurement.UCI:thal"].n_missing, 2)

    def test_kind_inference(self):
        self.assertEqual(self.catalog["person.age_at_index"].kind, "continuous")
        self.assertEqual(self.catalog["measurement.UCI:thalach"].kind, "continuous")
        # ca 是 0-3 支血管，唯一值少且为整数，应判为分类
        self.assertEqual(self.catalog["measurement.UCI:ca"].kind, "categorical")
        self.assertEqual(self.catalog["condition.UCI:exang"].kind, "binary")

    def test_categorical_measurement_uses_value_text(self):
        thal = self.catalog["measurement.UCI:thal"]
        self.assertEqual(set(thal.levels or []), {"正常", "固定缺损", "可逆缺损"})


class BaselineTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)
        cls.catalog = {v.id: v for v in builder.list_variables(DS)}

    def _run(self, variables, group_by=None, p_adjust="fdr_bh"):
        analysis = registry.get("describe.baseline_table")
        params = analysis.Params(
            variables=variables, group_by=group_by, p_adjust=p_adjust
        )
        frame = builder.build_feature_frame(DS, analysis.required_variables(params))
        ctx = AnalysisContext(DS, frame, self.catalog, params)
        return analysis.run(ctx).payload

    def test_group_sizes(self):
        out = self._run(["person.age_at_index"], group_by="outcome.heart_disease")
        self.assertEqual(out["overall_n"], 303)
        self.assertEqual([(g["label"], g["n"]) for g in out["groups"]],
                         [("否", 164), ("是", 139)])

    def test_gender_counts(self):
        out = self._run(["person.gender"], group_by="outcome.heart_disease")
        levels = {lv["label"]: lv["cells"] for lv in out["rows"][0]["levels"]}
        self.assertEqual(levels["F"]["__overall__"], "97 (32.0)")
        self.assertEqual(levels["F"]["否"], "72 (43.9)")
        self.assertEqual(levels["F"]["是"], "25 (18.0)")
        self.assertEqual(levels["M"]["__overall__"], "206 (68.0)")

    def test_age_is_nonparametric_and_matches_hand_calc(self):
        """年龄在两组内均不通过正态性检验，应走 Mann-Whitney 并报中位数(IQR)。"""
        out = self._run(["person.age_at_index"], group_by="outcome.heart_disease")
        row = out["rows"][0]
        self.assertEqual(row["stat"], "median_iqr")
        self.assertEqual(row["test"], "Mann-Whitney U")
        self.assertEqual(row["cells"]["否"], "52.0 (44.8–59.0)")
        self.assertEqual(row["cells"]["是"], "58.0 (52.0–62.0)")

    def test_group_by_variable_excluded_from_rows(self):
        out = self._run(
            ["person.age_at_index", "outcome.heart_disease"],
            group_by="outcome.heart_disease",
        )
        self.assertEqual([r["variable"] for r in out["rows"]], ["person.age_at_index"])

    def test_fdr_correction_is_monotone_and_larger(self):
        out = self._run(
            ["person.age_at_index", "person.gender", "measurement.LOINC:2093-3",
             "condition.UCI:fbs_high"],
            group_by="outcome.heart_disease",
        )
        for row in out["rows"]:
            self.assertIsNotNone(row["p_adj"])
            self.assertGreaterEqual(row["p_adj"], row["p"])

    def test_no_correction_leaves_p_adj_null(self):
        out = self._run(["person.age_at_index"], group_by="outcome.heart_disease",
                        p_adjust="none")
        self.assertIsNone(out["rows"][0]["p_adj"])

    def test_ungrouped_has_no_p_values(self):
        out = self._run(["person.age_at_index"])
        self.assertEqual(out["groups"], [])
        self.assertIsNone(out["rows"][0]["p"])

    def test_fisher_used_for_sparse_2x2(self):
        """期望频数 < 5 的 2x2 表应改用 Fisher 精确检验。"""
        analysis = registry.get("describe.baseline_table")
        params = analysis.Params(variables=["condition.UCI:fbs_high"],
                                 group_by="outcome.heart_disease")
        frame = builder.build_feature_frame(DS, analysis.required_variables(params))
        # 人为制造稀疏格子：只留前 12 例
        ctx = AnalysisContext(DS, frame.head(12), self.catalog, params)
        row = analysis.run(ctx).payload["rows"][0]
        self.assertEqual(row["test"], "Fisher 精确检验")


class MissingnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)
        cls.catalog = {v.id: v for v in builder.list_variables(DS)}

    def test_counts_and_complete_cases(self):
        analysis = registry.get("describe.missingness")
        variables = ["measurement.UCI:ca", "measurement.UCI:thal", "person.age_at_index"]
        params = analysis.Params(variables=variables)
        frame = builder.build_feature_frame(DS, variables)
        out = analysis.run(AnalysisContext(DS, frame, self.catalog, params)).payload

        by_var = {r["variable"]: r for r in out["rows"]}
        self.assertEqual(by_var["measurement.UCI:ca"]["n_missing"], 4)
        self.assertEqual(by_var["measurement.UCI:thal"]["n_missing"], 2)
        self.assertEqual(by_var["person.age_at_index"]["n_missing"], 0)
        # 6 行有缺失，且 ca 与 thal 的缺失不重叠 -> 完整记录 297
        self.assertEqual(out["complete_cases"], 297)


if __name__ == "__main__":
    unittest.main()
