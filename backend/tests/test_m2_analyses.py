"""M2 算子回归测试：生存分析、回归、分布、设计矩阵、schema 迁移。

生存曲线的期望值用纯标准库按 KM 定义 S(t)=∏(1-d/n) 独立重算，
不是从 lifelines 输出反抄的。
"""
from __future__ import annotations

import csv
import unittest
from itertools import groupby

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

import numpy as np
import polars as pl

from app.adapters import heart_failure, uci_heart  # noqa: F401
from app.analyses import describe, regression, registry, survival  # noqa: F401
from app.analyses.base import AnalysisContext
from app.analyses.design import DesignError, build_design
from app.cdm import etl, schema
from app.cohort import builder

HF = "heart_failure"
UCI = "uci_heart"
SOURCE = _ROOT / "data" / "raw" / "heart_failure" / "heart_failure_clinical_records_dataset.csv"

AGE = "person.age_at_index"
SEX = "person.gender"
EF = "measurement.LOINC:10230-1"
CREAT = "measurement.LOINC:2160-0"
HTN = "condition.HF:hypertension"
DEATH = "outcome.death"


def _setup_datasets():
    schema.init()
    etl.import_dataset(HF)
    etl.import_dataset(UCI)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup_datasets()
        cls.catalog = {
            HF: {v.id: v for v in builder.list_variables(HF)},
            UCI: {v.id: v for v in builder.list_variables(UCI)},
        }

    def run_analysis(self, analysis_id, dataset=HF, **params):
        analysis = registry.get(analysis_id)
        parsed = analysis.Params(**params)
        frame = builder.build_feature_frame(dataset, analysis.required_variables(parsed))
        ctx = AnalysisContext(dataset, frame, self.catalog[dataset], parsed)
        return analysis.run(ctx)


class AdapterTest(Base):
    def test_row_counts(self):
        cat = self.catalog[HF]
        self.assertEqual(cat[AGE].n_available, 299)
        # anaemia 129 + diabetes 125 + hypertension 105 + smoking 96 = 455
        self.assertEqual(cat[HTN].n_available, 299)

    def test_age_is_not_truncated_to_int(self):
        """原始文件里有 60.667 这样的小数年龄，按 i64 推断会被静默截断。"""
        frame = builder.build_feature_frame(HF, [AGE])
        ages = frame[AGE].to_list()
        self.assertTrue(any(a != int(a) for a in ages), "小数年龄被截断了")

    def test_creatinine_unit_normalised(self):
        """mg/dL -> umol/L，系数 88.4。"""
        cat = self.catalog[HF]
        self.assertEqual(cat[CREAT].unit, "umol/L")
        frame = builder.build_feature_frame(HF, [CREAT])
        first = float(frame[CREAT][0])
        self.assertAlmostEqual(first, round(1.9 * 88.4, 4), places=4)

    def test_survival_flag(self):
        self.assertTrue(self.catalog[HF][DEATH].has_survival)
        # UCI Heart Disease 是横断面数据，没有随访时长
        self.assertFalse(self.catalog[UCI]["outcome.heart_disease"].has_survival)

    def test_existing_operators_work_on_new_dataset_unchanged(self):
        """架构主张：新增数据集只写 adapter，已有算子直接可用。"""
        out = self.run_analysis(
            "describe.baseline_table", variables=[AGE, SEX, EF], group_by=DEATH
        ).payload
        self.assertEqual([(g["label"], g["n"]) for g in out["groups"]],
                         [("否", 203), ("是", 96)])


class KaplanMeierTest(Base):
    @staticmethod
    def _manual_km():
        """按定义 S(t)=∏(1-d_i/n_i) 手工重算，只用标准库。"""
        with open(SOURCE) as f:
            rows = list(csv.DictReader(f))
        obs = sorted((float(r["time"]), int(r["DEATH_EVENT"])) for r in rows)
        at_risk, surv, out = len(obs), 1.0, {}
        for t, grp in groupby(obs, key=lambda x: x[0]):
            grp = list(grp)
            deaths = sum(e for _, e in grp)
            if deaths:
                surv *= 1 - deaths / at_risk
            out[t] = surv
            at_risk -= len(grp)
        return out

    def test_curve_matches_manual_calculation(self):
        out = self.run_analysis("survival.kaplan_meier", outcome=DEATH).payload
        series = out["series"][0]
        manual = self._manual_km()
        by_time = dict(zip(series["t"], series["survival"]))

        checked = 0
        for t, expected in manual.items():
            if t in by_time:
                self.assertAlmostEqual(by_time[t], expected, places=9,
                                       msg=f"t={t} 处生存率不符")
                checked += 1
        self.assertGreater(checked, 100, "核对的时点太少")

    def test_counts(self):
        series = self.run_analysis("survival.kaplan_meier", outcome=DEATH).payload["series"][0]
        self.assertEqual(series["n"], 299)
        self.assertEqual(series["events"], 96)
        self.assertEqual(series["censored"], 203)

    def test_median_not_reached_is_null(self):
        """超过半数存活时中位生存时间是 inf，必须序列化成 null 而不是 Infinity。"""
        series = self.run_analysis("survival.kaplan_meier", outcome=DEATH).payload["series"][0]
        self.assertIsNone(series["median_survival"])

    def test_censor_marks_align_with_curve(self):
        series = self.run_analysis("survival.kaplan_meier", outcome=DEATH).payload["series"][0]
        self.assertEqual(len(series["censor_t"]), len(series["censor_s"]))
        self.assertEqual(len(series["censor_t"]), 203)
        self.assertTrue(all(0 <= s <= 1 for s in series["censor_s"]))

    def test_ci_brackets_estimate(self):
        series = self.run_analysis("survival.kaplan_meier", outcome=DEATH).payload["series"][0]
        for lo, s, hi in zip(series["ci_lower"], series["survival"], series["ci_upper"]):
            self.assertLessEqual(lo, s + 1e-9)
            self.assertGreaterEqual(hi, s - 1e-9)

    def test_grouped_logrank(self):
        out = self.run_analysis("survival.kaplan_meier", outcome=DEATH, group_by=HTN).payload
        self.assertEqual([s["name"] for s in out["series"]], ["否", "是"])
        self.assertEqual([s["n"] for s in out["series"]], [194, 105])
        self.assertEqual(out["test"], "Log-rank 检验")
        self.assertAlmostEqual(out["logrank_p"], 0.0358, places=3)

    def test_at_risk_is_monotone_decreasing(self):
        out = self.run_analysis("survival.kaplan_meier", outcome=DEATH, group_by=HTN).payload
        for row in out["at_risk"]["rows"]:
            self.assertEqual(row["counts"], sorted(row["counts"], reverse=True))
            self.assertEqual(row["counts"][0], 194 if row["name"] == "否" else 105)

    def test_cross_sectional_outcome_is_rejected(self):
        """横断面数据没有随访时长，必须明确报错而不是返回空曲线。"""
        with self.assertRaises(ValueError) as cm:
            self.run_analysis("survival.kaplan_meier", dataset=UCI,
                              outcome="outcome.heart_disease")
        self.assertIn("没有随访时长", str(cm.exception))


class CoxTest(Base):
    def test_effect_directions_match_literature(self):
        """该数据集的已知结论：射血分数是保护因素，年龄与肌酐是危险因素。"""
        out = self.run_analysis("survival.cox", outcome=DEATH,
                                covariates=[AGE, EF, CREAT]).payload
        by_var = {r["variable"]: r for r in out["rows"]}
        self.assertGreater(by_var[AGE]["estimate"], 1.0)
        self.assertLess(by_var[EF]["estimate"], 1.0)
        self.assertGreater(by_var[CREAT]["estimate"], 1.0)
        for row in out["rows"]:
            self.assertLess(row["p"], 0.05)

    def test_confidence_interval_columns_resolve_at_any_level(self):
        """lifelines 的 CI 列名带置信水平（exp(coef) lower 90%），不能写死 95%。"""
        for level in (0.90, 0.95, 0.99):
            out = self.run_analysis("survival.cox", outcome=DEATH,
                                    covariates=[AGE], conf_level=level).payload
            row = out["rows"][0]
            self.assertLess(row["ci_lower"], row["estimate"])
            self.assertGreater(row["ci_upper"], row["estimate"])

    def test_wider_level_gives_wider_interval(self):
        narrow = self.run_analysis("survival.cox", outcome=DEATH,
                                   covariates=[AGE], conf_level=0.90).payload["rows"][0]
        wide = self.run_analysis("survival.cox", outcome=DEATH,
                                 covariates=[AGE], conf_level=0.99).payload["rows"][0]
        self.assertLess(wide["ci_lower"], narrow["ci_lower"])
        self.assertGreater(wide["ci_upper"], narrow["ci_upper"])

    def test_categorical_covariate_gets_reference_level(self):
        out = self.run_analysis("survival.cox", outcome=DEATH,
                                covariates=[EF, SEX]).payload
        sex_rows = [r for r in out["rows"] if r["variable"] == SEX]
        self.assertEqual(len(sex_rows), 1, "二水平变量应只产生一个哑变量列")
        self.assertEqual(sex_rows[0]["reference"], "F")
        self.assertIn("M", sex_rows[0]["label"])

    def test_ph_violation_is_reported(self):
        result = self.run_analysis("survival.cox", outcome=DEATH,
                                   covariates=[AGE, EF, CREAT, HTN], check_ph=True)
        self.assertTrue(result.payload["ph_test"])
        labels = {t["label"] for t in result.payload["ph_test"]}
        self.assertEqual(len(labels), 4)

    def test_ph_check_can_be_disabled(self):
        out = self.run_analysis("survival.cox", outcome=DEATH,
                                covariates=[AGE], check_ph=False).payload
        self.assertEqual(out["ph_test"], [])


class LogisticTest(Base):
    def test_or_and_roc(self):
        out = self.run_analysis("regression.logistic", outcome=DEATH,
                                covariates=[AGE, EF, CREAT]).payload
        self.assertEqual(out["effect_label"], "OR")
        self.assertEqual(out["n_used"], 299)
        self.assertEqual(out["n_events"], 96)
        by_var = {r["variable"]: r for r in out["rows"]}
        self.assertLess(by_var[EF]["estimate"], 1.0)
        self.assertGreater(by_var[AGE]["estimate"], 1.0)

    def test_auc_in_range_and_beats_chance(self):
        out = self.run_analysis("regression.logistic", outcome=DEATH,
                                covariates=[AGE, EF, CREAT]).payload
        auc = out["roc"]["auc"]
        self.assertGreater(auc, 0.5)
        self.assertLess(auc, 1.0)
        self.assertEqual(len(out["roc"]["fpr"]), len(out["roc"]["tpr"]))

    def test_roc_is_monotone(self):
        roc = self.run_analysis("regression.logistic", outcome=DEATH,
                                covariates=[AGE, EF]).payload["roc"]
        self.assertEqual(roc["fpr"], sorted(roc["fpr"]))
        self.assertEqual(roc["tpr"], sorted(roc["tpr"]))

    def test_works_on_cross_sectional_dataset(self):
        """logistic 不需要随访时长，横断面数据集应该能跑。"""
        out = self.run_analysis(
            "regression.logistic", dataset=UCI,
            outcome="outcome.heart_disease",
            covariates=["person.age_at_index", "person.gender"],
        ).payload
        self.assertEqual(out["n_used"], 303)


class DesignMatrixTest(Base):
    def test_single_level_covariate_rejected(self):
        frame = builder.build_feature_frame(HF, [SEX])
        # 只保留女性 -> 性别只剩一个水平
        female_only = frame.filter(pl.col(SEX) == "F")
        with self.assertRaises(DesignError) as cm:
            build_design(female_only, self.catalog[HF], [SEX])
        self.assertIn("只有一个水平", str(cm.exception))

    def test_empty_covariates_rejected(self):
        frame = builder.build_feature_frame(HF, [AGE])
        with self.assertRaises(DesignError):
            build_design(frame, self.catalog[HF], [])

    def test_complete_mask_flags_missing(self):
        """UCI 的 ca 列有 4 个缺失，掩码应把这 4 行标为不完整。"""
        cat = self.catalog[UCI]
        variables = ["person.age_at_index", "measurement.UCI:ca"]
        frame = builder.build_feature_frame(UCI, variables)
        _, _, complete = build_design(frame, cat, variables)
        self.assertEqual(int((~complete).sum()), 4)


class DistributionTest(Base):
    def test_continuous_panel(self):
        out = self.run_analysis("describe.distribution", variables=[EF], bins=20).payload
        panel = out["panels"][0]
        self.assertEqual(panel["kind"], "continuous")
        self.assertEqual(len(panel["bin_edges"]), 21)
        self.assertEqual(sum(panel["series"][0]["counts"]), 299)

    def test_box_stats_are_tukey(self):
        out = self.run_analysis("describe.distribution", variables=[EF]).payload
        box = out["panels"][0]["series"][0]["box"]
        iqr = box["q3"] - box["q1"]
        # 须线不得超出 1.5 倍四分位距
        self.assertGreaterEqual(box["min"], box["q1"] - 1.5 * iqr - 1e-9)
        self.assertLessEqual(box["max"], box["q3"] + 1.5 * iqr + 1e-9)
        for outlier in box["outliers"]:
            self.assertTrue(outlier < box["min"] or outlier > box["max"])

    def test_grouped_bins_are_shared(self):
        """各组必须共用同一套分箱边界，否则并排的直方图没法比较。"""
        out = self.run_analysis("describe.distribution", variables=[EF],
                                group_by=DEATH).payload
        panel = out["panels"][0]
        self.assertEqual(len(panel["series"]), 2)
        for s in panel["series"]:
            self.assertEqual(len(s["counts"]), len(panel["bin_edges"]) - 1)
        self.assertEqual(sum(sum(s["counts"]) for s in panel["series"]), 299)

    def test_categorical_panel(self):
        out = self.run_analysis("describe.distribution", variables=[SEX]).payload
        panel = out["panels"][0]
        self.assertEqual(panel["kind"], "categorical")
        self.assertEqual(panel["levels"], ["F", "M"])
        self.assertEqual(sum(panel["series"][0]["counts"]), 299)


class SchemaMigrationTest(unittest.TestCase):
    def test_missing_column_is_added(self):
        """CDM 演进后老库要能自动补列，不能让用户手工删库。"""
        from app import store

        schema.init()
        with store.write() as conn:
            conn.execute("ALTER TABLE outcome DROP COLUMN followup_days")
            remaining = {
                r[0] for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='main' AND table_name='outcome'"
                ).fetchall()
            }
        self.assertNotIn("followup_days", remaining)

        added = schema.migrate()
        self.assertIn("outcome.followup_days", added)

        with store.write() as conn:
            restored = {
                r[0] for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='main' AND table_name='outcome'"
                ).fetchall()
            }
        self.assertIn("followup_days", restored)

    def test_migrate_is_idempotent(self):
        schema.init()
        self.assertEqual(schema.migrate(), [])


class ParamsSchemaTest(unittest.TestCase):
    def test_every_operator_declares_widgets_for_variable_fields(self):
        """变量类字段必须带 x-widget，否则前端会渲染成普通文本框。"""
        for descriptor in registry.describe_all():
            properties = descriptor["params_schema"].get("properties", {})
            for name, spec in properties.items():
                if name in ("variables", "covariates", "group_by", "outcome"):
                    self.assertIn("x-widget", spec,
                                  f"{descriptor['id']}.{name} 缺少 x-widget")
                    self.assertIn("x-variable-filter", spec,
                                  f"{descriptor['id']}.{name} 缺少 x-variable-filter")

    def test_every_field_has_a_title(self):
        for descriptor in registry.describe_all():
            for name, spec in descriptor["params_schema"].get("properties", {}).items():
                self.assertIn("title", spec, f"{descriptor['id']}.{name} 缺少 title")


if __name__ == "__main__":
    unittest.main()
