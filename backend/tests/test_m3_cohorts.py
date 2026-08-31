"""M3 队列构建器回归测试：筛选求值、缺失语义、CONSORT 流程、持久化、接入分析。

期望人数用纯标准库读原始 CSV 独立重算，不从流水线反抄。
"""
from __future__ import annotations

import csv
import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

import numpy as np

from app.adapters import heart_failure, uci_heart  # noqa: F401
from app.analyses import describe, registry, survival  # noqa: F401
from app.cdm import etl, schema
from app.cohort import builder, filters, preview
from app.cohort import store as cohort_store
from app.cohort.filters import Condition, FilterError, Group
from app.jobs import runner

HF, UCI = "heart_failure", "uci_heart"
AGE = "person.age_at_index"
SEX = "person.gender"
EF = "measurement.LOINC:10230-1"
CA = "measurement.UCI:ca"
DM = "condition.HF:diabetes"
HTN = "condition.HF:hypertension"
DEATH = "outcome.death"

HF_CSV = _ROOT / "data" / "raw" / "heart_failure" / "heart_failure_clinical_records_dataset.csv"


def _raw_hf():
    with open(HF_CSV) as f:
        return list(csv.DictReader(f))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(HF)
        etl.import_dataset(UCI)
        cls.catalog = {
            HF: {v.id: v for v in builder.list_variables(HF)},
            UCI: {v.id: v for v in builder.list_variables(UCI)},
        }

    def count(self, node, dataset=HF) -> int:
        catalog = self.catalog[dataset]
        columns = filters.collect_variables(node)
        frame = builder.build_feature_frame(dataset, columns or [AGE])
        return int(filters.evaluate(frame, catalog, node).sum())


class OperatorTest(Base):
    def test_between_matches_manual_count(self):
        expected = sum(1 for r in _raw_hf() if 40 <= float(r["age"]) <= 80)
        got = self.count(Condition(variable=AGE, op="between", value=[40, 80]))
        self.assertEqual(got, expected)
        self.assertEqual(got, 281)

    def test_comparison_operators(self):
        rows = _raw_hf()
        cases = [
            ("gte", 60, sum(1 for r in rows if float(r["age"]) >= 60)),
            ("gt", 60, sum(1 for r in rows if float(r["age"]) > 60)),
            ("lte", 50, sum(1 for r in rows if float(r["age"]) <= 50)),
            ("lt", 50, sum(1 for r in rows if float(r["age"]) < 50)),
            ("eq", 60, sum(1 for r in rows if float(r["age"]) == 60)),
            ("ne", 60, sum(1 for r in rows if float(r["age"]) != 60)),
        ]
        for op, value, expected in cases:
            with self.subTest(op=op):
                self.assertEqual(
                    self.count(Condition(variable=AGE, op=op, value=value)), expected
                )

    def test_binary_condition(self):
        expected = sum(1 for r in _raw_hf() if r["diabetes"] == "1")
        self.assertEqual(self.count(Condition(variable=DM, op="eq", value="是")), expected)
        self.assertEqual(expected, 125)

    def test_categorical_in(self):
        node = Condition(variable=SEX, op="in", value=["F"])
        expected = sum(1 for r in _raw_hf() if r["sex"] == "0")
        self.assertEqual(self.count(node), expected)

    def test_not_in_excludes_listed_levels(self):
        node = Condition(variable=SEX, op="not_in", value=["F"])
        expected = sum(1 for r in _raw_hf() if r["sex"] == "1")
        self.assertEqual(self.count(node), expected)

    def test_operator_rejected_for_wrong_kind(self):
        with self.assertRaises(FilterError) as cm:
            self.count(Condition(variable=AGE, op="in", value=["x"]))
        self.assertIn("连续型", str(cm.exception))

    def test_reversed_range_rejected(self):
        with self.assertRaises(FilterError) as cm:
            self.count(Condition(variable=AGE, op="between", value=[80, 40]))
        self.assertIn("下界大于上界", str(cm.exception))

    def test_missing_value_rejected(self):
        with self.assertRaises(FilterError):
            self.count(Condition(variable=AGE, op="gte", value=None))

    def test_unknown_variable_rejected(self):
        frame = builder.build_feature_frame(HF, [AGE])
        with self.assertRaises(FilterError) as cm:
            filters.evaluate(frame, self.catalog[HF],
                             Condition(variable="person.nope", op="eq", value=1))
        self.assertIn("没有变量", str(cm.exception))


class MissingSemanticsTest(Base):
    """缺失值语义：除 is_null / not_null 外一律判 False。"""

    def test_is_null_and_not_null(self):
        # UCI 的 ca 列有 4 个缺失
        self.assertEqual(self.count(Condition(variable=CA, op="is_null"), UCI), 4)
        self.assertEqual(self.count(Condition(variable=CA, op="not_null"), UCI), 299)

    def test_missing_fails_equality(self):
        eq = self.count(Condition(variable=CA, op="eq", value="0.0"), UCI)
        ne = self.count(Condition(variable=CA, op="ne", value="0.0"), UCI)
        # eq + ne 不等于全量 —— 缺失的 4 例两边都不算
        self.assertEqual(eq + ne, 303 - 4)

    def test_missing_fails_not_in(self):
        """not_in 遇到缺失也判 False：无法确认的人不进队列。"""
        node = Condition(variable=CA, op="not_in", value=["0.0"])
        got = self.count(node, UCI)
        in_count = self.count(Condition(variable=CA, op="in", value=["0.0"]), UCI)
        self.assertEqual(got + in_count, 303 - 4)


class NestingTest(Base):
    def test_and_is_intersection(self):
        rows = _raw_hf()
        expected = sum(
            1 for r in rows
            if 40 <= float(r["age"]) <= 80 and r["diabetes"] == "1"
        )
        node = Group(op="and", children=[
            Condition(variable=AGE, op="between", value=[40, 80]),
            Condition(variable=DM, op="eq", value="是"),
        ])
        self.assertEqual(self.count(node), expected)

    def test_or_is_union(self):
        rows = _raw_hf()
        expected = sum(
            1 for r in rows if r["diabetes"] == "1" or r["high_blood_pressure"] == "1"
        )
        node = Group(op="or", children=[
            Condition(variable=DM, op="eq", value="是"),
            Condition(variable=HTN, op="eq", value="是"),
        ])
        self.assertEqual(self.count(node), expected)

    def test_nested_and_or(self):
        rows = _raw_hf()
        expected = sum(
            1 for r in rows
            if float(r["age"]) >= 60
            and (r["diabetes"] == "1" or r["high_blood_pressure"] == "1")
        )
        node = Group(op="and", children=[
            Condition(variable=AGE, op="gte", value=60),
            Group(op="or", children=[
                Condition(variable=DM, op="eq", value="是"),
                Condition(variable=HTN, op="eq", value="是"),
            ]),
        ])
        self.assertEqual(self.count(node), expected)

    def test_empty_group_keeps_everyone(self):
        self.assertEqual(self.count(Group(op="and", children=[])), 299)


class ConsortFlowTest(Base):
    def _flow(self, node, dataset=HF):
        columns = filters.collect_variables(node) or [AGE]
        frame = builder.build_feature_frame(dataset, columns)
        return filters.consort_flow(frame, self.catalog[dataset], node)

    def test_sequential_steps_for_top_level_and(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="between", value=[40, 80]),
            Condition(variable=DM, op="eq", value="是"),
        ])
        flow = self._flow(node)
        self.assertTrue(flow["sequential"])
        self.assertEqual(len(flow["steps"]), 2)
        self.assertEqual(flow["steps"][0]["n_before"], 299)
        self.assertEqual(flow["steps"][0]["n_after"], 281)
        self.assertEqual(flow["steps"][1]["n_before"], 281)
        self.assertEqual(flow["steps"][1]["n_after"], 121)
        self.assertEqual(flow["final_n"], 121)

    def test_step_labels_are_readable(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="between", value=[40, 80]),
        ])
        label = self._flow(node)["steps"][0]["label"]
        self.assertIn("年龄", label)
        self.assertIn("40", label)
        self.assertIn("岁", label)

    def test_missing_exclusions_are_attributed(self):
        """因变量缺失被排除的人要单独计数，否则用户以为是条件不满足。"""
        node = Group(op="and", children=[
            Condition(variable=CA, op="eq", value="0.0"),
        ])
        step = self._flow(node, UCI)["steps"][0]
        self.assertEqual(step["n_excluded_missing"], 4)
        self.assertGreater(step["n_excluded"], step["n_excluded_missing"])

    def test_or_root_is_not_sequential(self):
        node = Group(op="or", children=[
            Condition(variable=DM, op="eq", value="是"),
            Condition(variable=HTN, op="eq", value="是"),
        ])
        flow = self._flow(node)
        self.assertFalse(flow["sequential"])
        self.assertEqual(len(flow["steps"]), 1)

    def test_empty_group_has_no_steps(self):
        flow = self._flow(Group(op="and", children=[]))
        self.assertEqual(flow["steps"], [])
        self.assertTrue(flow["sequential"])
        self.assertEqual(flow["final_n"], 299)


class PreviewTest(Base):
    def test_preview_counts_and_summary(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="between", value=[40, 80]),
            Condition(variable=DM, op="eq", value="是"),
        ])
        out = preview.preview(HF, node)
        self.assertEqual(out["total"], 299)
        self.assertEqual(out["n"], 121)
        self.assertAlmostEqual(out["pct"], 40.5, places=1)

        by_var = {s["variable"]: s for s in out["summary"]}
        self.assertEqual(by_var[SEX]["kind"], "categorical")
        self.assertEqual(sum(l["n"] for l in by_var[SEX]["levels"]), 121)
        self.assertEqual(by_var[AGE]["kind"], "continuous")
        self.assertEqual(by_var[AGE]["n"], 121)

    def test_preview_without_filter(self):
        out = preview.preview(HF, None)
        self.assertEqual(out["n"], out["total"])
        self.assertEqual(out["pct"], 100.0)

    def test_unknown_dataset(self):
        with self.assertRaises(KeyError):
            preview.preview("nope", None)


class CohortStoreTest(Base):
    def test_save_load_update_delete(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="gte", value=60),
        ])
        saved = cohort_store.save(HF, "老年组", node, description="60 岁及以上")
        self.assertTrue(saved["id"])
        self.assertEqual(saved["name"], "老年组")

        loaded = cohort_store.load_node(saved["id"])
        self.assertEqual(self.count(loaded), self.count(node))

        updated = cohort_store.save(
            HF, "老年组（改）",
            Group(op="and", children=[Condition(variable=AGE, op="gte", value=70)]),
            cohort_id=saved["id"],
        )
        self.assertEqual(updated["id"], saved["id"])
        self.assertEqual(updated["name"], "老年组（改）")
        self.assertEqual(len(cohort_store.list_for(HF)), 1)

        cohort_store.delete(saved["id"])
        self.assertEqual(cohort_store.list_for(HF), [])

    def test_definition_survives_round_trip(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="between", value=[40, 80]),
            Group(op="or", children=[
                Condition(variable=DM, op="eq", value="是"),
                Condition(variable=SEX, op="in", value=["F"]),
            ]),
        ])
        saved = cohort_store.save(HF, "嵌套", node)
        try:
            self.assertEqual(self.count(cohort_store.load_node(saved["id"])),
                             self.count(node))
        finally:
            cohort_store.delete(saved["id"])

    def test_update_unknown_cohort_raises(self):
        with self.assertRaises(KeyError):
            cohort_store.save(HF, "x", Group(op="and", children=[]), cohort_id="nope")


class AnalysisWithCohortTest(Base):
    def _wait(self, job):
        import time
        for _ in range(200):
            if job.status in ("succeeded", "failed"):
                return job
            time.sleep(0.02)
        raise TimeoutError

    def test_cohort_narrows_analysis(self):
        node = Group(op="and", children=[
            Condition(variable=AGE, op="between", value=[40, 80]),
            Condition(variable=DM, op="eq", value="是"),
        ])
        job = self._wait(runner.submit_analysis(
            HF, "describe.baseline_table", {"variables": [AGE]}, cohort=node
        ))
        self.assertEqual(job.status, "succeeded", job.error)
        self.assertEqual(job.cohort_n, 121)
        self.assertEqual(job.cohort_total, 299)
        self.assertEqual(job.result["overall_n"], 121)

    def test_fingerprint_changes_with_cohort(self):
        params = {"variables": [AGE]}
        plain = runner.submit_analysis(HF, "describe.baseline_table", params)
        filtered = runner.submit_analysis(
            HF, "describe.baseline_table", params,
            cohort=Condition(variable=AGE, op="gte", value=60),
        )
        self.assertNotEqual(
            plain.spec["fingerprint"], filtered.spec["fingerprint"]
        )

    def test_same_cohort_same_fingerprint(self):
        params = {"variables": [AGE]}
        node = Condition(variable=AGE, op="gte", value=60)
        a = runner.submit_analysis(HF, "describe.baseline_table", params, cohort=node)
        b = runner.submit_analysis(HF, "describe.baseline_table", params, cohort=node)
        self.assertEqual(a.spec["fingerprint"], b.spec["fingerprint"])

    def test_empty_cohort_fails_clearly(self):
        job = self._wait(runner.submit_analysis(
            HF, "describe.baseline_table", {"variables": [AGE]},
            cohort=Condition(variable=AGE, op="gt", value=500),
        ))
        self.assertEqual(job.status, "failed")
        self.assertIn("一个人都没有", job.error)

    def test_filter_variable_not_in_analysis_still_available(self):
        """筛选用到的变量不在分析变量里时，也要被取进宽表。"""
        job = self._wait(runner.submit_analysis(
            HF, "describe.baseline_table", {"variables": [SEX]},
            cohort=Condition(variable=EF, op="lt", value=40),
        ))
        self.assertEqual(job.status, "succeeded", job.error)
        self.assertLess(job.cohort_n, 299)

    def test_cohort_applies_to_survival(self):
        job = self._wait(runner.submit_analysis(
            HF, "survival.kaplan_meier", {"outcome": DEATH},
            cohort=Condition(variable=DM, op="eq", value="是"),
        ))
        self.assertEqual(job.status, "succeeded", job.error)
        self.assertEqual(job.result["series"][0]["n"], 125)


if __name__ == "__main__":
    unittest.main()
