"""M5 报告回归测试。

设计文档给 M5 定的验收标准是「把一份三个月前的报告重跑一遍，数字完全一致」。
一个永远说「一致」的检查器毫无价值，所以这里既测正例，也**故意改动底层数据**
测负例 —— 数字真变了必须被抓出来。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

from app.adapters import heart_failure, mimic_demo, nhanes, uci_heart  # noqa: F401
from app.analyses import describe, regression, registry, survival  # noqa: F401
from app.cdm import etl, schema
from app.cohort.filters import Condition
from app.jobs import cache
from app.report import runner as report_runner
from app.report import store as report_store
from app.report.models import ReportInput, ReportSection

HF = "heart_failure"
AGE = "person.age_at_index"
SEX = "person.gender"
EF = "measurement.LOINC:10230-1"
DEATH = "outcome.death"


def _sections() -> list[ReportSection]:
    return [
        ReportSection(
            title="基线特征", note="按死亡结局分组",
            dataset=HF, analysis="describe.baseline_table",
            params={"variables": [AGE, SEX, EF], "group_by": DEATH},
        ),
        ReportSection(
            title="生存曲线",
            dataset=HF, analysis="survival.kaplan_meier",
            params={"outcome": DEATH},
        ),
    ]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(HF)

    def setUp(self):
        cache.clear()
        for r in report_store.list_all():
            report_store.delete(r["id"])

    def _save(self, sections=None, title="测试报告"):
        payload = ReportInput(title=title, sections=sections or _sections())
        outputs = [report_runner.run_section(s) for s in payload.sections]
        for o in outputs:
            self.assertEqual(o["status"], "succeeded", o.get("error"))
        from app.report.models import SectionSnapshot
        snapshot = [SectionSnapshot(**o["snapshot"]) for o in outputs]
        return report_store.save(payload, snapshot=snapshot)


class PersistenceTest(Base):
    def test_save_and_load(self):
        saved = self._save()
        self.assertTrue(saved["id"])
        self.assertEqual(len(saved["sections"]), 2)
        self.assertEqual(len(saved["snapshot"]), 2)

        loaded = report_store.load_sections(saved["id"])
        self.assertEqual([s.title for s in loaded], ["基线特征", "生存曲线"])
        self.assertEqual(loaded[0].note, "按死亡结局分组")

    def test_report_stores_definition_not_numbers(self):
        """报告存的是分析定义。存数字就没法重跑，也没法回答「现在还成立吗」。"""
        saved = self._save()
        blob = str(saved["sections"])
        self.assertIn("describe.baseline_table", blob)
        self.assertIn(AGE, blob)
        # 不该出现算出来的结果内容（cells / p_adj / at_risk 都是结果里的键）
        for key in ("cells", "p_adj", "at_risk", "effective_n", "logrank_p"):
            self.assertNotIn(key, blob, f"报告里存了结果内容：{key}")

    def test_update_keeps_id(self):
        saved = self._save()
        payload = ReportInput(title="改名了", sections=_sections()[:1])
        updated = report_store.save(payload, report_id=saved["id"])
        self.assertEqual(updated["id"], saved["id"])
        self.assertEqual(updated["title"], "改名了")
        self.assertEqual(len(updated["sections"]), 1)

    def test_delete(self):
        saved = self._save()
        report_store.delete(saved["id"])
        self.assertEqual(report_store.list_all(), [])

    def test_update_unknown_report_raises(self):
        with self.assertRaises(KeyError):
            report_store.save(ReportInput(title="x"), report_id="nope")


class ReproducibilityTest(Base):
    def test_rerun_matches_baseline(self):
        """正例：定义和数据都没变，逐节必须完全一致。"""
        saved = self._save()
        run = report_runner.run_report(saved["id"])
        self.assertTrue(run["reproducible"])
        self.assertTrue(run["has_baseline"])
        for c in run["comparisons"]:
            self.assertEqual(c["verdict"], "match", c["detail"])

    def test_reimport_without_data_change_still_matches(self):
        """重新导入同样的源文件，版本号变了但数字没变 —— 应判为一致。"""
        saved = self._save()
        etl.import_dataset(HF)
        run = report_runner.run_report(saved["id"])
        self.assertTrue(run["reproducible"])

    def test_changed_data_is_detected(self):
        """负例：底层数据真变了，必须被抓出来并说明变在哪。

        直接从 CDM 删掉一批人，模拟「数据集更新后队列变小」。
        """
        from app import store

        saved = self._save()
        cache.clear()
        with store.write() as conn:
            conn.execute(
                """DELETE FROM person WHERE dataset = ? AND person_id IN (
                     SELECT person_id FROM person WHERE dataset = ? LIMIT 40)""",
                [HF, HF],
            )
        try:
            run = report_runner.run_report(saved["id"])
            self.assertFalse(run["reproducible"], "数据少了 40 人却说可复现")
            changed = [c for c in run["comparisons"] if c["verdict"] == "changed"]
            self.assertEqual(len(changed), 2)
            for c in changed:
                self.assertIn("纳入例数", c["detail"])
                self.assertEqual(c["baseline"]["n"], 299)
                self.assertEqual(c["current"]["n"], 259)
        finally:
            etl.import_dataset(HF)  # 恢复，免得影响后续测试

    def test_changed_definition_is_detected(self):
        """定义被改动（换了队列）也要判为不一致。"""
        saved = self._save()
        cache.clear()
        narrowed = _sections()
        narrowed[0].cohort = Condition(variable=AGE, op="gte", value=60)
        report_store.save(
            ReportInput(title="测试报告", sections=narrowed), report_id=saved["id"]
        )
        run = report_runner.run_report(saved["id"])
        self.assertFalse(run["reproducible"])
        first = run["comparisons"][0]
        self.assertEqual(first["verdict"], "changed")
        self.assertIn("分析定义变了", first["detail"])

    def test_no_baseline_is_reported_not_silently_passed(self):
        payload = ReportInput(title="没有基线", sections=_sections())
        saved = report_store.save(payload)  # 不传 snapshot
        run = report_runner.run_report(saved["id"])
        self.assertFalse(run["has_baseline"])
        for c in run["comparisons"]:
            self.assertEqual(c["verdict"], "no_baseline")

    def test_rebaseline_makes_it_match_again(self):
        from app import store
        from app.report.models import SectionSnapshot

        saved = self._save()
        cache.clear()
        with store.write() as conn:
            conn.execute(
                """DELETE FROM person WHERE dataset = ? AND person_id IN (
                     SELECT person_id FROM person WHERE dataset = ? LIMIT 40)""",
                [HF, HF],
            )
        try:
            self.assertFalse(report_runner.run_report(saved["id"])["reproducible"])

            outputs = [report_runner.run_section(s)
                       for s in report_store.load_sections(saved["id"])]
            report_store.update_snapshot(
                saved["id"], [SectionSnapshot(**o["snapshot"]) for o in outputs]
            )
            self.assertTrue(report_runner.run_report(saved["id"])["reproducible"])
        finally:
            etl.import_dataset(HF)

    def test_failed_section_does_not_claim_reproducible(self):
        broken = [ReportSection(
            title="跑不通的一节", dataset=HF, analysis="survival.kaplan_meier",
            params={"outcome": DEATH},
        )]
        payload = ReportInput(title="含坏节", sections=broken)
        outputs = [report_runner.run_section(s) for s in broken]
        from app.report.models import SectionSnapshot
        saved = report_store.save(
            payload, snapshot=[SectionSnapshot(**outputs[0]["snapshot"])]
        )
        # 把结局换成一个不存在的变量，让这一节必然失败
        broken[0].params = {"outcome": "outcome.does_not_exist"}
        report_store.save(ReportInput(title="含坏节", sections=broken),
                          report_id=saved["id"])
        run = report_runner.run_report(saved["id"])
        self.assertFalse(run["reproducible"])
        self.assertEqual(run["comparisons"][0]["verdict"], "failed")


class ResultHashTest(unittest.TestCase):
    def test_hash_is_order_independent(self):
        a = {"x": 1, "y": [1, 2], "z": {"p": 1, "q": 2}}
        b = {"z": {"q": 2, "p": 1}, "y": [1, 2], "x": 1}
        self.assertEqual(report_runner.result_hash(a), report_runner.result_hash(b))

    def test_hash_changes_with_numbers(self):
        a = {"value": 1.0}
        b = {"value": 1.0000001}
        self.assertNotEqual(report_runner.result_hash(a), report_runner.result_hash(b))

    def test_list_order_matters(self):
        """列表顺序是数据的一部分，不该被当成相同。"""
        self.assertNotEqual(
            report_runner.result_hash({"t": [1, 2]}),
            report_runner.result_hash({"t": [2, 1]}),
        )


class ExtractNTest(unittest.TestCase):
    def test_reads_overall_n(self):
        self.assertEqual(report_runner._extract_n({"overall_n": 42}), 42)

    def test_reads_model_n(self):
        self.assertEqual(report_runner._extract_n({"n_used": 17}), 17)

    def test_sums_survival_series(self):
        payload = {"series": [{"n": 10}, {"n": 5}]}
        self.assertEqual(report_runner._extract_n(payload), 15)

    def test_returns_none_when_absent(self):
        self.assertIsNone(report_runner._extract_n({"kind": "whatever"}))


if __name__ == "__main__":
    unittest.main()
