"""NHANES 死亡关联文件。

引入它的原因很具体：审计时四个数据集里，「抽样权重」和「随访时间」两列
没有任何一行同时打勾 —— NHANES 有权重没随访，MIMIC 与 heart_failure 有随访
没权重。加权生存分析写出来只能靠合成数据验证，跑不了任何真实分析。
这个文件把 nhanes_2017 变成唯一同时具备两者的数据集。

期望值来自对原始 .dat 的独立核对（定宽布局逐列确认、按 ELIGSTAT 分组计数），
不是从流水线输出反抄的。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

from app import store
from app.adapters import base, nhanes  # noqa: F401  触发注册
from app.adapters.nhanes import MORTALITY_FILE
from app.cdm import etl, schema
from app.cohort import builder

DS = "nhanes_2017"

# 对原始文件独立核对出来的数字
TOTAL_ROWS = 9254          # 与 DEMO_J 的人数一致
ELIGIBLE = 5809            # ELIGSTAT=1
DEATHS_ALL = 145           # MORTSTAT=1
WITH_EXAM_FOLLOWUP = 5498  # ELIGSTAT=1 且 PERMTH_EXM 非缺失
DEATHS_WITH_EXAM = 130


def has_data() -> bool:
    return base.get(DS).is_available() and MORTALITY_FILE.is_file()


@unittest.skipUnless(has_data(), "需要 NHANES XPT 与死亡关联文件")
class RawFileTest(unittest.TestCase):
    """先确认定宽布局解对了，再谈导入。"""

    @classmethod
    def setUpClass(cls):
        cls.frame = base.get(DS)._read_mortality()

    def test_one_row_per_participant(self):
        self.assertEqual(len(self.frame), TOTAL_ROWS)

    def test_eligibility_breakdown(self):
        counts = self.frame["eligstat"].value_counts().to_dict()
        self.assertEqual(counts["1"], ELIGIBLE)      # 符合关联条件
        self.assertEqual(counts["2"], 3398)          # 未满 18 岁
        self.assertEqual(counts["3"], 47)            # 其他不符合

    def test_death_count(self):
        self.assertEqual(int((self.frame["mortstat"] == "1").sum()), DEATHS_ALL)

    def test_ineligible_have_no_followup(self):
        """不符合关联条件的人不该有任何随访或结局 —— 布局解错了这里会露馅。"""
        other = self.frame[self.frame["eligstat"] != "1"]
        self.assertEqual(set(other["mortstat"]), {"."})
        self.assertEqual(set(other["permth_exm"]), {"."})

    def test_interview_followup_covers_everyone_eligible(self):
        eligible = self.frame[self.frame["eligstat"] == "1"]
        self.assertTrue(eligible["permth_int"].str.isdigit().all())

    def test_exam_followup_is_missing_for_interview_only(self):
        """只完成访谈的人没有体检随访。他们的 MEC 权重恰好也是 0，两边自洽。"""
        eligible = self.frame[self.frame["eligstat"] == "1"]
        with_exam = eligible["permth_exm"].str.isdigit().sum()
        self.assertEqual(int(with_exam), WITH_EXAM_FOLLOWUP)
        self.assertEqual(ELIGIBLE - WITH_EXAM_FOLLOWUP, 311)


@unittest.skipUnless(has_data(), "需要 NHANES XPT 与死亡关联文件")
class ImportedOutcomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)

    def test_outcome_rows_only_for_followed_people(self):
        with store.read() as conn:
            n, events = conn.execute(
                "SELECT count(*), count(*) FILTER (WHERE is_event)"
                " FROM outcome WHERE dataset = ? AND event_type = 'death'", [DS]
            ).fetchone()
        self.assertEqual(n, WITH_EXAM_FOLLOWUP)
        self.assertEqual(events, DEATHS_WITH_EXAM)

    def test_followup_is_months_converted_to_days(self):
        """NCHS 只给到月，折成天用 30.4375 = 365.25 / 12。"""
        with store.read() as conn:
            low, high = conn.execute(
                "SELECT min(followup_days), max(followup_days)"
                " FROM outcome WHERE dataset = ?", [DS]
            ).fetchone()
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 37 * 30.4375, places=4)

    def test_variable_catalog_exposes_survival(self):
        catalog = {v.id: v for v in builder.list_variables(DS)}
        death = catalog["outcome.death"]
        self.assertTrue(death.has_survival)
        self.assertEqual(death.n_available, WITH_EXAM_FOLLOWUP)
        self.assertEqual(death.n_positive, DEATHS_WITH_EXAM)

    def test_dataset_now_has_both_weights_and_survival(self):
        """这正是引入这个文件的目的 —— 加权生存分析终于有数据可验证了。"""
        self.assertTrue(builder.has_survey_design(DS))
        self.assertTrue(any(v.has_survival for v in builder.list_variables(DS)))

    def test_no_outcome_row_for_people_outside_the_followed_population(self):
        """未满 18 岁的人不该出现在 outcome 表里 —— 给他们一条
        followup=NULL 的记录会把「不适用」和「结局缺失」混成一谈。"""
        with store.read() as conn:
            orphans = conn.execute(
                "SELECT count(*) FROM outcome o WHERE o.dataset = ?"
                " AND o.followup_days IS NULL", [DS]
            ).fetchone()[0]
        self.assertEqual(orphans, 0)


@unittest.skipUnless(has_data(), "需要 NHANES XPT 与死亡关联文件")
class SurvivalAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)
        from app.jobs import runner
        import app.analyses.survival  # noqa: F401
        cls.runner = runner

    def run_cox(self, **params):
        job = self.runner.wait(self.runner.submit_analysis(
            DS, "survival.cox",
            {"outcome": "outcome.death", **params}, None, None, "first"))
        self.assertEqual(job.status, "succeeded", job.error)
        return job

    def test_cox_gives_epidemiologically_sane_estimates(self):
        """年龄每大一岁风险升高、男性风险高于女性 —— 教科书结论。
        对不上说明数据接错了。"""
        job = self.run_cox(covariates=["person.age_at_index", "person.gender"])
        by_var = {r["variable"]: r for r in job.result["rows"]}
        age = by_var["person.age_at_index"]
        self.assertGreater(age["estimate"], 1.0)
        self.assertLess(age["p"], 0.001)
        male = by_var["person.gender"]
        self.assertGreater(male["estimate"], 1.0)

    def test_people_without_followup_are_reported_separately(self):
        """没有随访的人不是「缺失」—— 把他们算进缺失率会得出
        「41% 缺失，建议用多重插补」这条错误建议，而生存时间根本不能插补。"""
        job = self.run_cox(covariates=["person.age_at_index", "person.gender"])
        result = job.result
        self.assertGreater(result["n_no_followup"], 3000)
        self.assertEqual(result["n_missing_covariate"], 0)
        joined = " ".join(result["notes"])
        self.assertIn("不在被随访的人群里", joined)
        self.assertNotIn("可改用多重插补", " ".join(job.warnings))

    def test_unweighted_warning_still_fires(self):
        """算子还没做设计校正，就必须显眼地说明结论不能外推到人群。"""
        job = self.run_cox(covariates=["person.age_at_index"])
        self.assertTrue(any("未做加权" in w for w in job.warnings))

    def test_kaplan_meier_runs(self):
        job = self.runner.wait(self.runner.submit_analysis(
            DS, "survival.kaplan_meier",
            {"outcome": "outcome.death", "group_by": "person.gender"},
            None, None, "first"))
        self.assertEqual(job.status, "succeeded", job.error)
        series = job.result["series"]
        self.assertEqual({s["name"] for s in series}, {"F", "M"})
        self.assertEqual(sum(s["events"] for s in series), DEATHS_WITH_EXAM)


if __name__ == "__main__":
    unittest.main()
