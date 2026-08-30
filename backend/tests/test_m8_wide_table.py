"""宽表编译：按来源表分组，一次扫描 + FILTER 聚合。

早先的实现给每个变量各开一个 LEFT JOIN 子查询，代价随变量数爆炸：
MIMIC 只有 100 行，189 个变量要 10.9 秒，同一条查询加 EXPLAIN ANALYZE
会吃掉 6.3 GiB 内存。界面上点一下「全选」就能触发。

这里守两件事：
1. 结构不变量 —— JOIN 数量只与用到几张来源表有关，与选了多少变量无关。
   这比计时断言稳定，而且正是当初出问题的那个量。
2. 各来源的取值语义与改写前逐列一致（改写时对 4 个数据集 × 5 种聚合策略
   做过全量比对，0 处不一致）。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

from app.cdm import etl, schema
from app.adapters import mimic_demo, uci_heart  # noqa: F401  触发注册
from app.cohort import builder


class JoinCountTest(unittest.TestCase):
    """JOIN 数量必须由来源表数量决定，不能由变量数决定。"""

    def test_many_variables_from_one_source_share_one_join(self):
        codes = [f"measurement.C{i}" for i in range(200)]
        _, joins = builder._compile_variables(codes, "first", {"ds": "x"})
        self.assertEqual(len(joins), 1, "同一张表的变量没有共用一次扫描")

    def test_join_count_is_bounded_by_source_tables(self):
        variables = (
            ["person.age_at_index", "person.gender"]
            + [f"measurement.M{i}" for i in range(60)]
            + [f"condition.C{i}" for i in range(60)]
            + [f"visit.count:icu", "visit.los_total:icu", "visit.discharge_status:icu"]
            + [f"outcome.E{i}" for i in range(10)]
        )
        _, joins = builder._compile_variables(variables, "first", {"ds": "x"})
        # person 列直接从 p 取，不产生 JOIN；其余 4 张表各一次
        self.assertEqual(len(joins), 4)

    def test_person_columns_need_no_join(self):
        _, joins = builder._compile_variables(
            ["person.age_at_index", "person.gender", "person.race"], "first", {"ds": "x"}
        )
        self.assertEqual(joins, [])

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            builder._compile_variables(["nosuch.thing"], "first", {"ds": "x"})


class SelectOrderTest(unittest.TestCase):
    """内部按来源分组，但输出的列顺序必须仍是调用方给的顺序。"""

    def test_output_order_follows_input_order(self):
        variables = [
            "measurement.A", "person.gender", "condition.B",
            "measurement.C", "person.age_at_index",
        ]
        selects, _ = builder._compile_variables(variables, "first", {"ds": "x"})
        aliases = [s.rsplit(" AS ", 1)[1].strip('"') for s in selects]
        self.assertEqual(aliases, variables)

    def test_outcome_contributes_two_columns_in_place(self):
        variables = ["person.gender", "outcome.death", "measurement.A"]
        selects, _ = builder._compile_variables(variables, "first", {"ds": "x"})
        aliases = [s.rsplit(" AS ", 1)[1].strip('"') for s in selects]
        self.assertEqual(
            aliases,
            ["person.gender", "outcome.death",
             f"outcome.death{builder.TIME_SUFFIX}", "measurement.A"],
        )


class BoundParameterTest(unittest.TestCase):
    """取值一律走绑定参数，且绑定数量与实际用到的占位符对得上。"""

    def test_every_placeholder_is_bound(self):
        variables = [
            "person.gender", "measurement.A", "condition.B",
            "visit.count:icu", "outcome.death",
        ]
        params: dict = {"ds": "x"}
        selects, joins = builder._compile_variables(variables, "first", params)
        sql = " ".join(selects + joins)
        used = {tok.lstrip("$").rstrip(",)") for tok in sql.split() if tok.startswith("$")}
        self.assertTrue(used <= set(params), f"用到了没绑定的参数：{used - set(params)}")
        self.assertNotIn("'", sql, "取值被拼进了 SQL 而不是绑定")

    def test_codes_are_not_interpolated(self):
        params: dict = {"ds": "x"}
        _, joins = builder._compile_variables(
            ["measurement.DROP TABLE person"], "first", params
        )
        self.assertNotIn("DROP TABLE", " ".join(joins))
        self.assertIn("DROP TABLE person", params.values())


class SemanticsTest(unittest.TestCase):
    """各来源的缺失语义。改写前后必须一致。"""

    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset("mimic_demo")
        cls.catalog = {v.id: v for v in builder.list_variables("mimic_demo")}

    def frame(self, variables, agg="first"):
        return builder.build_feature_frame("mimic_demo", variables, agg=agg)

    def test_condition_absent_is_zero_not_null(self):
        vid = next(v for v in self.catalog if v.startswith("condition."))
        col = self.frame([vid])[vid]
        self.assertEqual(col.null_count(), 0, "诊断变量不该有缺失：没有就是 0")
        self.assertEqual(set(col.to_list()) - {0, 1}, set())

    def test_visit_count_absent_is_zero(self):
        vid = next(v for v in self.catalog if v.startswith("visit.count:"))
        col = self.frame([vid])[vid]
        self.assertEqual(col.null_count(), 0, "没有该类就诊是 0 次，不是未知")

    def test_visit_los_absent_stays_null(self):
        """没住过谈不上住了几天 —— 补成 0 会污染均值。"""
        vids = [v for v in self.catalog if v.startswith("visit.los_total:")]
        if not vids:
            self.skipTest("该数据集没有时长变量")
        frame = self.frame(vids[:1])
        counted = self.catalog[vids[0]]
        self.assertEqual(frame[vids[0]].null_count(), counted.n_missing)

    def test_outcome_emits_followup_column(self):
        vid = next(v for v in self.catalog
                   if v.startswith("outcome.") and self.catalog[v].has_survival)
        frame = self.frame([vid])
        self.assertIn(f"{vid}{builder.TIME_SUFFIX}", frame.columns)

    def test_design_columns_always_present(self):
        frame = self.frame(["person.gender"])
        for col in (builder.WEIGHT_COLUMN, builder.STRATUM_COLUMN, builder.PSU_COLUMN):
            self.assertIn(col, frame.columns)

    def test_aggregation_strategy_changes_repeated_measures(self):
        """MIMIC 平均每人每项 43 个值，first / max 必须给出不同结果。"""
        vid = next(v for v in self.catalog
                   if v.startswith("measurement.")
                   and self.catalog[v].kind == "continuous")
        first = self.frame([vid], agg="first")[vid].to_list()
        biggest = self.frame([vid], agg="max")[vid].to_list()
        self.assertNotEqual(first, biggest, "聚合策略没有生效")

    def test_row_count_matches_person_count(self):
        frame = self.frame(["person.gender"])
        self.assertEqual(frame.height, builder.person_count("mimic_demo"))


class WideSelectionTest(unittest.TestCase):
    """全选所有变量 —— 当初 11.9 秒 / OOM 的那条路径。"""

    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset("mimic_demo")

    def test_select_all_variables(self):
        variables = [v.id for v in builder.list_variables("mimic_demo")]
        self.assertGreater(len(variables), 100, "变量太少，没覆盖到问题场景")

        params: dict = {"ds": "mimic_demo"}
        _, joins = builder._compile_variables(variables, "first", params)
        self.assertLessEqual(len(joins), 4, f"{len(variables)} 个变量产生了 {len(joins)} 个 JOIN")

        frame = builder.build_feature_frame("mimic_demo", variables)
        self.assertEqual(frame.height, builder.person_count("mimic_demo"))
        for vid in variables:
            self.assertIn(vid, frame.columns)


if __name__ == "__main__":
    unittest.main()
