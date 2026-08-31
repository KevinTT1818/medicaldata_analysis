"""pandas.read_sas 把 XPT 里的数值 0 读成次正规数 5.397605346934028e-79。

它既不等于 0 也大于 0，于是好几处「等于 0」「大于 0」的判断静默失效。
影响最大的一处：BPX_J 里 0 表示未测得，`_mean_bp` 用 replace(0, nan) 去掉它，
但匹配不上，81 条未测得的舒张压被当成真实读数平均了进去 ——
均值从 69.5 掉到 68.4，偏低 1.2 mmHg。

判据不是猜的：RIDAGEYR 为该值的 357 人，月龄 RIDAGEMN 全在 1–11 之间，
是未满 1 岁的婴儿，真值就是 0；而 RIDAGEYR=1 的人月龄正好 12–23。
"""
from __future__ import annotations

import unittest

from _env import ROOT as _ROOT  # noqa: F401  必须最先导入：它负责设好数据目录

import numpy as np

from app import store
from app.cdm import etl, schema
from app.adapters import nhanes  # noqa: F401  触发注册

DS = "nhanes_2017"
DENORMAL = 5.397605346934028e-79


def available() -> bool:
    from app.adapters import base
    return base.get(DS).is_available()


@unittest.skipUnless(available(), "需要 data/raw/nhanes 下的 XPT 文件")
class ZeroNormalisationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema.init()
        etl.import_dataset(DS)

    def query(self, sql: str):
        with store.read() as conn:
            return conn.execute(sql, [DS]).fetchone()

    def test_no_denormals_survive_import(self):
        """库里不该再有这个次正规数。"""
        for table, column in (("person", "age_at_index"),
                              ("person", "sample_weight"),
                              ("measurement", "value_num")):
            with self.subTest(column=f"{table}.{column}"):
                n = self.query(
                    f"SELECT count(*) FROM {table} WHERE dataset = ? "
                    f"AND {column} > 0 AND {column} < 1e-70")[0]
                self.assertEqual(n, 0)

    def test_infants_have_age_zero_not_missing(self):
        """未满 1 岁的婴儿年龄是 0，不是缺失，也不是 5.4e-79。"""
        low, zeros, missing = self.query(
            "SELECT min(age_at_index), count(*) FILTER (WHERE age_at_index = 0),"
            " count(*) FILTER (WHERE age_at_index IS NULL)"
            " FROM person WHERE dataset = ?")
        self.assertEqual(low, 0.0)
        self.assertEqual(zeros, 357)
        self.assertEqual(missing, 0)

    def test_interview_only_participants_have_zero_mec_weight(self):
        """只完成访谈没参加体检的人，MEC 权重是 0 —— weights._clean 靠
        `w > 0` 把他们排除在加权估计之外，5.4e-79 会让这个排除失效。"""
        zeros = self.query(
            "SELECT count(*) FROM person WHERE dataset = ? AND sample_weight = 0")[0]
        self.assertEqual(zeros, 550)

    def test_unmeasured_diastolic_is_missing_not_zero(self):
        """BPX 里 0 表示未测得。留在数据里会把均值拉低约 1.2 mmHg。"""
        n, mean, low = self.query(
            "SELECT count(*), avg(value_num), min(value_num) FROM measurement"
            " WHERE dataset = ? AND code = 'LOINC:8462-4'")
        self.assertEqual(n, 6636)
        self.assertGreater(low, 1.0, "未测得的 0 值还留在数据里")
        self.assertAlmostEqual(mean, 69.525, places=2)

    def test_systolic_is_unaffected(self):
        """收缩压没有 0 读数，是对照组 —— 修复不该动到它。"""
        n, mean = self.query(
            "SELECT count(*), avg(value_num) FROM measurement"
            " WHERE dataset = ? AND code = 'LOINC:8480-6'")
        self.assertEqual(n, 6717)
        self.assertAlmostEqual(mean, 121.544, places=2)


@unittest.skipUnless(available(), "需要 data/raw/nhanes 下的 XPT 文件")
class ReadHelperTest(unittest.TestCase):
    def test_read_returns_exact_zeros(self):
        from app.adapters.base import get

        frame = get(DS)._read("DEMO_J")
        weight = frame["WTMEC2YR"].to_numpy()
        self.assertEqual(int((weight == 0).sum()), 550)
        self.assertEqual(int(((weight > 0) & (weight < 1e-70)).sum()), 0)

    def test_denormal_threshold_is_far_below_real_values(self):
        """阈值要远低于任何合法取值，不能误伤真实的小数值。"""
        from app.adapters.base import get

        adapter = get(DS)
        self.assertLess(adapter.ZERO_EPS, 1e-60)
        frame = adapter._read("BMX_J")
        bmi = frame["BMXBMI"].to_numpy()
        real = bmi[np.isfinite(bmi) & (bmi > 0)]
        self.assertGreater(real.min(), 1.0, "合法取值不该接近阈值")


if __name__ == "__main__":
    unittest.main()
