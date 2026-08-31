"""NHANES 2017–2018（J 周期）适配器。

原始文件：data/raw/nhanes/*.xpt（SAS XPORT 格式，用 pandas.read_sas 解析）
9254 名受访者，其中 8704 人完成了体检（MEC）。多文件按 SEQN 主键合并。

三个 NHANES 特有的坑，处理不好结论直接是错的：

1. **抽样权重**。NHANES 是复杂抽样设计，不同人群的抽样概率不同。直接求均值得到的
   是样本均值，不是人群估计。WTMEC2YR 存进 person.sample_weight，
   SDMVPSU / SDMVSTRA 存进 psu / stratum。

2. **不同子样本用不同权重**。禁食血糖（GLU_J）用的是 WTSAF2YR 而非 WTMEC2YR。
   当前 CDM 每人只存一个权重，装不下这种「按变量选权重」的语义，
   所以**本适配器不导入 GLU_J** —— 导进来只会让人用错权重、得到静默错误的人群估计。
   要支持它得在 measurement 上加一列指明该用哪个权重，那是后话。

3. **问卷的特殊编码**。7 = 拒答、9 = 不知道（长问卷里是 77/99、777/999），
   当成数据用会把「不知道」算成一个诊断类别。这里统一映射为缺失。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..cdm import units
from ..config import RAW_DIR
from .base import Adapter, CdmTables, register

SOURCE = RAW_DIR / "nhanes"
CYCLE = "J"  # 2017–2018

#: NCHS 公开版死亡关联文件（定宽 ASCII）。可选 —— 没有它其余部分照常工作。
MORTALITY_FILE = SOURCE / "NHANES_2017_2018_MORT_2019_PUBLIC.dat"

#: 定宽布局，1 起始的闭区间。取自 NCHS 随文件发布的 SAS 读入语句，
#: 并对着实际文件逐列核对过（哪些位置有非空白取值）。
MORTALITY_LAYOUT = {
    "seqn": (1, 6),
    "eligstat": (15, 15),     # 1=符合关联条件 2=未满 18 岁 3=其他不符合
    "mortstat": (16, 16),     # 0=存活 1=死亡 .=不适用
    "ucod_leading": (17, 19),
    "diabetes": (20, 20),
    "hyperten": (21, 21),
    "permth_int": (43, 45),   # 自访谈日起的随访月数
    "permth_exm": (46, 48),   # 自体检日起的随访月数
}

#: 问卷的「拒答 / 不知道」编码，一律当缺失
REFUSED_DONT_KNOW = {7, 9, 77, 99, 777, 999, 7777, 9999}

RACE = {
    1.0: "墨西哥裔美国人", 2.0: "其他西班牙裔", 3.0: "非西班牙裔白人",
    4.0: "非西班牙裔黑人", 6.0: "非西班牙裔亚裔", 7.0: "其他或混合",
}

#: 连续型测量：(源文件, 列名) -> (code, 中文名, 原始单位)
MEASUREMENTS = {
    ("BMX_J", "BMXBMI"):   ("LOINC:39156-5", "体质指数 BMI", "kg/m2"),
    ("BMX_J", "BMXWT"):    ("LOINC:29463-7", "体重", "kg"),
    ("BMX_J", "BMXHT"):    ("LOINC:8302-2", "身高", "cm"),
    ("BMX_J", "BMXWAIST"): ("LOINC:56086-2", "腰围", "cm"),
    ("TCHOL_J", "LBXTC"):  ("LOINC:2093-3", "总胆固醇", "mg/dL"),
}

#: 问卷项：(源文件, 列名) -> (code, 中文名, {编码: 文字})
#:
#: 存成带 value_text 的 measurement 而不是 condition。condition 表是「只记阳性」的
#: 语义 —— 没有记录就等同于「否」，这会把「明确回答否」和「没回答 / 拒答」混为一谈。
#: 问卷数据里这两者必须分开：把没回答的人当对照组，分母就是错的。
QUESTIONNAIRE = {
    ("DIQ_J", "DIQ010"): ("NHANES:diabetes", "被告知患有糖尿病",
                          {1.0: "是", 2.0: "否", 3.0: "临界"}),
    ("BPQ_J", "BPQ020"): ("NHANES:hypertension", "被告知患有高血压",
                          {1.0: "是", 2.0: "否"}),
    ("BPQ_J", "BPQ080"): ("NHANES:high_chol", "被告知胆固醇偏高",
                          {1.0: "是", 2.0: "否"}),
    ("SMQ_J", "SMQ020"): ("NHANES:smoker", "一生吸烟超过 100 支",
                          {1.0: "是", 2.0: "否"}),
}


def _clean_codes(series: pd.Series) -> pd.Series:
    """把 7 / 9 这类「拒答 / 不知道」编码换成缺失。"""
    return series.where(~series.isin(REFUSED_DONT_KNOW))


class NhanesAdapter(Adapter):
    dataset_id = "nhanes_2017"
    label = "NHANES 2017–2018"
    description = (
        "美国国家健康与营养检查调查，9254 名受访者（8704 人完成体检）。"
        "复杂抽样设计，带抽样权重与分层信息，人群估计必须加权。"
    )
    source_url = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017"
    version = "2017-2018-J"

    def is_available(self) -> bool:
        required = {"DEMO_J", "BMX_J", "BPX_J", "TCHOL_J", "DIQ_J", "BPQ_J", "SMQ_J"}
        present = {f.stem for f in SOURCE.glob("*.xpt")}
        return required.issubset(present)

    #: pandas.read_sas 把 XPT 里的数值 0 读成这个次正规数而不是 0.0。
    #: NHANES 没有任何合法取值小到这个量级，凡是绝对值小于它的都是 0。
    ZERO_EPS = 1e-70

    @classmethod
    def _read(cls, name: str) -> pd.DataFrame:
        """读一个 XPT，并把 read_sas 表示 0 的次正规数还原成 0。

        不还原会让好几处「等于 0」的判断静默失效，因为 5.4e-79 既不等于 0
        也大于 0：

        - `_mean_bp` 里「读数为 0 表示未测得」的 replace(0, nan) 匹配不上，
          81 条未测得的舒张压被当成真实读数平均进去，均值从 69.5 掉到 68.4
        - 只完成访谈者的 WTMEC2YR 真值是 0，但 `weights._clean` 的 `w > 0`
          判定为真，这些人没被排除在加权估计之外（点估计不受影响 ——
          乘 5.4e-79 等于没有 —— 但报出来的 n 把他们算进去了）
        - 未满 1 岁的婴儿年龄真值是 0，库里存成 5.4e-79
        """
        frame = pd.read_sas(SOURCE / f"{name}.xpt", format="xport")
        for column in frame.columns:
            if frame[column].dtype.kind == "f":
                frame[column] = frame[column].mask(
                    frame[column].abs() < cls.ZERO_EPS, 0.0)
        return frame

    @staticmethod
    def _mean_bp(frame: pd.DataFrame, prefix: str) -> pd.Series:
        """血压取多次读数的均值。

        NHANES 最多测 4 次。分析指南建议在有 2 次以上读数时舍去第一次
        （第一次普遍偏高），这里取第 2–4 次的均值，只有第一次时才用它。
        读数为 0 表示未测得，按缺失处理。
        """
        later = [f"{prefix}{i}" for i in (2, 3, 4) if f"{prefix}{i}" in frame]
        first = f"{prefix}1"

        block = frame[later].replace(0, np.nan) if later else None
        mean_later = block.mean(axis=1, skipna=True) if block is not None else None
        first_value = frame[first].replace(0, np.nan) if first in frame else None

        if mean_later is None:
            return first_value
        return mean_later.fillna(first_value) if first_value is not None else mean_later

    @staticmethod
    def has_mortality() -> bool:
        return MORTALITY_FILE.is_file()

    @classmethod
    def _read_mortality(cls) -> pd.DataFrame | None:
        """读死亡关联文件。没有就返回 None，其余部分照常工作。

        文件是定宽 ASCII，缺失写作 `.`。行尾空格被截掉了，所以有的行只有 46
        个字符而布局到 48 —— 切片天然容错，不用补齐。
        """
        if not cls.has_mortality():
            return None

        records = []
        for line in MORTALITY_FILE.read_text().splitlines():
            if not line.strip():
                continue
            records.append({
                name: line[start - 1:end].strip()
                for name, (start, end) in MORTALITY_LAYOUT.items()
            })
        return pd.DataFrame(records)

    @classmethod
    def _mortality_outcome(cls, ds: str, pid_of: dict[int, str]) -> pl.DataFrame:
        """全因死亡结局。

        只给 ELIGSTAT=1（符合关联条件）的人建记录 —— 未满 18 岁与其他不符合的人
        本来就没有随访，给他们一条 followup=NULL 的记录只会让「结局缺失」和
        「不在随访范围内」混成一谈。

        随访时间用 PERMTH_EXM（自体检日起）而不是 PERMTH_INT（自访谈日起）：
        person.sample_weight 存的是 MEC 体检权重，权重与随访时间必须指向同一个
        抽样框。只完成访谈的 311 人没有 PERMTH_EXM，他们的 MEC 权重恰好也是 0，
        两边是自洽的。

        **只导入全因死亡。** 文件里有 UCOD_LEADING（心脏病 37 例、恶性肿瘤 34
        例、其他 74 例），但病因别死亡要按竞争风险处理才严谨，而且 37 例事件跑
        Cox 估计很不稳 —— 导进来只会诱使人做欠功效的分析。
        """
        empty = pl.DataFrame(schema={
            "dataset": pl.Utf8, "person_id": pl.Utf8, "event_type": pl.Utf8,
            "event_ts": pl.Datetime, "censor_ts": pl.Datetime,
            "followup_days": pl.Float64, "is_event": pl.Boolean,
        })

        mort = cls._read_mortality()
        if mort is None:
            return empty

        eligible = mort[mort["eligstat"] == "1"]

        rows: list[dict] = []
        for record in eligible.to_dict("records"):
            person_id = pid_of.get(int(record["seqn"]))
            months = record["permth_exm"]
            if person_id is None or not months.isdigit():
                continue
            rows.append({
                "dataset": ds,
                "person_id": person_id,
                "event_type": "death",
                "event_ts": None,
                "censor_ts": None,
                # 月折成天。NCHS 只给到月，30.4375 = 365.25 / 12
                "followup_days": int(months) * 30.4375,
                "is_event": record["mortstat"] == "1",
            })

        return pl.DataFrame(rows, schema=empty.schema) if rows else empty

    def build(self) -> CdmTables:
        ds = self.dataset_id
        demo = self._read("DEMO_J")
        demo["SEQN"] = demo["SEQN"].astype("int64")
        pid_of = {seqn: f"nh-{seqn}" for seqn in demo["SEQN"]}

        person = pl.DataFrame({
            "dataset": [ds] * len(demo),
            "person_id": [pid_of[s] for s in demo["SEQN"]],
            "gender": [
                "M" if g == 1 else "F" if g == 2 else None for g in demo["RIAGENDR"]
            ],
            "birth_year": pl.Series([None] * len(demo), dtype=pl.Int32),
            "age_at_index": demo["RIDAGEYR"].astype("float64").to_list(),
            "race": [RACE.get(r) for r in demo["RIDRETH3"]],
            "ethnicity": [None] * len(demo),
            "sample_weight": demo["WTMEC2YR"].astype("float64").to_list(),
            "psu": [f"psu-{int(p)}" for p in demo["SDMVPSU"]],
            "stratum": [f"str-{int(s)}" for s in demo["SDMVSTRA"]],
        })

        # --- 测量 ---
        measurement_frames: list[pl.DataFrame] = []

        def add_measurement(seqn: pd.Series, values: pd.Series,
                            code: str, text: str, unit_raw: str) -> None:
            mask = values.notna()
            if not mask.any():
                return
            ids = [pid_of[s] for s in seqn[mask].astype("int64")]
            raw = values[mask].astype("float64").to_numpy()

            factor, target_unit = units.factor(unit_raw, code)
            normalised = np.round(raw * factor, 4) if factor != 1.0 else raw

            measurement_frames.append(pl.DataFrame({
                "dataset": [ds] * len(ids),
                "measurement_id": [f"{code}-{p}" for p in ids],
                "person_id": ids,
                "visit_id": [None] * len(ids),
                "code": [code] * len(ids),
                "code_system": ["LOINC"] * len(ids),
                "code_text": [text] * len(ids),
                "value_num": normalised.tolist(),
                "value_text": [None] * len(ids),
                "unit": [target_unit] * len(ids),
                "value_raw": raw.tolist(),
                "unit_raw": [unit_raw] * len(ids),
                "ts": pl.Series([None] * len(ids), dtype=pl.Datetime),
                "ref_low": [None] * len(ids),
                "ref_high": [None] * len(ids),
            }, schema_overrides={
                "visit_id": pl.Utf8, "value_text": pl.Utf8,
                "ref_low": pl.Float64, "ref_high": pl.Float64,
            }))

        for (file_name, column), (code, text, unit_raw) in MEASUREMENTS.items():
            frame = self._read(file_name)
            if column not in frame:
                continue
            add_measurement(frame["SEQN"], frame[column], code, text, unit_raw)

        bpx = self._read("BPX_J")
        add_measurement(bpx["SEQN"], self._mean_bp(bpx, "BPXSY"),
                        "LOINC:8480-6", "收缩压（多次读数均值）", "mmHg")
        add_measurement(bpx["SEQN"], self._mean_bp(bpx, "BPXDI"),
                        "LOINC:8462-4", "舒张压（多次读数均值）", "mmHg")

        # --- 问卷项（分类型 measurement，保留「否」与缺失的区别）---
        for (file_name, column), (code, text, levels) in QUESTIONNAIRE.items():
            frame = self._read(file_name)
            if column not in frame:
                continue
            answers = _clean_codes(frame[column]).map(levels)
            mask = answers.notna()
            if not mask.any():
                continue
            ids = [pid_of[s] for s in frame.loc[mask, "SEQN"].astype("int64")]
            measurement_frames.append(pl.DataFrame({
                "dataset": [ds] * len(ids),
                "measurement_id": [f"{code}-{p}" for p in ids],
                "person_id": ids,
                "visit_id": [None] * len(ids),
                "code": [code] * len(ids),
                "code_system": ["NHANES"] * len(ids),
                "code_text": [text] * len(ids),
                "value_num": [None] * len(ids),
                "value_text": answers[mask].tolist(),
                "unit": [None] * len(ids),
                "value_raw": frame.loc[mask, column].astype("float64").tolist(),
                "unit_raw": [None] * len(ids),
                "ts": pl.Series([None] * len(ids), dtype=pl.Datetime),
                "ref_low": [None] * len(ids),
                "ref_high": [None] * len(ids),
            }, schema_overrides={
                "visit_id": pl.Utf8, "value_num": pl.Float64, "unit": pl.Utf8,
                "unit_raw": pl.Utf8, "ref_low": pl.Float64, "ref_high": pl.Float64,
            }))

        # 横断面调查，没有从就诊记录来的诊断
        empty_condition = pl.DataFrame(schema={
            "dataset": pl.Utf8, "condition_id": pl.Utf8, "person_id": pl.Utf8,
            "visit_id": pl.Utf8, "code": pl.Utf8, "code_system": pl.Utf8,
            "code_text": pl.Utf8, "onset_ts": pl.Datetime, "is_primary": pl.Boolean,
        })

        empty_visit = pl.DataFrame(schema={
            "dataset": pl.Utf8, "visit_id": pl.Utf8, "person_id": pl.Utf8,
            "start_ts": pl.Datetime, "end_ts": pl.Datetime,
            "visit_type": pl.Utf8, "discharge_status": pl.Utf8,
        })
        empty_drug = pl.DataFrame(schema={
            "dataset": pl.Utf8, "drug_id": pl.Utf8, "person_id": pl.Utf8,
            "visit_id": pl.Utf8, "code": pl.Utf8, "code_system": pl.Utf8,
            "name": pl.Utf8, "start_ts": pl.Datetime, "end_ts": pl.Datetime,
            "dose": pl.Float64, "dose_unit": pl.Utf8, "route": pl.Utf8,
        })
        return {
            "person": person,
            "visit": empty_visit,
            "condition": empty_condition,
            "measurement": pl.concat(measurement_frames, how="vertical"),
            "drug": empty_drug,
            "outcome": self._mortality_outcome(ds, pid_of),
        }


register(NhanesAdapter())
