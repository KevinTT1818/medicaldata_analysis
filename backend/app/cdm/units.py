"""单位归一。

同一化验项在不同数据集里单位不同（血糖 mg/dL 与 mmol/L 差 18 倍，
肌酐 mg/dL 与 umol/L 差 88 倍）。CDM 层统一到目标单位，
同时在 value_raw / unit_raw 保留原始值以便追溯和重算。
"""
from __future__ import annotations

# (源单位, 目标单位) -> 乘数
_FACTORS: dict[tuple[str, str], float] = {
    ("mg/dL", "mmol/L::cholesterol"): 0.02586,   # 总胆固醇 / LDL / HDL
    ("mg/dL", "mmol/L::glucose"): 0.05551,       # 葡萄糖
    ("mg/dL", "mmol/L::triglyceride"): 0.01129,  # 甘油三酯
    ("mg/dL", "umol/L::creatinine"): 88.4,       # 肌酐
    ("mg/dL", "mmol/L::urea"): 0.357,            # 尿素氮
    ("g/dL", "g/L"): 10.0,
    ("mmHg", "kPa"): 0.1333,
    ("lb", "kg"): 0.45359237,
    ("in", "cm"): 2.54,
}

# 注意：本模块只支持乘法换算。华氏转摄氏这类需要偏移量的换算表达不了，
# 得在适配器里显式处理（曾经这里放过一条 ("F","C"): 0.0 的占位，
# 真被用上会把每个温度值抹成 0）。

# 各化验项的目标单位与所用换算键
TARGETS: dict[str, tuple[str, str]] = {
    # code -> (目标单位, 换算键后缀)
    "LOINC:2093-3": ("mmol/L", "cholesterol"),   # 总胆固醇
    "LOINC:2571-8": ("mmol/L", "triglyceride"),  # 甘油三酯
    "LOINC:2345-7": ("mmol/L", "glucose"),       # 葡萄糖
    "LOINC:2160-0": ("umol/L", "creatinine"),    # 肌酐
    "LOINC:3094-0": ("mmol/L", "urea"),          # 尿素氮 BUN
}


class UnitError(ValueError):
    pass


def factor(from_unit: str, code: str) -> tuple[float, str]:
    """返回 (换算系数, 目标单位)。

    系数本身不做舍入 —— 调用方拿它去乘一整列时，先舍入系数会把误差
    放大到每一个值上（0.02586 舍成 0.0259 就是 0.15% 的系统性偏差）。
    只在最终数值上舍入。
    """
    if code not in TARGETS:
        return 1.0, from_unit

    target_unit, kind = TARGETS[code]
    if from_unit == target_unit:
        return 1.0, target_unit

    f = _FACTORS.get((from_unit, f"{target_unit}::{kind}"))
    if f is None:
        f = _FACTORS.get((from_unit, target_unit))
    if f is None:
        raise UnitError(f"没有登记 {from_unit} -> {target_unit} ({kind}) 的换算系数")
    return f, target_unit


def convert(value: float | None, from_unit: str, code: str) -> tuple[float | None, str]:
    """把单个 value 从 from_unit 归一到 code 对应的目标单位。"""
    f, target_unit = factor(from_unit, code)
    if value is None:
        return None, target_unit
    return round(value * f, 4), target_unit
