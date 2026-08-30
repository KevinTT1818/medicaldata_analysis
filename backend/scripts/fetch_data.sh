#!/usr/bin/env bash
#
# 下载四个公开数据集到 data/raw/。幂等：已经存在且校验通过的文件不重下。
#
# 原始文件只读、永不修改 —— 这是可复现的前提，所以脚本只负责取回来，
# 不做任何清洗，清洗全在 app/adapters/ 里。
set -euo pipefail

cd "$(dirname "$0")/.."
RAW="${MEDDATA_RAW_DIR:-$PWD/data/raw}"
mkdir -p "$RAW"

ONLY="${1:-all}"
want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }
say()  { printf '  %s\n' "$*"; }

fetch() {  # fetch <url> <目标路径>
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  curl -sSL --retry 3 --retry-delay 2 -o "$out" "$url"
}

# --- UCI Heart Disease（303 例，横断面）---
if want uci_heart; then
  if [ -f "$RAW/uci_heart/processed.cleveland.data" ]; then
    say "uci_heart 已存在，跳过"
  else
    echo "下载 uci_heart …"
    fetch "https://archive.ics.uci.edu/static/public/45/heart+disease.zip" "$RAW/heart_disease.zip"
    unzip -o -q "$RAW/heart_disease.zip" -d "$RAW/uci_heart"
    rm -f "$RAW/heart_disease.zip"
    say "ok"
  fi
fi

# --- UCI Heart Failure（299 例，带随访时间，可做生存分析）---
if want heart_failure; then
  TARGET="$RAW/heart_failure/heart_failure_clinical_records_dataset.csv"
  if [ -f "$TARGET" ]; then
    say "heart_failure 已存在，跳过"
  else
    echo "下载 heart_failure …"
    fetch "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip" "$RAW/hf.zip"
    unzip -o -q "$RAW/hf.zip" -d "$RAW/heart_failure"
    rm -f "$RAW/hf.zip"
    say "ok"
  fi
fi

# --- NHANES 2017-2018（9254 人，复杂抽样）---
#
# 必须串行下载：并发会被 CDC 限流，而限流返回的是 HTTP 200 的 HTML 错误页 ——
# 曾经 8 个并发请求全部"成功"，拿回来的却是同一张 Page Not Found。
# 所以每个文件都要校验 XPT 魔数，不合格的直接删掉重来。
if want nhanes; then
  echo "下载 nhanes（串行，CDC 会限流）…"
  for f in DEMO_J BMX_J BPX_J TCHOL_J DIQ_J BPQ_J SMQ_J; do
    OUT="$RAW/nhanes/$f.xpt"
    if [ -f "$OUT" ] && [ "$(head -c 20 "$OUT" | tr -d '\0')" = "HEADER RECORD*******" ]; then
      say "$f 已存在，跳过"
      continue
    fi
    fetch "https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/$f.xpt" "$OUT"
    if [ "$(head -c 20 "$OUT" | tr -d '\0')" = "HEADER RECORD*******" ]; then
      say "ok $f"
    else
      rm -f "$OUT"
      echo "  BAD $f —— 拿到的不是 XPT（多半被限流了），请稍后重试" >&2
      exit 1
    fi
  done
fi

# --- MIMIC-IV Clinical Database Demo（100 名患者，免认证）---
if want mimic_demo; then
  if [ -f "$RAW/mimic_demo/hosp/patients.csv.gz" ]; then
    say "mimic_demo 已存在，跳过"
  else
    echo "下载 mimic_demo …"
    fetch "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip" \
          "$RAW/mimic.zip"
    unzip -o -q "$RAW/mimic.zip" -d "$RAW/_mimic_tmp"
    # 压缩包里套了一层版本目录，摊平到 mimic_demo/
    INNER="$(find "$RAW/_mimic_tmp" -maxdepth 1 -mindepth 1 -type d | head -1)"
    rm -rf "$RAW/mimic_demo"
    mv "$INNER" "$RAW/mimic_demo"
    rm -rf "$RAW/_mimic_tmp" "$RAW/mimic.zip"
    say "ok"
  fi
fi

echo
echo "完成。data/raw 现有："
du -sh "$RAW"/*/ 2>/dev/null || true
