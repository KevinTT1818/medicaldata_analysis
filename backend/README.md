# 医疗数据分析平台 · 后端

FastAPI + DuckDB + Polars。公开医疗数据集经适配层映射到统一数据模型（CDM），
再由可插拔的分析算子消费。

## 跑起来

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
./run.sh
```

服务在 <http://localhost:8000>，交互式 API 文档在 `/docs`。
前端 `ng serve` 已配 `proxy.conf.json`，把 `/api` 转到这里。

## 数据

原始文件放 `data/raw/<dataset_id>/`，只读，永不修改 —— 这是可复现的前提。

UCI Heart Disease（横断面，303 例）：

```bash
curl -sSL -o data/raw/heart_disease.zip https://archive.ics.uci.edu/static/public/45/heart+disease.zip && unzip -o -q data/raw/heart_disease.zip -d data/raw/uci_heart
```

UCI Heart Failure Clinical Records（随访队列，299 例，可做生存分析）：

```bash
curl -sSL -o data/raw/hf.zip https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip && unzip -o -q data/raw/hf.zip -d data/raw/heart_failure
```

NHANES 2017–2018（9254 人，复杂抽样调查）。**必须串行下载**，并发会被 CDC 限流，
而且限流返回的是 HTTP 200 的错误页 —— 所以每个文件都要校验 XPT 魔数：

```bash
cd data/raw && mkdir -p nhanes && for f in DEMO_J BMX_J BPX_J TCHOL_J DIQ_J BPQ_J SMQ_J; do curl -sSL --retry 3 -o "nhanes/$f.xpt" "https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/$f.xpt"; [ "$(head -c 20 "nhanes/$f.xpt" | tr -d '\0')" = "HEADER RECORD*******" ] && echo "ok $f" || { echo "BAD $f"; rm -f "nhanes/$f.xpt"; }; done
```

导入走 API（`POST /api/datasets/uci_heart/import`）或前端「数据集」页的按钮。

数据分层：

| 目录 | 内容 |
| --- | --- |
| `data/raw/` | 原始下载文件，只读 |
| `data/cdm/` | CDM 6 张表的 Parquet，按数据集分区 |
| `data/warehouse.duckdb` | DuckDB 库，ETL 从 Parquet 灌入 |

`MEDDATA_DATA_DIR` 可覆盖数据目录，`MEDDATA_RAW_DIR` 可单独覆盖原始文件目录。
DuckDB 是单写入者，测试靠这个环境变量指向独立库文件，才能和开发服务器并行跑。

CDM 演进时不用手工删库：`schema.init()` 会做一次列级对账，
把 DDL 里新增的列 `ALTER TABLE ADD COLUMN` 补到已存在的表上（只加不删）。

## 加一个数据集

写一个 `app/adapters/<name>.py`，继承 `Adapter`，实现 `is_available()` 和
`build()`（返回 CDM 6 张表的 DataFrame），最后 `register()`。
**全部已有分析算子立刻可用，不需要改任何分析代码。**

参照 `app/adapters/uci_heart.py`。要点：

- 连续型测量写 `value_num` + `unit`，同时把原始值和原始单位留在 `value_raw` / `unit_raw`
- 分类型检查结果写 `value_text`（如铊显像的「可逆缺损」）
- 单位换算在 `app/cdm/units.py` 登记，**系数本身不要舍入**，只舍入最终值
- 有随访时长的队列填 `outcome.followup_days`，生存分析靠它；只有结局有无的
  横断面数据留空即可，算子会明确报错而不是返回空曲线
- `pl.read_csv` 的类型推断只看前若干行，数值列建议显式指定 Float64
  （心衰数据集的 age 在文件后段才出现 60.667，按 i64 推断会被静默截断）

MIMIC-IV Demo 2.2（100 例 ICU 患者，免认证）：

```bash
cd data/raw && curl -sSL --retry 3 -o m.zip "https://physionet.org/content/mimic-iv-demo/get-zip/2.2/" && unzip -qo m.zip -d t && mv t/mimic-iv-clinical-database-demo-2.2 mimic_demo && rm -rf t m.zip
```

## 重复测量：一人一项多值

MIMIC 平均每人每个化验项有 43 个值（最多 949 个）。取首次、末次还是极值会
**实实在在改变结论**——肌酐中位数：首次 88.4、最大 97.2、最小 57.5 µmol/L。

所以 `build_feature_frame(..., agg=...)` 必须由调用方显式指定，
API 上是 `measurement_agg`，取值 `first`（默认，基线）/ `last` / `mean` / `min` / `max`。
**它进指纹**——同参数不同聚合是两次不同的分析。

数据集有没有重复测量由后端判定（`has_repeated_measures`），
横断面数据一人一值，前端不会拿这个选项打扰用户。

## 变量目录的覆盖率阈值

MIMIC 有 498 个化验项、几百个 ICD 码，多数只出现在一两个人身上。全列出来变量
选择器没法用，而且只覆盖一两个人的变量也做不了统计。低于
`MIN_VARIABLE_COVERAGE`（5%）且低于 `MIN_VARIABLE_PATIENTS`（2 人）的编码不列出，
隐藏数在 schema 响应里报出（MIMIC：列出 181 个，隐藏 1328 个）。

诊断与结局变量额外给出 `n_positive`（阳性人数）——presence-only 语义下人人都有值，
光看 `n_available` 分不出哪些诊断常见。

## 复杂抽样与加权

NHANES 这类调查按不同概率抽样，**直接求均值得到的是样本均值，不是人群估计**。
`person` 表上有 `sample_weight` / `psu` / `stratum` 三列，宽表里权重以保留列
`__weight` 携带（它是抽样设计元数据，不出现在变量目录里）。

`describe.baseline_table` 的 `weighting` 参数：

| 取值 | 行为 |
| --- | --- |
| `auto`（默认） | 有权重就加权 |
| `weighted` | 强制加权；数据集没权重时报错 |
| `unweighted` | 不加权，只描述样本 |

### 标准误：Taylor 线性化

`app/analyses/survey.py`。分层多阶段抽样下不能把观测当独立同分布 —— 同一个初级
抽样单元（PSU）里的人是相关的，忽略聚类会**系统性低估方差**，把不显著的结论算成显著。

用的是标准的线性化估计量：先把统计量线性化成每个观测的残差 z，
再按「层内 PSU 之间的离散度」求和：

```
V = Σ_h  n_h/(n_h-1)  Σ_i (z_hi· - z̄_h·)²
```

设计自由度 = PSU 总数 − 层数。NHANES 2017–2018 是 15 层 × 2 PSU，
自由度 15 —— 与 NCHS 官方口径一致。

**域（子人群）估计不能先切数据再算。** 切掉会让某些层/PSU 整个消失，方差就不对了。
正确做法是保留全部 PSU，用指示变量把域外观测的 z 置零。

分组比较走**设计校正 Wald 检验**：连续变量比各组均值，分类变量比
(组数−1)×(水平数−1) 维的构成比对比向量。对比数超过设计自由度时该检验不可估计，
这时明说而不是给个数字。

加权时连续变量一律报**均值**：检验比的是均值，表里显示中位数读者对不上。

### 加权 logistic 回归

`survey.weighted_logit`。点估计是伪极大似然：解加权得分方程 `Σ w x (y − p) = 0`，
用自写的 Newton-Raphson 而不是 statsmodels 的 `freq_weights` —— 后者的语义是
「这一行代表多少个观测」，填抽样权重在数学上恰好给出同一个得分方程，
但它据此算出的标准误是错的，容易被人当成对的拿去用。

方差用设计校正的三明治估计量：

```
V(β) = J⁻¹ · V(U) · J⁻¹
```

`J = X' diag(w p (1−p)) X` 是加权信息矩阵（面包），`V(U)` 是得分贡献
`u_i = w_i x_i (y_i − p_i)` 的**设计方差**（肉）—— 聚类结构在这里被算进去。
与 R 的 `svyglm`、Stata 的 `svy: logit` 是同一套。置信区间用设计自由度的 t 分布。

AUC 也加权，否则系数描述人群、AUC 描述样本，放在同一张结果里没法解释。
伪 R² 在加权拟合下没有标准定义，加权时不给。

在 NHANES 成人高血压模型上，设计校正让标准误比朴素法大 1.06–1.29 倍，
而自由度从 5171 掉到 15；**更大的差别在点估计本身** ——
性别的 OR 加权是 1.42、不加权是 1.16。

其余算子（Cox / KM / 分布）仍不支持加权，在带权重的数据集上会显式警告。

结果里同时给出 **Kish 有效样本量** 与设计参数 —— 权重差异和聚类的代价该被看见。

### 外部校验

| 指标 | 本系统 | NCHS 官方 |
| --- | --- | --- |
| 成人肥胖率（BMI ≥ 30） | 42.7%，SE 1.72%，95%CI [39.0, 46.3] | 42.4% |
| 重度肥胖（BMI ≥ 40） | 9.1% | 9.2% |
| 设计自由度 | 15 | 15 |

同一个肥胖率若按简单随机抽样算，SE 只有 0.69% —— **低估 2.5 倍**，
设计效应 DEFF 6.3。这就是必须做设计校正的理由。


## 就诊派生变量

分析单位始终是**人** —— 所有算子都假设一人一行。所以不把就诊变成分析单位，
而是把就诊事实派生成人级变量，让「再入院」「ICU 停留超过 3 天」能进条件树。

数据集里每种 `visit_type` 各派生一组：

| 变量 | 含义 | 没有该类就诊时 |
| --- | --- | --- |
| `visit.count:<type>` | 次数 | **0**（没住过就是 0 次，这是事实） |
| `visit.los_total:<type>` | 总时长（天） | **NULL**（没住过谈不上住了几天） |
| `visit.los_max:<type>` | 单次最长（天） | **NULL** |
| `visit.discharge_status:<type>` | 结束状态 | NULL |

次数补 0 而时长留空是刻意的：把没住过的人当 0 天混进均值，
回答的就不是「住过的人住了多久」这个问题了。

结束状态按就诊类型分开也是刻意的：住院的结束状态是出院去向（HOME、REHAB），
ICU 停留的是末次监护单元（MICU、SICU），混在一个变量里会让人以为
「MICU」是一种出院去向。

## 队列

队列是一棵嵌套的 AND / OR 条件树，存的是**定义**而不是命中的人员名单 ——
数据集重新导入后队列依然有效，这也是分析可复现的前提。

```
POST /api/cohorts/preview   不落盘算一遍：命中人数 + CONSORT 入排流程 + 人口学摘要
POST /api/cohorts           保存
GET  /api/cohorts?dataset=  列出
PUT  /api/cohorts/{id}      更新
```

分析时用 `cohort_id` 引用已保存的队列，或用 `cohort` 传内联条件树。
**队列定义进指纹** —— 同一个算子换个队列必须算作另一次分析。

条件树有两条执行路径，**必须给出同一批人**（有测试逐个运算符比对）：

- `filters.evaluate()` 在 Python 里算掩码，队列预览用它，因为 CONSORT 流程要逐步的掩码
- `filters.to_sql()` 编译成 WHERE 下推给 DuckDB，分析用它，在物化到 Python 之前先筛掉人

SQL 的 NULL 语义天然就是「缺失一律判 False」，与 Python 侧一致。取值一律走绑定参数。

### 缺失值语义

除 `is_null` / `not_null` 外，**任何比较遇到缺失一律判 False**，`ne` 和 `not_in`
也不例外。这与 SQL 的 NULL 语义一致，临床上也更稳妥：无法确认满足条件的人不该进队列。

但这会静默丢人，所以 CONSORT 流程把「不满足条件」和「该变量缺失」分开计数。
缺失从来不是随机的 —— 做了某项检查本身就和病情严重程度相关，
按缺失删人会引入选择偏倚。

## 加一个分析算子

写一个 `app/analyses/` 下的类，继承 `Analysis`，用 Pydantic 声明 `Params`，
实现 `run(ctx)`，加 `@register`。
`GET /api/analyses/registry` 会把 `Params` 的 JSON Schema 吐给前端，
**前端动态渲染参数表单，零改动。**

变量类字段要用 `app/analyses/widgets.py` 里的辅助函数加标注 —— 纯 JSON Schema
表达不了「这个字符串是变量 ID，且只能选带随访时长的结局」：

```python
outcome: str = Field(..., json_schema_extra=widgets.variable(
    "结局", "必须带随访时长", widgets.SURVIVAL_OUTCOME))
```

可用筛选器：`ANY` / `GROUPING`（分类或二分类）/ `SURVIVAL_OUTCOME` /
`BINARY_OUTCOME` / `COVARIATE`。

## 结果缓存

缓存键 = 数据集 + **数据集版本（导入时间戳）** + 算子 + 参数 + 队列定义，
落盘在 `data/cache/`。重新导入数据后旧结果自动失效，不会拿着过期结论继续用。

## 报告与可复现

报告存的是**分析的定义**，不是算出来的数字——存数字就没法重跑，也答不出
「三个月前那份报告现在还成立吗」。

保存时会跑一遍并记下每节的基线快照（分析指纹 + 数据集版本 + 结果哈希 + 纳入例数）。
之后 `POST /api/reports/{id}/run` 重跑并逐节比对，给出四种判定：

| 判定 | 含义 |
| --- | --- |
| `match` | 结果与保存时完全一致 |
| `changed` | 数字变了，并指出是定义变了、数据重导了，还是例数变了 |
| `failed` | 这一节跑不通 |
| `no_baseline` | 没有基线可比 |

两条刻意的语义：

- **`reproducible` 只在「有基线且全部 match」时为 true。** 没有基线是「没核对过」，
  不是「可复现」，报成 true 会给人虚假的安心。
- **更新报告定义时不传快照会保留原基线**，不是抹掉。抹掉就再也答不出
  「和当初那份比变了没有」；定义改了而基线还是旧的，正是需要被报出来的情况。
  要换基线得显式调 `/rebaseline`。

导出走浏览器打印（`@media print` 已隐藏导航与操作按钮，图表原样渲染），
选「另存为 PDF」即可，不引入任何服务端渲染依赖。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

期望值全部来自对原始文件的独立手工核对或官方发布数字，不是从流水线输出反抄的。

`tests/_env.py` 必须被每个测试模块**最先**导入。`app.config` 在一个进程里只导入
一次，各测试文件各设各的 `MEDDATA_DATA_DIR` 的话只有第一个生效，其余文件拿到的路径
是假的（这曾让一个缓存测试单独跑通过、全量跑失败）。

## 算子清单

| ID | 产出 | 依赖 |
| --- | --- | --- |
| `describe.baseline_table` | Baseline / Table 1，自动选检验方法 + FDR 校正 + 抽样加权 | scipy |
| `describe.missingness` | 逐变量缺失率与完整病例数 | polars |
| `describe.distribution` | 直方图 + Tukey 箱线图 / 构成比 | numpy |
| `survival.kaplan_meier` | KM 曲线 + 置信带 + 删失标记 + at-risk 表 + log-rank | lifelines |
| `survival.cox` | HR + 置信区间 + 森林图 + 比例风险假设检验 | lifelines |
| `regression.logistic` | OR + 置信区间 + 森林图 + ROC/AUC | statsmodels |

## 数据集

| ID | 规模 | 特点 |
| --- | --- | --- |
| `uci_heart` | 303 例 | 横断面，二分类结局，无随访时间 |
| `heart_failure` | 299 例 | 随访队列，只有随访天数没有日期 |
| `nhanes_2017` | 9254 人 | 复杂抽样调查，需加权才是人群估计 |
| `mimic_demo` | 100 例 / 275 次住院 | 住院时序，六张 CDM 表全用上，带真实时间戳 |

`mimic_demo` 是唯一填上 `visit` 与 `drug` 两张表的数据集，也是唯一同时有
`event_ts` / `censor_ts` 和 `followup_days` 的数据集。它的肌酐经 LOINC 映射后
与 `heart_failure` 的肌酐是同一个变量，单位都归一到 µmol/L。

## 现状

M1–M5 已完成：CDM 6 表（全部有数据）、四个适配器、6 个算子、
JSON Schema 驱动的参数表单、异步任务 + SSE、schema 迁移、
队列构建器 + CONSORT 流程 + 持久化、抽样加权、重复测量聚合、SQL 下推、结果缓存、
报告组装 + 可复现核对 + 打印导出。前端有 KM 曲线、森林图、ROC、直方图/箱线图。

万级样本（NHANES 9254 人）上端到端 50–75 ms，缓存命中 12 ms。

**未做**：

- 按变量选抽样权重（NHANES 的禁食子样本 GLU_J 因此没导入）
- Cox 与 KM 的设计校正 —— 目前 Table 1 与 logistic 支持
- **真正的就诊级分析**：现在是把就诊派生成人级变量，分析单位仍是人。
  要做「每次住院一行」的分析（比如按就诊建模再入院风险），得让 CDM 支持
  以就诊为主键的宽表，那是另一套东西
- DOCX 导出（当前只有打印 / PDF）
