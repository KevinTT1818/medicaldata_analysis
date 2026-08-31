/** 与后端 FastAPI 的响应一一对应。后续可用 openapi-generator 自动生成替换本文件。 */

export type VariableKind = 'continuous' | 'categorical' | 'binary';
export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed';

export interface DatasetSummary {
  id: string;
  label: string;
  description: string;
  source_url: string;
  version: string;
  available: boolean;
  imported: boolean;
  n_person?: number;
  imported_at?: string;
}

export interface VariableInfo {
  id: string;
  label: string;
  kind: VariableKind;
  source: 'person' | 'measurement' | 'condition' | 'outcome';
  unit: string | null;
  n_available: number;
  n_missing: number;
  levels: string[] | null;
  /** 结局变量是否带随访时长，决定能否作为 KM / Cox 的结局。 */
  has_survival: boolean;
  /** 二分类变量的阳性人数。诊断/结局人人都有值，光看 n_available 分不出哪些常见。 */
  n_positive: number | null;
}

/**
 * 一人一项有多个值时取哪一个。
 * 横断面数据取哪个都一样；住院时序数据上这个选择会实实在在改变结论
 * （MIMIC 的肌酐中位数：取首次 88.4，取最大 97.2 µmol/L）。
 */
export type MeasurementAgg = 'first' | 'last' | 'mean' | 'min' | 'max';

export const MEASUREMENT_AGG_LABELS: Record<MeasurementAgg, string> = {
  first: '首次值（基线）',
  last: '末次值',
  mean: '均值',
  min: '最小值',
  max: '最大值',
};

export interface DatasetSchema {
  dataset: string;
  n_person: number;
  /** 覆盖人数过少、未列入目录的变量个数。 */
  hidden_variables: number;
  /** 是否存在一人一项多值的测量 —— 有才需要让用户选聚合策略。 */
  has_repeated_measures: boolean;
  /** 是否带复杂抽样权重。 */
  has_sample_weights: boolean;
  variables: VariableInfo[];
}

export interface AnalysisDescriptor {
  id: string;
  label: string;
  description: string;
  result_kind: string;
  params_schema: Record<string, unknown>;
}

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  stage: string;
  progress: number;
  error: string | null;
  warnings: string[];
  created_at: string;
  finished_at: string | null;
  /** 队列筛选后的人数与数据集全量人数，未筛选时等于全量。 */
  cohort_n: number | null;
  cohort_total: number | null;
  spec: {
    dataset: string;
    analysis?: string;
    params?: unknown;
    cohort?: unknown;
    cohort_id?: string;
    fingerprint?: string;
  };
}

/** 一行的单元格：键为分组标签，'__overall__' 是总体列。 */
export type Cells = Record<string, string>;

export interface BaselineLevel {
  label: string;
  cells: Cells;
}

export interface BaselineRow {
  variable: string;
  label: string;
  unit: string | null;
  kind: VariableKind;
  stat: 'mean_sd' | 'median_iqr' | 'n_pct';
  n_missing: number;
  cells: Cells;
  levels: BaselineLevel[] | null;
  p: number | null;
  p_adj: number | null;
  test: string | null;
  /** 缺失信息占比。仅多重插补时有值。 */
  fmi: number | null;
  warnings: string[];
}

export interface BaselineTableResult {
  kind: 'baseline_table';
  overall_n: number;
  groups: { key: string; label: string; n: number; weighted_pct: number | null }[];
  group_by: string | null;
  group_by_label: string | null;
  rows: BaselineRow[];
  p_adjust: string;
  /** 是否按抽样权重给出人群估计。加权模式下不出 p 值。 */
  weighted: boolean;
  /** 数据集本身是否带抽样权重。 */
  weights_available: boolean;
  /** Kish 有效样本量：权重差异越大，有效信息越少。 */
  effective_n: number | null;
  /** 多重插补的执行情况。默认按观测值统计时为 null。 */
  imputation: ImputationReport | null;
  /** 抽样设计参数。加权时 p 值来自设计校正 Wald 检验，自由度就是这里的 df。 */
  design: {
    n_strata: number;
    n_psu: number;
    df: number;
    /** 只有权重、没有分层与 PSU 时为 true，标准误按有放回抽样近似。 */
    approximate: boolean;
  } | null;
  notes: string[];
}

export interface MissingnessResult {
  kind: 'missingness';
  overall_n: number;
  rows: { variable: string; label: string; n_missing: number; pct_missing: number }[];
  complete_cases: number;
  complete_pct: number;
}

export interface KmSeries {
  name: string;
  n: number;
  events: number;
  censored: number;
  /** 未过半时为 null，前端显示「未达到」。 */
  median_survival: number | null;
  t: number[];
  survival: number[];
  ci_lower: number[];
  ci_upper: number[];
  censor_t: number[];
  censor_s: number[];
}

export interface KmCurveResult {
  kind: 'km_curve';
  time_unit: string;
  conf_level: number;
  group_by_label: string | null;
  series: KmSeries[];
  at_risk: { times: number[]; rows: { name: string; counts: number[] }[] };
  logrank_p: number | null;
  test: string | null;
}

export interface ForestRow {
  label: string;
  variable: string;
  reference: string | null;
  estimate: number;
  ci_lower: number;
  ci_upper: number;
  p: number;
  /** 缺失信息占比：这个系数有多少不确定性是插补带来的。仅多重插补时有值。 */
  fmi?: number | null;
}

export interface ImputationReport {
  m: number;
  iterations: number;
  imputed_rows: number;
  total_rows: number;
  columns: { variable: string; n_missing: number; pct: number; kind: string }[];
}

export interface RocData {
  auc: number | null;
  fpr: number[];
  tpr: number[];
}

export interface ForestResult {
  kind: 'forest';
  model: string;
  /** 效应量名称：HR 或 OR。 */
  effect_label: string;
  /** 无效应参考线的位置，比值类效应量为 1。 */
  null_value: number;
  conf_level: number;
  rows: ForestRow[];
  ph_test: { label: string; p: number }[];
  roc?: RocData;
  /** 是否按抽样权重估计。为 true 时置信区间来自设计校正的三明治方差。 */
  weighted?: boolean;
  design?: {
    n_strata: number;
    n_psu: number;
    df: number;
    approximate: boolean;
  } | null;
  /** 加权拟合与合并结果下伪 R² 都没有标准定义，此时为 null。 */
  pseudo_r2?: number | null;
  /** 多重插补的执行情况。未用插补时为 null。 */
  imputation?: ImputationReport | null;
  concordance?: number;
  n_used: number;
  n_events: number;
  n_dropped: number;
  notes: string[];
}

export interface BoxStats {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
  mean: number;
  sd: number | null;
  n: number;
}

export interface DistributionPanel {
  variable: string;
  label: string;
  unit: string | null;
  kind: VariableKind;
  bin_edges?: number[];
  levels?: string[];
  series: { name: string; counts: number[]; box?: BoxStats; total?: number }[];
  n_missing: number;
}

export interface DistributionResult {
  kind: 'distribution';
  overall_n: number;
  group_by_label: string | null;
  grouped: boolean;
  panels: DistributionPanel[];
}

export type AnalysisResult =
  | BaselineTableResult
  | MissingnessResult
  | KmCurveResult
  | ForestResult
  | DistributionResult;

export interface JobResultEnvelope {
  job: Job;
  result: AnalysisResult;
}

export const OVERALL = '__overall__';


// ---------------------------------------------------------------- 队列

export type FilterOp =
  | 'eq' | 'ne' | 'lt' | 'lte' | 'gt' | 'gte'
  | 'between' | 'in' | 'not_in' | 'is_null' | 'not_null';

export interface FilterCondition {
  kind: 'condition';
  variable: string;
  op: FilterOp;
  /** between 用 [下界, 上界]；in / not_in 用水平数组；is_null / not_null 不用。 */
  value?: number | string | (number | string)[] | null;
}

export interface FilterGroup {
  kind: 'group';
  op: 'and' | 'or';
  children: FilterNode[];
}

export type FilterNode = FilterCondition | FilterGroup;

export interface OperatorMeta {
  allowed: Record<VariableKind, FilterOp[]>;
  symbols: Record<FilterOp, string>;
  no_value_ops: FilterOp[];
  list_ops: FilterOp[];
  range_ops: FilterOp[];
}

export interface ConsortStep {
  label: string;
  n_before: number;
  n_after: number;
  n_excluded: number;
  /** 这一步被排除的人里，因该变量缺失而非不满足条件的人数。 */
  n_excluded_missing: number;
}

export interface CohortSummaryItem {
  variable: string;
  label: string;
  unit: string | null;
  kind: 'continuous' | 'categorical';
  n: number;
  median?: number;
  q1?: number;
  q3?: number;
  levels?: { label: string; n: number; pct: number }[];
}

export interface CohortPreview {
  dataset: string;
  total: number;
  n: number;
  pct: number;
  flow: {
    total: number;
    final_n: number;
    steps: ConsortStep[];
    /** OR 条件拆不成先后步骤，此时为 false，前端不画流程箭头。 */
    sequential: boolean;
  };
  summary: CohortSummaryItem[];
}

export interface SavedCohort {
  id: string;
  dataset: string;
  name: string;
  description: string | null;
  definition: FilterNode;
  created_at: string;
  updated_at: string;
}

export function emptyGroup(): FilterGroup {
  return { kind: 'group', op: 'and', children: [] };
}


// ---------------------------------------------------------------- 报告

export interface ReportSection {
  title: string;
  note: string | null;
  dataset: string;
  analysis: string;
  params: Record<string, unknown>;
  cohort_id: string | null;
  cohort: FilterNode | null;
  measurement_agg: MeasurementAgg;
}

export interface SectionSnapshot {
  fingerprint: string;
  dataset_version: string;
  result_hash: string;
  n: number | null;
}

export interface SavedReport {
  id: string;
  title: string;
  description: string | null;
  sections: ReportSection[];
  /** 保存时各小节的结果指纹，重跑时用来比对。 */
  snapshot: SectionSnapshot[] | null;
  created_at: string;
  updated_at: string;
}

export type ReproVerdict = 'match' | 'changed' | 'failed' | 'no_baseline';

export interface SectionComparison {
  index: number;
  title: string;
  verdict: ReproVerdict;
  detail: string;
  baseline?: SectionSnapshot;
  current?: SectionSnapshot;
}

export interface ReportRunSection {
  title: string;
  note: string | null;
  status: 'succeeded' | 'failed';
  error: string | null;
  job: Job;
  result: AnalysisResult | null;
}

export interface ReportRun {
  report: { id: string; title: string; description: string | null;
            created_at: string; updated_at: string };
  sections: ReportRunSection[];
  /** 全部小节都与基线一致才为 true。 */
  reproducible: boolean;
  has_baseline: boolean;
  comparisons: SectionComparison[];
}
