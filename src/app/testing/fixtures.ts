/**
 * 测试夹具。形状与后端下发的一致 —— 各字段的真实取值范围见
 * backend/app/analyses/ 下对应算子的 result 构造处。
 */
import type {
  ForestResult,
  KmCurveResult,
  KmSeries,
  OperatorMeta,
  ReportSection,
  VariableInfo,
} from '../core/api.types';

export function variable(patch: Partial<VariableInfo> & { id: string }): VariableInfo {
  return {
    label: patch.id,
    kind: 'continuous',
    source: 'person',
    unit: null,
    n_available: 100,
    n_missing: 0,
    levels: null,
    has_survival: false,
    n_positive: null,
    ...patch,
  };
}

/** 覆盖四类变量筛选器（grouping / survival / binary / covariate）会用到的组合。 */
export const VARIABLES: VariableInfo[] = [
  variable({ id: 'person.age_at_index', label: '年龄', unit: '岁' }),
  variable({ id: 'person.gender', label: '性别', kind: 'binary', levels: ['F', 'M'] }),
  variable({ id: 'person.race', label: '种族', kind: 'categorical', levels: ['A', 'B', 'C'] }),
  variable({
    id: 'measurement.chol', label: '胆固醇', source: 'measurement',
    unit: 'mmol/L', n_missing: 7,
  }),
  variable({
    id: 'condition.hf', label: '心衰', kind: 'binary', source: 'condition',
    levels: ['否', '是'], n_positive: 32,
  }),
  variable({
    id: 'outcome.death', label: '结局：死亡', kind: 'binary', source: 'outcome',
    levels: ['否', '是'], has_survival: true, n_positive: 41,
  }),
  variable({
    id: 'outcome.readmit', label: '结局：再入院', kind: 'binary', source: 'outcome',
    levels: ['否', '是'], has_survival: false, n_positive: 12,
  }),
];

export const OPERATORS: OperatorMeta = {
  allowed: {
    continuous: ['lt', 'lte', 'gt', 'gte', 'between', 'is_null', 'not_null'],
    categorical: ['eq', 'ne', 'in', 'not_in', 'is_null', 'not_null'],
    binary: ['eq', 'ne', 'in', 'not_in', 'is_null', 'not_null'],
  },
  symbols: {
    eq: '=', ne: '≠', lt: '<', lte: '≤', gt: '>', gte: '≥',
    between: '介于', in: '属于', not_in: '不属于',
    is_null: '缺失', not_null: '非缺失',
  },
  no_value_ops: ['is_null', 'not_null'],
  list_ops: ['in', 'not_in'],
  range_ops: ['between'],
};

export function kmSeries(patch: Partial<KmSeries> & { name: string }): KmSeries {
  return {
    n: 50,
    events: 20,
    censored: 30,
    median_survival: 180,
    t: [0, 30, 90, 180],
    survival: [1, 0.9, 0.75, 0.5],
    ci_lower: [1, 0.82, 0.64, 0.37],
    ci_upper: [1, 0.95, 0.85, 0.63],
    censor_t: [45, 120],
    censor_s: [0.9, 0.75],
    ...patch,
  };
}

export function kmResult(series: KmSeries[] = [kmSeries({ name: '全体' })]): KmCurveResult {
  return {
    kind: 'km_curve',
    time_unit: '天',
    conf_level: 0.95,
    group_by_label: series.length > 1 ? '性别' : null,
    series,
    at_risk: { times: [0, 90, 180], rows: series.map((s) => ({ name: s.name, counts: [s.n, 30, 18] })) },
    logrank_p: series.length > 1 ? 0.032 : null,
    test: series.length > 1 ? 'log-rank' : null,
  };
}

export function forestResult(patch: Partial<ForestResult> = {}): ForestResult {
  return {
    kind: 'forest',
    model: 'Cox 比例风险',
    effect_label: 'HR',
    null_value: 1,
    conf_level: 0.95,
    rows: [
      // 显著、风险升高
      { label: '年龄', variable: 'person.age_at_index', reference: null,
        estimate: 1.8, ci_lower: 1.2, ci_upper: 2.7, p: 0.004 },
      // 置信区间跨过 1 —— 不显著
      { label: '性别 M', variable: 'person.gender', reference: 'F',
        estimate: 1.1, ci_lower: 0.7, ci_upper: 1.7, p: 0.62 },
      // 显著、风险降低
      { label: '心衰', variable: 'condition.hf', reference: '否',
        estimate: 0.45, ci_lower: 0.25, ci_upper: 0.8, p: 0.007 },
    ],
    ph_test: [],
    n_used: 280,
    n_events: 96,
    n_dropped: 19,
    notes: [],
    ...patch,
  };
}

export function section(patch: Partial<ReportSection> = {}): ReportSection {
  return {
    title: '基线特征',
    note: null,
    dataset: 'uci_heart',
    analysis: 'describe.baseline_table',
    params: { variables: ['person.age_at_index'] },
    cohort_id: null,
    cohort: null,
    measurement_agg: 'first',
    ...patch,
  };
}
