/**
 * 森林图的 option 构建。
 *
 * 守两件容易错且错了看不出来的事：
 * 1. 比值类效应量必须画在对数刻度上 —— 线性轴会让 HR=0.5 和 HR=2 看起来
 *    离参考线的距离差一倍，而它们其实是等幅的相反效应。
 * 2. 置信区间跨过无效应线的项要画成中性色 —— 颜色是这张图传达显著性的
 *    主要手段，判断反了会把"不显著"读成"有效应"。
 */
import { TestBed } from '@angular/core/testing';

import { ForestPlot } from './forest-plot';
import { forestResult } from '../../../testing/fixtures';
import type { ForestResult } from '../../../core/api.types';

function make(result: ForestResult = forestResult()) {
  const fixture = TestBed.createComponent(ForestPlot);
  fixture.componentRef.setInput('result', result);
  fixture.detectChanges();
  const c = fixture.componentInstance;
  return { c, option: c.option() as Record<string, unknown> };
}

describe('ForestPlot 坐标轴', () => {
  it('x 轴必须是对数刻度', () => {
    const { option } = make();
    const x = option['xAxis'] as Record<string, unknown>;
    expect(x['type']).toBe('log');
    expect(x['name']).toContain('对数刻度');
  });

  it('参考线画在 null_value 上', () => {
    const { option } = make();
    const series = (option['series'] as Record<string, unknown>[])[0];
    const markLine = series['markLine'] as { data: { xAxis: number }[] };
    expect(markLine.data[0].xAxis).toBe(1);
  });

  it('y 轴的类目顺序与表格相反 —— ECharts 索引 0 在底部', () => {
    const result = forestResult();
    const { c, option } = make(result);
    const y = option['yAxis'] as { data: string[] };
    expect(c.rows().map((r) => r.label)).toEqual(['年龄', '性别 M', '心衰']);
    expect(y.data).toEqual(['心衰', '性别 M', '年龄']);
  });

  it('图里的数据行也跟着反过来，和类目轴对得上', () => {
    const { option } = make();
    const data = (option['series'] as Record<string, unknown>[])[0]['data'] as number[][];
    // 第 0 行对应类目轴的第 0 项（心衰，HR 0.45）
    expect(data[0][0]).toBe(0);
    expect(data[0][1]).toBeCloseTo(0.45, 6);
    expect(data[2][1]).toBeCloseTo(1.8, 6);
  });
});

describe('ForestPlot 取值钳制', () => {
  it('置信区间下界为 0 时钳到对数轴容得下的正数', () => {
    const result = forestResult();
    result.rows = [{ ...result.rows[0], estimate: 2, ci_lower: 0, ci_upper: 5, p: 0.01 }];
    const { option } = make(result);
    const data = (option['series'] as Record<string, unknown>[])[0]['data'] as number[][];
    expect(data[0][2]).toBeGreaterThan(0);
  });

  it('上界极大时钳到上限', () => {
    const result = forestResult();
    result.rows = [{ ...result.rows[0], estimate: 3, ci_lower: 1.1, ci_upper: 1e9, p: 0.01 }];
    const { option } = make(result);
    const data = (option['series'] as Record<string, unknown>[])[0]['data'] as number[][];
    expect(data[0][3]).toBeLessThanOrEqual(1e3);
  });

  it('被钳过的行在表格里标出来，不能让人以为区间真那么窄', () => {
    const result = forestResult();
    result.rows = [
      { ...result.rows[0], ci_lower: 0, ci_upper: 5 },
      { ...result.rows[1] },
    ];
    const { c } = make(result);
    expect(c.rows().map((r) => r.clamped)).toEqual([true, false]);
  });

  it('区间正常的行不标钳制', () => {
    expect(make().c.rows().every((r) => !r.clamped)).toBe(true);
  });
});

describe('ForestPlot 显著性', () => {
  it('p < 0.05 标为显著', () => {
    expect(make().c.rows().map((r) => r.significant)).toEqual([true, false, true]);
  });

  it('极小的 p 显示为 <0.001', () => {
    expect(make().c.formatP(1e-8)).toBe('<0.001');
    expect(make().c.formatP(0.0312)).toBe('0.031');
  });

  it('效应量极大或极小时用科学记数法，避免显示成 0.000', () => {
    const { c } = make();
    expect(c.formatEffect(0.0004)).toBe('4.00e-4');
    expect(c.formatEffect(1234)).toBe('1.23e+3');
    expect(c.formatEffect(1.5)).toBe('1.500');
  });

  it('非有限的效应量显示为破折号而不是 NaN', () => {
    expect(make().c.formatEffect(Number.POSITIVE_INFINITY)).toBe('—');
    expect(make().c.formatEffect(Number.NaN)).toBe('—');
  });
});

describe('ForestPlot 说明文字', () => {
  it('未加权时不出设计说明', () => {
    expect(make().c.designNote()).toBeNull();
  });

  it('只有权重没有分层时明确说方差会偏小', () => {
    const { c } = make(forestResult({
      weighted: true,
      design: { n_strata: 0, n_psu: 0, df: 0, approximate: true },
    }));
    expect(c.designNote()).toContain('偏小');
  });

  it('完整设计时把层数、PSU 数与自由度亮出来', () => {
    const { c } = make(forestResult({
      weighted: true,
      design: { n_strata: 15, n_psu: 30, df: 15, approximate: false },
    }));
    const note = c.designNote()!;
    expect(note).toContain('15 层');
    expect(note).toContain('30 个 PSU');
    expect(note).toContain('自由度 15');
  });

  it('未用插补时不出插补说明', () => {
    expect(make().c.imputationNote()).toBeNull();
    expect(make().c.hasFmi()).toBe(false);
  });

  it('用了插补时报出份数、轮数与补全比例', () => {
    const { c } = make(forestResult({
      imputation: {
        m: 5, iterations: 10, imputed_rows: 40, total_rows: 300,
        columns: [{ name: 'person.age_at_index', pct: 13.3 }],
      } as never,
    }));
    const note = c.imputationNote()!;
    expect(note).toContain('5 份');
    expect(note).toContain('10 轮');
    expect(note).toContain('40 / 300');
  });

  it('缺失信息占比为空时显示破折号', () => {
    const { c } = make();
    expect(c.formatFmi(null)).toBe('—');
    expect(c.formatFmi(undefined)).toBe('—');
    expect(c.formatFmi(0.42)).toBe('0.420');
  });

  it('无障碍描述带上项数与参考线位置', () => {
    const label = make().c.ariaLabel();
    expect(label).toContain('共 3 项');
    expect(label).toContain('HR = 1');
  });

  it('图高随行数增长，保证每行有落脚空间', () => {
    const one = forestResult(); one.rows = one.rows.slice(0, 1);
    const short = parseInt(make(one).c.chartHeight(), 10);
    const tall = parseInt(make().c.chartHeight(), 10);
    expect(tall).toBeGreaterThan(short);
  });
});

describe('ForestPlot ROC', () => {
  it('没有 ROC 数据时不显示', () => {
    expect(make().c.hasRoc()).toBe(false);
  });

  it('有 ROC 时画对角参考线和曲线本体', () => {
    const { c } = make(forestResult({
      roc: { fpr: [0, 0.2, 1], tpr: [0, 0.7, 1], auc: 0.81 } as never,
    }));
    expect(c.hasRoc()).toBe(true);
    const series = (c.rocOption() as Record<string, unknown>)['series'] as Record<string, unknown>[];
    expect(series.map((s) => s['name'])).toEqual(['随机猜测', 'ROC']);
    expect(series[0]['data']).toEqual([[0, 0], [1, 1]]);
    expect(series[1]['data']).toEqual([[0, 0], [0.2, 0.7], [1, 1]]);
  });
});
