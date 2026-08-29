import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';

import { ChartTheme } from '../chart-theme';
import { EChart } from '../echart';
import type { ForestResult } from '../../../core/api.types';

/** log 轴容不下 0 和负数；置信区间下界极小时钳到这个值并在表格里标出。 */
const LOG_FLOOR = 1e-3;
const LOG_CEIL = 1e3;

@Component({
  selector: 'app-forest-plot',
  imports: [EChart],
  templateUrl: './forest-plot.html',
  styleUrl: './forest-plot.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ForestPlot {
  readonly result = input.required<ForestResult>();

  private readonly theme = inject(ChartTheme);

  readonly rows = computed(() =>
    this.result().rows.map((r) => ({
      ...r,
      clamped: r.ci_lower < LOG_FLOOR || r.ci_upper > LOG_CEIL,
      significant: r.p < 0.05,
    })),
  );

  /** 图从下往上画，表格从上往下读，标签顺序要反过来才对得上。 */
  readonly chartLabels = computed(() => this.rows().map((r) => r.label).reverse());

  readonly phByLabel = computed(() => {
    const map = new Map<string, number>();
    for (const t of this.result().ph_test) map.set(t.label, t.p);
    return map;
  });

  readonly hasPh = computed(() => this.result().ph_test.length > 0);

  /** 加权拟合时把设计参数亮出来 —— 自由度决定了置信区间有多宽。 */
  readonly designNote = computed(() => {
    const r = this.result();
    if (!r.weighted) return null;
    if (!r.design || r.design.approximate) {
      return '只有抽样权重、没有分层与初级抽样单元，方差按有放回抽样近似，会偏小';
    }
    return (
      `置信区间来自设计校正的三明治方差 · ${r.design.n_strata} 层 · ` +
      `${r.design.n_psu} 个 PSU · 自由度 ${r.design.df}`
    );
  });
  readonly hasRoc = computed(() => !!this.result().roc?.fpr.length);

  readonly chartHeight = computed(() => `${Math.max(200, this.rows().length * 40 + 84)}px`);

  readonly ariaLabel = computed(() => {
    const r = this.result();
    return (
      `${r.model}森林图，共 ${r.rows.length} 项。` +
      `参考线在 ${r.effect_label} = ${r.null_value}。具体数值见右侧表格。`
    );
  });

  formatP(p: number): string {
    return p < 0.001 ? '<0.001' : p.toFixed(3);
  }

  formatEffect(value: number): string {
    if (!Number.isFinite(value)) return '—';
    return value >= 100 || value < 0.01 ? value.toExponential(2) : value.toFixed(3);
  }

  readonly option = computed<EChartsCoreOption>(() => {
    const result = this.result();
    const t = this.theme.tokens();
    const rows = this.rows();
    const axis = { color: t.muted, fontFamily: t.fontFamily, fontSize: 11 };

    const clamp = (v: number) => Math.min(Math.max(v, LOG_FLOOR), LOG_CEIL);
    // reverse：ECharts 的类目轴索引 0 在底部
    const data = [...rows].reverse().map((r, i) => [
      i, clamp(r.estimate), clamp(r.ci_lower), clamp(r.ci_upper), r.p,
    ]);

    const positive = this.theme.seriesColor(1);   // 风险升高
    const negative = this.theme.seriesColor(0);   // 风险降低
    const neutral = t.muted;

    return {
      animation: false,
      grid: { left: 8, right: 24, top: 16, bottom: 46, containLabel: true },
      tooltip: {
        trigger: 'item',
        backgroundColor: t.surface,
        borderColor: t.rule,
        textStyle: { color: t.ink, fontFamily: t.fontFamily, fontSize: 12 },
        formatter: (p: unknown) => {
          const v = (p as { value: number[] }).value;
          const label = this.chartLabels()[v[0]];
          return (
            `${label}<br>${result.effect_label} ${v[1].toFixed(3)}` +
            ` (${(result.conf_level * 100).toFixed(0)}% CI ${v[2].toFixed(3)}–${v[3].toFixed(3)})` +
            `<br>p = ${this.formatP(v[4])}`
          );
        },
      },
      xAxis: {
        type: 'log',
        name: `${result.effect_label}（对数刻度）`,
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: axis,
        axisLabel: axis,
        axisLine: { lineStyle: { color: t.rule } },
        splitLine: { lineStyle: { color: t.rule, type: 'dashed' } },
      },
      yAxis: {
        type: 'category',
        data: this.chartLabels(),
        axisLabel: { color: t.ink2, fontFamily: t.fontFamily, fontSize: 12 },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series: [
        {
          type: 'custom',
          data,
          markLine: {
            silent: true,
            symbol: 'none',
            label: { show: false },
            lineStyle: { color: t.ink2, width: 1, type: 'solid' },
            data: [{ xAxis: result.null_value }],
          },
          renderItem: (_params: unknown, api: {
            value: (i: number) => number;
            coord: (v: number[]) => number[];
          }) => {
            const category = api.value(0);
            const [x, y] = api.coord([api.value(1), category]);
            const [xLow] = api.coord([api.value(2), category]);
            const [xHigh] = api.coord([api.value(3), category]);

            const estimate = api.value(1);
            const lower = api.value(2);
            const upper = api.value(3);
            // 置信区间跨过无效应线 = 不显著，画成中性灰
            const crosses = lower <= result.null_value && upper >= result.null_value;
            const color = crosses
              ? neutral
              : estimate > result.null_value
                ? positive
                : negative;

            const cap = 4;
            return {
              type: 'group',
              children: [
                { type: 'line', shape: { x1: xLow, y1: y, x2: xHigh, y2: y },
                  style: { stroke: color, lineWidth: 2 } },
                { type: 'line', shape: { x1: xLow, y1: y - cap, x2: xLow, y2: y + cap },
                  style: { stroke: color, lineWidth: 2 } },
                { type: 'line', shape: { x1: xHigh, y1: y - cap, x2: xHigh, y2: y + cap },
                  style: { stroke: color, lineWidth: 2 } },
                { type: 'rect', shape: { x: x - 5, y: y - 5, width: 10, height: 10, r: 2 },
                  style: { fill: color, stroke: t.surface, lineWidth: 2 } },
              ],
            };
          },
        },
      ],
    };
  });

  readonly rocOption = computed<EChartsCoreOption>(() => {
    const roc = this.result().roc;
    const t = this.theme.tokens();
    const axis = { color: t.muted, fontFamily: t.fontFamily, fontSize: 11 };
    const color = this.theme.seriesColor(0);

    return {
      animation: false,
      grid: { left: 48, right: 16, top: 16, bottom: 42 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: t.surface,
        borderColor: t.rule,
        textStyle: { color: t.ink, fontFamily: t.fontFamily, fontSize: 12 },
      },
      xAxis: {
        type: 'value', min: 0, max: 1,
        name: '1 − 特异度', nameLocation: 'middle', nameGap: 26, nameTextStyle: axis,
        axisLabel: axis, axisLine: { lineStyle: { color: t.rule } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value', min: 0, max: 1,
        name: '灵敏度', nameTextStyle: { ...axis, align: 'left' },
        axisLabel: axis, axisLine: { show: false },
        splitLine: { lineStyle: { color: t.rule, type: 'dashed' } },
      },
      series: [
        {
          name: '随机猜测', type: 'line', symbol: 'none', silent: true,
          lineStyle: { color: t.muted, width: 1, type: 'dashed' },
          data: [[0, 0], [1, 1]],
        },
        {
          name: 'ROC', type: 'line', symbol: 'none',
          lineStyle: { color, width: 2 },
          areaStyle: { color, opacity: t.isDark ? 0.14 : 0.1 },
          data: (roc?.fpr ?? []).map((x, i) => [x, roc!.tpr[i]]),
        },
      ],
    };
  });
}
