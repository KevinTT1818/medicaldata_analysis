import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';

import { ChartTheme } from '../chart-theme';
import { EChart } from '../echart';
import type { DistributionPanel, DistributionResult } from '../../../core/api.types';

/** 单个变量的分布图：连续变量出直方图 + 箱线图，分类变量出分组条形图。 */
@Component({
  selector: 'app-distribution-figure',
  imports: [EChart],
  templateUrl: './distribution-figure.html',
  styleUrl: './distribution-panel.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DistributionFigure {
  readonly panel = input.required<DistributionPanel>();

  private readonly theme = inject(ChartTheme);

  readonly isContinuous = computed(() => this.panel().kind === 'continuous');
  readonly seriesNames = computed(() => this.panel().series.map((s) => s.name));
  readonly showLegend = computed(() => this.panel().series.length > 1);

  readonly title = computed(() => {
    const p = this.panel();
    return p.unit ? `${p.label}（${p.unit}）` : p.label;
  });

  readonly boxRows = computed(() =>
    this.panel().series
      .filter((s) => s.box)
      .map((s, i) => ({ name: s.name, box: s.box!, color: this.theme.seriesColor(i) })),
  );

  private axisStyle() {
    const t = this.theme.tokens();
    return { color: t.muted, fontFamily: t.fontFamily, fontSize: 11 };
  }

  private legendConfig() {
    const t = this.theme.tokens();
    return this.showLegend()
      ? {
          data: this.seriesNames(), top: 0, left: 0,
          icon: 'roundRect', itemWidth: 12, itemHeight: 8,
          textStyle: { color: t.ink2, fontFamily: t.fontFamily, fontSize: 12 },
        }
      : { show: false };
  }

  readonly histogramOption = computed<EChartsCoreOption>(() => {
    const p = this.panel();
    const t = this.theme.tokens();
    const axis = this.axisStyle();

    const edges = p.bin_edges ?? [];
    const labels = edges.slice(0, -1).map((lo, i) => {
      const hi = edges[i + 1];
      const digits = Math.abs(hi - lo) >= 10 ? 0 : 1;
      return `${lo.toFixed(digits)}–${hi.toFixed(digits)}`;
    });

    return {
      animation: false,
      grid: { left: 46, right: 14, top: this.showLegend() ? 34 : 12, bottom: 52 },
      legend: this.legendConfig(),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: t.surface,
        borderColor: t.rule,
        textStyle: { color: t.ink, fontFamily: t.fontFamily, fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { ...axis, rotate: 45, hideOverlap: true },
        axisLine: { lineStyle: { color: t.rule } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: '例数',
        nameTextStyle: { ...axis, align: 'left' },
        axisLabel: axis,
        axisLine: { show: false },
        splitLine: { lineStyle: { color: t.rule, type: 'dashed' } },
      },
      series: p.series.map((s, i) => ({
        name: s.name,
        type: 'bar',
        data: s.counts,
        // 相邻柱之间留 2px 底色缝隙
        barGap: '8%',
        barCategoryGap: '18%',
        itemStyle: {
          color: this.theme.seriesColor(i),
          borderRadius: [3, 3, 0, 0],
        },
      })),
    };
  });

  readonly boxOption = computed<EChartsCoreOption>(() => {
    const t = this.theme.tokens();
    const axis = this.axisStyle();
    const rows = this.boxRows();

    return {
      animation: false,
      grid: { left: 52, right: 14, top: 14, bottom: 34 },
      tooltip: {
        trigger: 'item',
        backgroundColor: t.surface,
        borderColor: t.rule,
        textStyle: { color: t.ink, fontFamily: t.fontFamily, fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: rows.map((r) => r.name),
        axisLabel: { ...axis, color: t.ink2, fontSize: 12 },
        axisLine: { lineStyle: { color: t.rule } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: axis,
        axisLine: { show: false },
        splitLine: { lineStyle: { color: t.rule, type: 'dashed' } },
      },
      series: [
        {
          type: 'boxplot',
          data: rows.map((r) => ({
            value: [r.box.min, r.box.q1, r.box.median, r.box.q3, r.box.max],
            itemStyle: { color: r.color, borderColor: r.color, opacity: 0.9 },
          })),
          boxWidth: [10, 44],
        },
        {
          type: 'scatter',
          name: '离群点',
          symbolSize: 6,
          itemStyle: { color: t.muted, opacity: 0.6 },
          data: rows.flatMap((r, i) => r.box.outliers.map((v) => [i, v])),
        },
      ],
    };
  });

  readonly categoricalOption = computed<EChartsCoreOption>(() => {
    const p = this.panel();
    const t = this.theme.tokens();
    const axis = this.axisStyle();

    return {
      animation: false,
      grid: { left: 46, right: 14, top: this.showLegend() ? 34 : 12, bottom: 46 },
      legend: this.legendConfig(),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: t.surface,
        borderColor: t.rule,
        textStyle: { color: t.ink, fontFamily: t.fontFamily, fontSize: 12 },
      },
      xAxis: {
        type: 'category',
        data: p.levels ?? [],
        axisLabel: { ...axis, color: t.ink2, fontSize: 12, hideOverlap: true },
        axisLine: { lineStyle: { color: t.rule } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: '例数',
        nameTextStyle: { ...axis, align: 'left' },
        axisLabel: axis,
        axisLine: { show: false },
        splitLine: { lineStyle: { color: t.rule, type: 'dashed' } },
      },
      series: p.series.map((s, i) => ({
        name: s.name,
        type: 'bar',
        data: s.counts,
        barGap: '8%',
        barCategoryGap: '34%',
        itemStyle: { color: this.theme.seriesColor(i), borderRadius: [3, 3, 0, 0] },
      })),
    };
  });
}

/** 分布结果的容器：把每个变量渲染成一块图。 */
@Component({
  selector: 'app-distribution-panel',
  imports: [DistributionFigure],
  template: `
    <div class="figures">
      @for (panel of result().panels; track panel.variable) {
        <app-distribution-figure [panel]="panel" />
      }
    </div>
    @if (result().grouped) {
      <p class="foot">按「{{ result().group_by_label }}」分组，共 {{ result().overall_n }} 例。</p>
    } @else {
      <p class="foot">总体 {{ result().overall_n }} 例。</p>
    }
  `,
  styles: `
    .figures { display: grid; gap: 22px; }
    @media (min-width: 1200px) { .figures { grid-template-columns: 1fr 1fr; } }
    .foot { margin: 16px 0 0; font-size: 13px; color: var(--muted); }
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DistributionPanelView {
  readonly result = input.required<DistributionResult>();
}
