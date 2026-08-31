import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import type { EChartsCoreOption } from 'echarts/core';

import { ChartTheme } from '../chart-theme';
import { EChart } from '../echart';
import type { KmCurveResult } from '../../../core/api.types';

@Component({
  selector: 'app-km-curve',
  imports: [EChart],
  templateUrl: './km-curve.html',
  styleUrl: './km-curve.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KmCurve {
  readonly result = input.required<KmCurveResult>();

  private readonly theme = inject(ChartTheme);

  readonly hasGroups = computed(() => this.result().series.length > 1);

  readonly summary = computed(() =>
    this.result().series.map((s, i) => ({
      ...s,
      color: this.theme.seriesColor(i),
      medianText: s.median_survival === null ? '未达到' : `${s.median_survival} 天`,
    })),
  );

  readonly logrankText = computed(() => {
    const p = this.result().logrank_p;
    if (p === null) return null;
    return p < 0.001 ? '<0.001' : p.toFixed(3);
  });

  readonly ariaLabel = computed(() => {
    const r = this.result();
    const parts = r.series.map(
      (s) => `${s.name}组 ${s.n} 例、${s.events} 例发生事件`,
    );
    return `Kaplan-Meier 生存曲线。${parts.join('；')}。详细数值见下方表格。`;
  });

  readonly option = computed<EChartsCoreOption>(() => {
    const r = this.result();
    const t = this.theme.tokens();
    const axis = { color: t.muted, fontFamily: t.fontFamily, fontSize: 11 };

    const series = r.series.flatMap((s, i) => {
      const color = this.theme.seriesColor(i);
      const stack = `ci-${i}`;
      return [
        // 置信带下沿：透明线，只作为堆叠基准
        {
          name: `__ci_lo_${i}`, type: 'line', stack, step: 'end', symbol: 'none',
          silent: true, lineStyle: { opacity: 0 }, tooltip: { show: false },
          data: s.t.map((x, j) => [x, s.ci_lower[j]]),
        },
        // 置信带本体：堆在下沿之上，高度为上下沿之差
        {
          name: `__ci_hi_${i}`, type: 'line', stack, step: 'end', symbol: 'none',
          silent: true, lineStyle: { opacity: 0 }, tooltip: { show: false },
          areaStyle: { color, opacity: t.isDark ? 0.16 : 0.12 },
          data: s.t.map((x, j) => [x, s.ci_upper[j] - s.ci_lower[j]]),
        },
        // 生存曲线本体。step:'end' 是关键 —— KM 是阶梯函数，
        // 用平滑折线连接会暗示事件时点之间存在连续下降，与估计量定义矛盾。
        {
          name: s.name, type: 'line', step: 'end', symbol: 'none',
          lineStyle: { width: 2, color }, itemStyle: { color },
          emphasis: { focus: 'series' },
          data: s.t.map((x, j) => [x, s.survival[j]]),
        },
        // 删失标记：竖向短划
        {
          name: `__censor_${i}`, type: 'scatter', silent: true,
          symbol: 'rect', symbolSize: [1.5, 9],
          itemStyle: { color }, tooltip: { show: false },
          data: s.censor_t.map((x, j) => [x, s.censor_s[j]]),
        },
      ];
    });

    return {
      animation: false,
      color: r.series.map((_, i) => this.theme.seriesColor(i)),
      grid: { left: 52, right: 18, top: this.hasGroups() ? 38 : 16, bottom: 44 },
      legend: this.hasGroups()
        ? {
            data: r.series.map((s) => s.name),
            top: 0, left: 0, icon: 'roundRect', itemWidth: 14, itemHeight: 3,
            textStyle: { color: t.ink2, fontFamily: t.fontFamily, fontSize: 12 },
          }
        : { show: false },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'line', lineStyle: { color: t.rule } },
        backgroundColor: t.surface,
        borderColor: t.rule,
        textStyle: { color: t.ink, fontFamily: t.fontFamily, fontSize: 12 },
        formatter: (params: unknown) => {
          const rows = (params as { seriesName: string; value: [number, number]; color: string }[])
            .filter((p) => !p.seriesName.startsWith('__'));
          if (!rows.length) return '';
          const head = `${rows[0].value[0]} ${r.time_unit}`;
          const body = rows
            .map(
              (p) =>
                `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${p.color};margin-right:6px"></span>` +
                `${p.seriesName} ${(p.value[1] * 100).toFixed(1)}%`,
            )
            .join('<br>');
          return `${head}<br>${body}`;
        },
      },
      xAxis: {
        type: 'value',
        name: `随访时间（${r.time_unit}）`,
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: axis,
        axisLabel: axis,
        axisLine: { lineStyle: { color: t.rule } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        name: '生存概率',
        nameTextStyle: { ...axis, align: 'left' },
        axisLabel: { ...axis, formatter: (v: number) => `${Math.round(v * 100)}%` },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: t.rule, type: 'dashed' } },
      },
      series,
    };
  });
}
