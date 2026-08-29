import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  afterNextRender,
  effect,
  inject,
  input,
  viewChild,
} from '@angular/core';

import * as echarts from 'echarts/core';
import { BarChart, BoxplotChart, CustomChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsType } from 'echarts/core';

echarts.use([
  LineChart, BarChart, ScatterChart, BoxplotChart, CustomChart,
  GridComponent, TooltipComponent, LegendComponent, MarkLineComponent,
  CanvasRenderer,
]);

/**
 * ECharts 薄封装。
 *
 * 初始化必须放在 afterNextRender 里 —— ECharts 依赖 document 和 canvas，
 * 在服务端渲染阶段执行会直接报错。本项目已关闭 SSR，这里仍保留守卫，
 * 以免将来重新开启时踩坑。
 */
@Component({
  selector: 'app-echart',
  template: '<div #host [style.height]="height()" role="img" [attr.aria-label]="ariaLabel()"></div>',
  styles: ':host { display: block; } div { width: 100%; }',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class EChart {
  readonly option = input.required<echarts.EChartsCoreOption>();
  readonly height = input('340px');
  readonly ariaLabel = input('数据图表');

  private readonly host = viewChild.required<ElementRef<HTMLDivElement>>('host');
  private readonly destroyRef = inject(DestroyRef);
  private chart: EChartsType | null = null;

  constructor() {
    afterNextRender(() => {
      const element = this.host().nativeElement;
      this.chart = echarts.init(element, undefined, { renderer: 'canvas' });
      this.chart.setOption(this.option(), true);

      const observer = new ResizeObserver(() => this.chart?.resize());
      observer.observe(element);

      this.destroyRef.onDestroy(() => {
        observer.disconnect();
        this.chart?.dispose();
        this.chart = null;
      });
    });

    // option 是 computed 出来的，主题切换时也会重算，这里统一重绘
    effect(() => {
      const option = this.option();
      this.chart?.setOption(option, true);
    });
  }
}
