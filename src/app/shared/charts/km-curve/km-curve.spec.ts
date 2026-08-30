/**
 * KM 曲线的 option 构建。
 *
 * 这里断言的是几条统计学上不能错的性质：曲线必须是阶梯、置信带靠堆叠实现、
 * y 轴锁死在 [0,1]。这些错了图还是能画出来，只是画的不是 Kaplan-Meier 估计量 ——
 * 光看截图很难发现，所以要用断言守住。
 */
import { TestBed } from '@angular/core/testing';

import { KmCurve } from './km-curve';
import { kmResult, kmSeries } from '../../../testing/fixtures';

type Series = Record<string, unknown> & { name: string; type: string };

function optionOf(result = kmResult()) {
  const fixture = TestBed.createComponent(KmCurve);
  fixture.componentRef.setInput('result', result);
  fixture.detectChanges();
  const option = fixture.componentInstance.option() as Record<string, unknown>;
  return { option, series: option['series'] as Series[], component: fixture.componentInstance };
}

describe('KmCurve option', () => {
  it('生存曲线必须是阶梯函数', () => {
    const { series } = optionOf();
    const curve = series.find((s) => s.name === '全体')!;
    expect(curve.type).toBe('line');
    // step:'end' —— 平滑折线会暗示事件时点之间存在连续下降，与估计量定义矛盾
    expect(curve['step']).toBe('end');
    expect(curve['symbol']).toBe('none');
  });

  it('置信带的上下沿也必须是阶梯，否则会和曲线错开', () => {
    const { series } = optionOf();
    for (const s of series.filter((x) => x.name.startsWith('__ci_'))) {
      expect(s['step']).toBe('end');
    }
  });

  it('置信带靠堆叠实现：上沿的数据是上下沿之差', () => {
    const result = kmResult();
    const { series } = optionOf(result);
    const s = result.series[0];
    const lo = series.find((x) => x.name === '__ci_lo_0')!['data'] as number[][];
    const hi = series.find((x) => x.name === '__ci_hi_0')!['data'] as number[][];

    expect(lo.map((p) => p[1])).toEqual(s.ci_lower);
    expect(hi.map((p) => p[1])).toEqual(s.ci_lower.map((v, i) => s.ci_upper[i] - v));
    // 同一个 stack 才能叠起来
    expect(series.find((x) => x.name === '__ci_lo_0')!['stack'])
      .toBe(series.find((x) => x.name === '__ci_hi_0')!['stack']);
  });

  it('辅助系列不参与交互，也不进 tooltip', () => {
    const { series } = optionOf();
    for (const s of series.filter((x) => x.name.startsWith('__'))) {
      expect(s['silent']).toBe(true);
      expect((s['tooltip'] as { show: boolean }).show).toBe(false);
    }
  });

  it('y 轴锁死在 0 到 1 —— 生存概率不会超出这个范围', () => {
    const { option } = optionOf();
    const y = option['yAxis'] as Record<string, unknown>;
    expect(y['min']).toBe(0);
    expect(y['max']).toBe(1);
  });

  it('x 轴标题带上时间单位', () => {
    const { option } = optionOf();
    expect((option['xAxis'] as { name: string }).name).toContain('天');
  });

  it('每组画四个系列：置信带上下沿、曲线、删失标记', () => {
    const { series } = optionOf(kmResult([kmSeries({ name: '男' }), kmSeries({ name: '女' })]));
    expect(series.length).toBe(8);
    expect(series.filter((s) => !s.name.startsWith('__')).map((s) => s.name))
      .toEqual(['男', '女']);
  });

  it('删失标记画在曲线上，用竖向短划', () => {
    const result = kmResult();
    const { series } = optionOf(result);
    const censor = series.find((s) => s.name === '__censor_0')!;
    expect(censor.type).toBe('scatter');
    expect(censor['symbol']).toBe('rect');
    expect(censor['data']).toEqual(
      result.series[0].censor_t.map((t, i) => [t, result.series[0].censor_s[i]]),
    );
  });

  it('单组时不显示图例，多组时列出组名', () => {
    expect((optionOf().option['legend'] as { show: boolean }).show).toBe(false);
    const grouped = optionOf(kmResult([kmSeries({ name: '男' }), kmSeries({ name: '女' })]));
    expect((grouped.option['legend'] as { data: string[] }).data).toEqual(['男', '女']);
  });

  it('关掉动画 —— 统计图不该有入场动效', () => {
    expect(optionOf().option['animation']).toBe(false);
  });
});

describe('KmCurve 文本', () => {
  it('中位生存未达到时显示「未达到」而不是 null', () => {
    const { component } = optionOf(kmResult([kmSeries({ name: '全体', median_survival: null })]));
    expect(component.summary()[0].medianText).toBe('未达到');
  });

  it('中位生存达到时带单位', () => {
    const { component } = optionOf();
    expect(component.summary()[0].medianText).toBe('180 天');
  });

  it('极小的 p 值显示为 <0.001 而不是 0.000', () => {
    const { component } = optionOf(
      { ...kmResult([kmSeries({ name: '男' }), kmSeries({ name: '女' })]), logrank_p: 1e-9 },
    );
    expect(component.logrankText()).toBe('<0.001');
  });

  it('没有分组时没有 log-rank 检验', () => {
    expect(optionOf().component.logrankText()).toBeNull();
  });

  it('无障碍描述里带上各组的例数与事件数', () => {
    const { component } = optionOf(
      kmResult([kmSeries({ name: '男', n: 120, events: 33 }), kmSeries({ name: '女', n: 98, events: 21 })]),
    );
    const label = component.ariaLabel();
    expect(label).toContain('男组 120 例、33 例发生事件');
    expect(label).toContain('女组 98 例、21 例发生事件');
  });
});
