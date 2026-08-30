/**
 * 图表配色。
 *
 * 分类色是固定顺序、经校验的八槽序列。守两条：颜色跟着实体走而不是跟着
 * 排名走（筛掉一个系列不能让其余系列换色），以及超过八个不再生成新色 ——
 * 生成的色没经过色觉障碍分离度校验，看着像能用其实不能。
 */
import { TestBed } from '@angular/core/testing';

import { ChartTheme } from './chart-theme';
import { resetColorScheme, setPrefersDark } from '../../testing/theme';

describe('ChartTheme 分类色', () => {
  afterEach(() => resetColorScheme());

  it('同一个下标永远给同一个颜色', () => {
    const theme = TestBed.inject(ChartTheme);
    expect(theme.seriesColor(0)).toBe(theme.seriesColor(0));
    expect(theme.seriesColor(3)).toBe(theme.seriesColor(3));
  });

  it('前八个颜色互不相同', () => {
    const theme = TestBed.inject(ChartTheme);
    const colors = Array.from({ length: 8 }, (_, i) => theme.seriesColor(i));
    expect(new Set(colors).size).toBe(8);
  });

  it('超过八个不循环回第一个 —— 循环会让两个系列同色', () => {
    const theme = TestBed.inject(ChartTheme);
    expect(theme.seriesColor(8)).not.toBe(theme.seriesColor(0));
    expect(theme.seriesColor(8)).toBe(theme.seriesColor(7));
  });

  it('深色模式用另一套步阶，不是浅色值的机械反转', () => {
    const theme = TestBed.inject(ChartTheme);
    const light = Array.from({ length: 8 }, (_, i) => theme.seriesColor(i));
    setPrefersDark(true);
    const dark = Array.from({ length: 8 }, (_, i) => theme.seriesColor(i));
    expect(dark).not.toEqual(light);
    expect(new Set(dark).size).toBe(8);
  });

  it('切换配色方案时 tokens 会重算', () => {
    const theme = TestBed.inject(ChartTheme);
    expect(theme.tokens().isDark).toBe(false);
    setPrefersDark(true);
    expect(theme.tokens().isDark).toBe(true);
  });

  it('字体令牌读不到时有兜底，不会给 ECharts 传空串', () => {
    expect(TestBed.inject(ChartTheme).tokens().fontFamily).toBeTruthy();
  });
});
