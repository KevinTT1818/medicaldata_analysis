import { Injectable, computed, signal } from '@angular/core';

/**
 * 图表配色与排版令牌。
 *
 * 分类色用固定顺序、经校验的八槽序列（不循环、不按排名分配），
 * 深色模式是为深色底重新取的步阶，不是浅色值的机械反转。
 * 两种模式下都通过了明度带、彩度下限、色觉障碍分离度与常视觉分离度校验。
 * 浅色模式下 aqua / yellow 对白底的对比度低于 3:1，所以每张图都必须
 * 同时提供图例与表格视图作为补偿 —— 这是校验器的硬性要求。
 */
const SERIES_LIGHT = [
  '#2a78d6', // 1 blue
  '#eb6834', // 2 orange
  '#1baf7a', // 3 aqua
  '#eda100', // 4 yellow
  '#e87ba4', // 5 magenta
  '#008300', // 6 green
  '#4a3aa7', // 7 violet
  '#e34948', // 8 red
];

const SERIES_DARK = [
  '#3987e5', '#d95926', '#199e70', '#c98500',
  '#d55181', '#008300', '#9085e9', '#e66767',
];

export interface ChartTokens {
  ink: string;
  ink2: string;
  muted: string;
  rule: string;
  surface: string;
  accent: string;
  amber: string;
  rose: string;
  series: string[];
  fontFamily: string;
  isDark: boolean;
}

@Injectable({ providedIn: 'root' })
export class ChartTheme {
  /** 系统配色方案变化时自增，让依赖它的 computed 重算。 */
  private readonly revision = signal(0);

  constructor() {
    const query = matchMedia('(prefers-color-scheme: dark)');
    query.addEventListener('change', () => this.revision.update((v) => v + 1));
  }

  readonly tokens = computed<ChartTokens>(() => {
    this.revision();
    const style = getComputedStyle(document.documentElement);
    const read = (name: string) => style.getPropertyValue(name).trim();
    const isDark = matchMedia('(prefers-color-scheme: dark)').matches;

    return {
      ink: read('--ink'),
      ink2: read('--ink-2'),
      muted: read('--muted'),
      rule: read('--rule'),
      surface: read('--surface'),
      accent: read('--accent'),
      amber: read('--amber'),
      rose: read('--rose'),
      series: isDark ? SERIES_DARK : SERIES_LIGHT,
      fontFamily: read('--font-body') || 'sans-serif',
      isDark,
    };
  });

  /** 第 n 个分类色。固定顺序，超过八个不再生成新色（调用方应折叠或分面）。 */
  seriesColor(index: number): string {
    const palette = this.tokens().series;
    return palette[index] ?? palette[palette.length - 1];
  }
}
