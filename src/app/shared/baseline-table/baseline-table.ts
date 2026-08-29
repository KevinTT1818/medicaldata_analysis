import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import { OVERALL, type BaselineRow, type BaselineTableResult } from '../../core/api.types';

@Component({
  selector: 'app-baseline-table',
  templateUrl: './baseline-table.html',
  styleUrl: './baseline-table.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class BaselineTable {
  readonly result = input.required<BaselineTableResult>();

  readonly overallKey = OVERALL;

  /** 表头：总体列 + 各分组列。 */
  readonly columns = computed(() => {
    const r = this.result();
    return [
      { key: OVERALL, label: '总体', n: r.overall_n, weightedPct: null as number | null },
      ...r.groups.map((g) => ({
        key: g.key, label: g.label, n: g.n, weightedPct: g.weighted_pct,
      })),
    ];
  });

  readonly hasGroups = computed(() => this.result().groups.length > 0);

  /** 有分组就有检验。加权时走设计校正 Wald，非加权时走 t / 卡方。 */
  readonly showTests = computed(() => this.hasGroups());

  /** 加权且带完整抽样设计时，把自由度亮出来 —— 它决定了 p 值有多可信。 */
  readonly designNote = computed(() => {
    const r = this.result();
    if (!r.weighted || !r.design) return null;
    if (r.design.approximate) {
      return '只有权重、没有分层与初级抽样单元，标准误按有放回抽样近似，会偏小';
    }
    return `${r.design.n_strata} 层 · ${r.design.n_psu} 个 PSU · 设计自由度 ${r.design.df}`;
  });

  rowLabel(row: BaselineRow): string {
    const unit = row.unit ? ` (${row.unit})` : '';
    const weighted = this.result().weighted;
    const stat =
      row.stat === 'mean_sd' ? (weighted ? '加权均值 ± SD' : '均值 ± SD')
      : row.stat === 'median_iqr' ? (weighted ? '加权中位数 (IQR)' : '中位数 (IQR)')
      : (weighted ? 'n (加权 %)' : 'n (%)');
    return `${row.label}${unit}，${stat}`;
  }

  formatP(p: number | null): string {
    if (p === null) return '—';
    if (p < 0.001) return '<0.001';
    return p.toFixed(3);
  }

  isSignificant(row: BaselineRow): boolean {
    const p = row.p_adj ?? row.p;
    return p !== null && p < 0.05;
  }
}
