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

  /** 加权模式下 p 值一律为空，整列不显示，免得让人以为是漏算了。 */
  readonly showTests = computed(() => this.hasGroups() && !this.result().weighted);

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
