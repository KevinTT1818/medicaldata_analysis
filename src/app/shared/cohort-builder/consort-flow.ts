import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import type { CohortPreview } from '../../core/api.types';

/** 队列预览：命中人数、CONSORT 式入排流程、人口学摘要。 */
@Component({
  selector: 'app-consort-flow',
  templateUrl: './consort-flow.html',
  styleUrl: './consort-flow.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConsortFlow {
  readonly preview = input.required<CohortPreview>();

  readonly steps = computed(() =>
    this.preview().flow.steps.map((s) => ({
      ...s,
      // 因该变量缺失而被排除的人，和「不满足条件」是两回事，单独提示
      missingShare:
        s.n_excluded > 0 ? Math.round((s.n_excluded_missing / s.n_excluded) * 100) : 0,
      keepPct: s.n_before > 0 ? (s.n_after / s.n_before) * 100 : 0,
    })),
  );

  readonly totalMissingExcluded = computed(() =>
    this.preview().flow.steps.reduce((sum, s) => sum + s.n_excluded_missing, 0),
  );

  readonly barWidth = computed(() => {
    const p = this.preview();
    return p.total > 0 ? (p.n / p.total) * 100 : 0;
  });
}
