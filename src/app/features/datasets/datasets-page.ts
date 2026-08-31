import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { httpResource } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

import { ApiClient } from '../../core/api.client';
import { JobTracker } from '../../core/job-tracker';
import type { DatasetSummary } from '../../core/api.types';

@Component({
  selector: 'app-datasets-page',
  templateUrl: './datasets-page.html',
  styleUrl: './datasets-page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DatasetsPage {
  private readonly api = inject(ApiClient);
  private readonly tracker = inject(JobTracker);

  readonly datasets = httpResource<DatasetSummary[]>(() => '/api/datasets', {
    defaultValue: [],
  });

  readonly importingId = signal<string | null>(null);
  readonly importStage = signal<string>('');
  readonly importError = signal<string | null>(null);

  async runImport(dataset: DatasetSummary): Promise<void> {
    this.importingId.set(dataset.id);
    this.importStage.set('提交中');
    this.importError.set(null);
    try {
      const job = await firstValueFrom(this.api.importDataset(dataset.id));
      const done = await this.tracker.watch(job.id, (j) => this.importStage.set(j.stage));
      if (done.status === 'failed') {
        this.importError.set(done.error ?? '导入失败');
      }
      this.datasets.reload();
    } catch (err) {
      this.importError.set(err instanceof Error ? err.message : String(err));
    } finally {
      this.importingId.set(null);
    }
  }
}
