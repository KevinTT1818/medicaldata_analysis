import { httpResource } from '@angular/common/http';
import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { firstValueFrom } from 'rxjs';

import { ApiClient } from '../../core/api.client';
import { ReportDraft } from '../../core/report-draft';
import { BaselineTable } from '../../shared/baseline-table/baseline-table';
import { DistributionPanelView } from '../../shared/charts/distribution-panel/distribution-panel';
import { ForestPlot } from '../../shared/charts/forest-plot/forest-plot';
import { KmCurve } from '../../shared/charts/km-curve/km-curve';
import type {
  AnalysisResult,
  BaselineTableResult,
  DistributionResult,
  ForestResult,
  KmCurveResult,
  MissingnessResult,
  ReportRun,
  ReproVerdict,
  SavedReport,
} from '../../core/api.types';

const VERDICT_LABEL: Record<ReproVerdict, string> = {
  match: '与基线一致',
  changed: '数字变了',
  failed: '跑不通',
  no_baseline: '无基线',
};

@Component({
  selector: 'app-reports-page',
  imports: [BaselineTable, DistributionPanelView, ForestPlot, KmCurve, RouterLink],
  templateUrl: './reports-page.html',
  styleUrl: './reports-page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ReportsPage {
  private readonly api = inject(ApiClient);
  readonly draft = inject(ReportDraft);

  readonly saved = httpResource<SavedReport[]>(() => '/api/reports', { defaultValue: [] });

  readonly run = signal<ReportRun | null>(null);
  readonly busy = signal(false);
  readonly busyLabel = signal('');
  readonly errorText = signal<string | null>(null);

  readonly canSave = computed(
    () => this.draft.title().trim().length > 0 && this.draft.count() > 0 && !this.busy(),
  );

  readonly verdictLabel = VERDICT_LABEL;

  readonly reproSummary = computed(() => {
    const r = this.run();
    if (!r) return null;
    const counts = { match: 0, changed: 0, failed: 0, no_baseline: 0 };
    for (const c of r.comparisons) counts[c.verdict] += 1;
    return { ...counts, total: r.comparisons.length, reproducible: r.reproducible };
  });

  comparisonFor(index: number) {
    return this.run()?.comparisons.find((c) => c.index === index) ?? null;
  }

  asBaseline(result: AnalysisResult | null): BaselineTableResult | null {
    return result?.kind === 'baseline_table' ? result : null;
  }
  asKm(result: AnalysisResult | null): KmCurveResult | null {
    return result?.kind === 'km_curve' ? result : null;
  }
  asForest(result: AnalysisResult | null): ForestResult | null {
    return result?.kind === 'forest' ? result : null;
  }
  asDistribution(result: AnalysisResult | null): DistributionResult | null {
    return result?.kind === 'distribution' ? result : null;
  }
  asMissingness(result: AnalysisResult | null): MissingnessResult | null {
    return result?.kind === 'missingness' ? result : null;
  }

  private async withBusy<T>(label: string, work: () => Promise<T>): Promise<T | null> {
    this.busy.set(true);
    this.busyLabel.set(label);
    this.errorText.set(null);
    try {
      return await work();
    } catch (err) {
      this.errorText.set(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      this.busy.set(false);
      this.busyLabel.set('');
    }
  }

  async save(): Promise<void> {
    const body = {
      title: this.draft.title().trim(),
      description: this.draft.description().trim() || null,
      sections: this.draft.sections(),
    };
    const editingId = this.draft.editingId();

    const result = await this.withBusy('保存并建立基线', () =>
      firstValueFrom(
        editingId ? this.api.updateReport(editingId, body) : this.api.saveReport(body),
      ),
    );
    if (result) {
      this.draft.load(result.id, result.title, result.description, result.sections);
      this.saved.reload();
      await this.runReport(result.id);
    }
  }

  async runReport(id: string): Promise<void> {
    const result = await this.withBusy('重跑全部小节', () =>
      firstValueFrom(this.api.runReport(id)),
    );
    if (result) this.run.set(result);
  }

  async rebaseline(id: string): Promise<void> {
    const result = await this.withBusy('以当前结果重设基线', () =>
      firstValueFrom(this.api.rebaseline(id)),
    );
    if (result) {
      this.saved.reload();
      await this.runReport(id);
    }
  }

  async remove(report: SavedReport): Promise<void> {
    await this.withBusy('删除', () => firstValueFrom(this.api.deleteReport(report.id)));
    if (this.draft.editingId() === report.id) this.draft.clear();
    if (this.run()?.report.id === report.id) this.run.set(null);
    this.saved.reload();
  }

  load(report: SavedReport): void {
    this.draft.load(report.id, report.title, report.description, report.sections);
    this.run.set(null);
  }

  newReport(): void {
    this.draft.clear();
    this.run.set(null);
  }

  /** 导出走浏览器打印。打印样式已把导航与操作按钮隐藏，选「另存为 PDF」即可。 */
  print(): void {
    window.print();
  }
}
