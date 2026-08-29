import { httpResource } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  linkedSignal,
  signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';

import { ApiClient } from '../../core/api.client';
import { JobTracker } from '../../core/job-tracker';
import { ReportDraft } from '../../core/report-draft';
import { BaselineTable } from '../../shared/baseline-table/baseline-table';
import { DistributionPanelView } from '../../shared/charts/distribution-panel/distribution-panel';
import { ForestPlot } from '../../shared/charts/forest-plot/forest-plot';
import { KmCurve } from '../../shared/charts/km-curve/km-curve';
import { SchemaForm, type SchemaFormChange } from '../../shared/schema-form/schema-form';
import type {
  AnalysisDescriptor,
  AnalysisResult,
  BaselineTableResult,
  DatasetSchema,
  DatasetSummary,
  DistributionResult,
  ForestResult,
  Job,
  KmCurveResult,
  MeasurementAgg,
  MissingnessResult,
  SavedCohort,
} from '../../core/api.types';
import { MEASUREMENT_AGG_LABELS } from '../../core/api.types';

@Component({
  selector: 'app-workbench-page',
  imports: [BaselineTable, DistributionPanelView, ForestPlot, KmCurve, SchemaForm],
  templateUrl: './workbench-page.html',
  styleUrl: './workbench-page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WorkbenchPage {
  private readonly api = inject(ApiClient);
  private readonly tracker = inject(JobTracker);
  readonly reportDraft = inject(ReportDraft);

  readonly datasets = httpResource<DatasetSummary[]>(() => '/api/datasets', {
    defaultValue: [],
  });

  readonly registry = httpResource<AnalysisDescriptor[]>(
    () => '/api/analyses/registry',
    { defaultValue: [] },
  );

  /** 默认选中第一个已导入的数据集；用户已选的在列表刷新后保持不变。 */
  readonly datasetId = linkedSignal<DatasetSummary[], string | null>({
    source: () => this.datasets.value(),
    computation: (list, prev) => {
      const imported = list.filter((d) => d.imported);
      if (prev?.value && imported.some((d) => d.id === prev.value)) return prev.value;
      return imported[0]?.id ?? null;
    },
  });

  readonly schema = httpResource<DatasetSchema>(() => {
    const id = this.datasetId();
    return id ? `/api/datasets/${id}/schema` : undefined;
  });

  readonly variables = computed(() => this.schema.value()?.variables ?? []);

  readonly analysisId = linkedSignal<AnalysisDescriptor[], string | null>({
    source: () => this.registry.value(),
    computation: (list, prev) =>
      prev?.value && list.some((a) => a.id === prev.value) ? prev.value : (list[0]?.id ?? null),
  });

  readonly currentAnalysis = computed(
    () => this.registry.value().find((a) => a.id === this.analysisId()) ?? null,
  );

  readonly cohorts = httpResource<SavedCohort[]>(() => {
    const id = this.datasetId();
    return id ? `/api/cohorts?dataset=${id}` : undefined;
  }, { defaultValue: [] });

  /** 换数据集时清掉已选队列 —— 队列绑定在数据集上。 */
  readonly cohortId = linkedSignal<string | null, string | null>({
    source: () => this.datasetId(),
    computation: () => null,
  });

  readonly currentCohort = computed(
    () => this.cohorts.value().find((c) => c.id === this.cohortId()) ?? null,
  );

  /** 一人一项多值时的取值策略。横断面数据无影响，时序数据会改变结论。 */
  readonly measurementAgg = signal<MeasurementAgg>('first');
  readonly aggOptions = Object.entries(MEASUREMENT_AGG_LABELS) as [MeasurementAgg, string][];

  /** 数据集里有没有一人多值的测量 —— 由后端判定，没有就不拿这个选项打扰用户。 */
  readonly hasRepeatedMeasures = computed(
    () => this.schema.value()?.has_repeated_measures ?? false,
  );

  readonly hiddenVariables = computed(() => this.schema.value()?.hidden_variables ?? 0);

  readonly params = signal<Record<string, unknown>>({});
  readonly paramsValid = signal(false);

  readonly job = signal<Job | null>(null);
  readonly result = signal<AnalysisResult | null>(null);
  readonly warnings = signal<string[]>([]);
  readonly errorText = signal<string | null>(null);
  readonly running = signal(false);

  readonly canRun = computed(
    () =>
      !!this.datasetId() &&
      !!this.analysisId() &&
      this.paramsValid() &&
      !this.running() &&
      !this.schema.isLoading(),
  );

  readonly fingerprint = computed(() => this.job()?.spec.fingerprint ?? null);

  // 按结果类型分发到对应的渲染组件
  readonly baselineResult = computed<BaselineTableResult | null>(() =>
    this.result()?.kind === 'baseline_table' ? (this.result() as BaselineTableResult) : null,
  );
  readonly missingnessResult = computed<MissingnessResult | null>(() =>
    this.result()?.kind === 'missingness' ? (this.result() as MissingnessResult) : null,
  );
  readonly kmResult = computed<KmCurveResult | null>(() =>
    this.result()?.kind === 'km_curve' ? (this.result() as KmCurveResult) : null,
  );
  readonly forestResult = computed<ForestResult | null>(() =>
    this.result()?.kind === 'forest' ? (this.result() as ForestResult) : null,
  );
  readonly distributionResult = computed<DistributionResult | null>(() =>
    this.result()?.kind === 'distribution' ? (this.result() as DistributionResult) : null,
  );

  readonly resultTitle = computed(() => this.currentAnalysis()?.label ?? '分析结果');

  onParamsChanged(change: SchemaFormChange): void {
    this.params.set(change.value);
    this.paramsValid.set(change.valid);
  }

  onDatasetChange(value: string): void {
    this.datasetId.set(value);
    this.clearResult();
  }

  onAnalysisChange(value: string): void {
    this.analysisId.set(value);
    this.clearResult();
  }

  onCohortChange(value: string): void {
    this.cohortId.set(value || null);
    this.clearResult();
  }

  /** 上一次「加入报告」用的指纹，避免同一结果被重复加入。 */
  readonly addedFingerprint = signal<string | null>(null);

  readonly alreadyInReport = computed(
    () => this.fingerprint() !== null && this.addedFingerprint() === this.fingerprint(),
  );

  /** 把当前这次分析的**定义**加入报告草稿（不是加结果，报告要能重跑）。 */
  addToReport(): void {
    const dataset = this.datasetId();
    const analysis = this.analysisId();
    const fp = this.fingerprint();
    if (!dataset || !analysis || !fp) return;

    this.reportDraft.add({
      title: `${this.currentAnalysis()?.label ?? analysis}`,
      note: null,
      dataset,
      analysis,
      params: this.params(),
      cohort_id: this.cohortId(),
      cohort: null,
      measurement_agg: this.measurementAgg(),
    });
    this.addedFingerprint.set(fp);
  }

  onAggChange(value: string): void {
    this.measurementAgg.set(value as MeasurementAgg);
    this.clearResult();
  }

  private clearResult(): void {
    this.addedFingerprint.set(null);
    this.result.set(null);
    this.job.set(null);
    this.warnings.set([]);
    this.errorText.set(null);
  }

  async run(): Promise<void> {
    const dataset = this.datasetId();
    const analysis = this.analysisId();
    if (!dataset || !analysis) return;

    this.running.set(true);
    this.errorText.set(null);
    this.warnings.set([]);
    this.result.set(null);

    try {
      const submitted = await firstValueFrom(
        this.api.submitAnalysis(
          dataset, analysis, this.params(), this.cohortId(), this.measurementAgg()),
      );
      this.job.set(submitted);

      const done = await this.tracker.watch(submitted.id, (j) => this.job.set(j));
      if (done.status === 'failed') {
        this.errorText.set(done.error ?? '分析失败');
        return;
      }
      const envelope = await firstValueFrom(this.api.jobResult(done.id));
      this.result.set(envelope.result);
      this.warnings.set(done.warnings);
    } catch (err) {
      this.errorText.set(err instanceof Error ? err.message : String(err));
    } finally {
      this.running.set(false);
    }
  }
}
