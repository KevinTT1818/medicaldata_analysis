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
import { CohortNode } from '../../shared/cohort-builder/cohort-node';
import { ConsortFlow } from '../../shared/cohort-builder/consort-flow';
import {
  emptyGroup,
  type CohortPreview,
  type DatasetSchema,
  type DatasetSummary,
  type FilterNode,
  type OperatorMeta,
  type SavedCohort,
} from '../../core/api.types';

const NO_VALUE_OPS = new Set(['is_null', 'not_null']);

/** 条件是否填完整。填了一半就发预览请求只会换来 400。 */
function isComplete(node: FilterNode): boolean {
  if (node.kind === 'group') return node.children.every(isComplete);
  if (NO_VALUE_OPS.has(node.op)) return true;
  const value = node.value;
  if (node.op === 'between') {
    return Array.isArray(value) && value.length === 2 && value.every((v) => v !== null && v !== '');
  }
  if (node.op === 'in' || node.op === 'not_in') {
    return Array.isArray(value) && value.length > 0;
  }
  return value !== null && value !== undefined && value !== '';
}

@Component({
  selector: 'app-cohorts-page',
  imports: [CohortNode, ConsortFlow],
  templateUrl: './cohorts-page.html',
  styleUrl: './cohorts-page.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CohortsPage {
  private readonly api = inject(ApiClient);

  readonly datasets = httpResource<DatasetSummary[]>(() => '/api/datasets', {
    defaultValue: [],
  });

  readonly operators = httpResource<OperatorMeta>(() => '/api/cohorts/operators');

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

  readonly saved = httpResource<SavedCohort[]>(() => {
    const id = this.datasetId();
    return id ? `/api/cohorts?dataset=${id}` : undefined;
  }, { defaultValue: [] });

  /** 换数据集时条件树作废——变量 ID 不通用。 */
  readonly definition = linkedSignal<string | null, FilterNode>({
    source: () => this.datasetId(),
    computation: () => emptyGroup(),
  });

  readonly name = signal('');
  readonly description = signal('');
  readonly editingId = signal<string | null>(null);
  readonly saveError = signal<string | null>(null);
  readonly busy = signal(false);

  readonly complete = computed(() => isComplete(this.definition()));

  readonly preview = httpResource<CohortPreview>(() => {
    const dataset = this.datasetId();
    if (!dataset || !this.complete() || !this.operators.hasValue()) return undefined;
    return {
      url: '/api/cohorts/preview',
      method: 'POST',
      body: { dataset, definition: this.definition() },
    };
  });

  readonly canSave = computed(
    () => !!this.datasetId() && this.name().trim().length > 0 && this.complete() && !this.busy(),
  );

  readonly conditionCount = computed(() => {
    const count = (node: FilterNode): number =>
      node.kind === 'group' ? node.children.reduce((n, c) => n + count(c), 0) : 1;
    return count(this.definition());
  });

  onDatasetChange(value: string): void {
    this.datasetId.set(value);
    this.resetForm();
  }

  onDefinitionChange(node: FilterNode): void {
    this.definition.set(node);
  }

  resetForm(): void {
    this.definition.set(emptyGroup());
    this.name.set('');
    this.description.set('');
    this.editingId.set(null);
    this.saveError.set(null);
  }

  load(cohort: SavedCohort): void {
    this.definition.set(cohort.definition);
    this.name.set(cohort.name);
    this.description.set(cohort.description ?? '');
    this.editingId.set(cohort.id);
    this.saveError.set(null);
  }

  async save(): Promise<void> {
    const dataset = this.datasetId();
    if (!dataset) return;

    this.busy.set(true);
    this.saveError.set(null);
    const body = {
      dataset,
      name: this.name().trim(),
      description: this.description().trim() || null,
      definition: this.definition(),
    };

    try {
      const id = this.editingId();
      const result = id
        ? await firstValueFrom(this.api.updateCohort(id, body))
        : await firstValueFrom(this.api.saveCohort(body));
      this.editingId.set(result.id);
      this.saved.reload();
    } catch (err) {
      this.saveError.set(err instanceof Error ? err.message : String(err));
    } finally {
      this.busy.set(false);
    }
  }

  async remove(cohort: SavedCohort): Promise<void> {
    this.busy.set(true);
    try {
      await firstValueFrom(this.api.deleteCohort(cohort.id));
      if (this.editingId() === cohort.id) this.resetForm();
      this.saved.reload();
    } finally {
      this.busy.set(false);
    }
  }
}
