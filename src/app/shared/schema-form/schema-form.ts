import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  input,
  linkedSignal,
  output,
  signal,
} from '@angular/core';

import type { VariableInfo } from '../../core/api.types';

type JsonSchema = Record<string, unknown>;

export type FieldKind =
  | 'variables'
  | 'variable'
  | 'enum'
  | 'boolean'
  | 'number'
  | 'text';

export interface FormField {
  name: string;
  kind: FieldKind;
  title: string;
  description: string;
  required: boolean;
  nullable: boolean;
  variableFilter: string;
  enumOptions: { value: string; label: string }[];
  min?: number;
  max?: number;
  step?: number;
}

export interface SchemaFormChange {
  value: Record<string, unknown>;
  valid: boolean;
}

/** 变量筛选器。后端用 x-variable-filter 声明，这里决定哪些变量能选。 */
function passesFilter(v: VariableInfo, filter: string): boolean {
  switch (filter) {
    case 'grouping':
      return v.kind !== 'continuous';
    case 'survival':
      return v.source === 'outcome' && v.has_survival;
    case 'binary':
      return v.kind === 'binary';
    case 'covariate':
      return v.source !== 'outcome';
    default:
      return true;
  }
}

/**
 * 由 JSON Schema 驱动的参数表单。
 *
 * 后端每个算子的 Params 会被转成 JSON Schema 下发，其中 x-widget 扩展字段
 * 标明哪些字符串其实是变量 ID。加一个新算子时前端不用改任何代码。
 */
@Component({
  selector: 'app-schema-form',
  templateUrl: './schema-form.html',
  styleUrl: './schema-form.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SchemaForm {
  readonly schema = input.required<JsonSchema>();
  readonly variables = input.required<VariableInfo[]>();
  /** 变化时上报当前参数与是否填齐必填项。 */
  readonly changed = output<SchemaFormChange>();

  readonly fields = computed<FormField[]>(() => {
    const schema = this.schema();
    const properties = (schema['properties'] ?? {}) as Record<string, JsonSchema>;
    const required = (schema['required'] ?? []) as string[];

    return Object.entries(properties).map(([name, spec]) => {
      const anyOf = (spec['anyOf'] ?? []) as JsonSchema[];
      const nullable = anyOf.some((s) => s['type'] === 'null');
      const widget = spec['x-widget'] as string | undefined;
      const enumValues = spec['enum'] as string[] | undefined;
      const labels = (spec['x-enum-labels'] ?? {}) as Record<string, string>;
      const type = spec['type'] as string | undefined;

      let kind: FieldKind = 'text';
      if (widget === 'variables') kind = 'variables';
      else if (widget === 'variable') kind = 'variable';
      else if (enumValues) kind = 'enum';
      else if (type === 'boolean') kind = 'boolean';
      else if (type === 'number' || type === 'integer') kind = 'number';

      return {
        name,
        kind,
        title: (spec['title'] as string) ?? name,
        description: (spec['description'] as string) ?? '',
        required: required.includes(name),
        nullable,
        variableFilter: (spec['x-variable-filter'] as string) ?? 'any',
        enumOptions: (enumValues ?? []).map((v) => ({ value: v, label: labels[v] ?? v })),
        min: spec['minimum'] as number | undefined,
        max: spec['maximum'] as number | undefined,
        step: type === 'integer' ? 1 : 0.005,
      };
    });
  });

  /** schema 或可选变量变化时重置为默认值。 */
  readonly state = linkedSignal<{ schema: JsonSchema; vars: VariableInfo[] }, Record<string, unknown>>({
    source: () => ({ schema: this.schema(), vars: this.variables() }),
    computation: ({ schema, vars }) => {
      const properties = (schema['properties'] ?? {}) as Record<string, JsonSchema>;
      const required = (schema['required'] ?? []) as string[];
      const initial: Record<string, unknown> = {};

      for (const [name, spec] of Object.entries(properties)) {
        const widget = spec['x-widget'];

        if (widget === 'variables') {
          initial[name] = [];
          continue;
        }
        if (widget === 'variable') {
          // 必填的单选变量直接选中第一个可选项。留 null 会让下拉框显示第一项
          // （浏览器跳过 disabled 的占位项）而模型仍是空，运行按钮灰着却看不出原因。
          const filter = (spec['x-variable-filter'] as string) ?? 'any';
          const first = vars.find((v) => passesFilter(v, filter));
          initial[name] = required.includes(name) ? (first?.id ?? null) : null;
          continue;
        }
        initial[name] = spec['default'] ?? '';
      }
      return initial;
    },
  });

  /** 展开/收起「更多参数」。必填项与变量选择器始终可见。 */
  readonly showAdvanced = signal(false);

  readonly primaryFields = computed(() =>
    this.fields().filter((f) => f.kind === 'variables' || f.kind === 'variable'),
  );

  readonly advancedFields = computed(() =>
    this.fields().filter((f) => f.kind !== 'variables' && f.kind !== 'variable'),
  );

  readonly valid = computed(() =>
    this.fields().every((f) => {
      if (!f.required) return true;
      const v = this.state()[f.name];
      return Array.isArray(v) ? v.length > 0 : v !== null && v !== '' && v !== undefined;
    }),
  );

  constructor() {
    effect(() => {
      this.changed.emit({ value: this.state(), valid: this.valid() });
    });
  }

  optionsFor(field: FormField): VariableInfo[] {
    return this.variables().filter((v) => passesFilter(v, field.variableFilter));
  }

  selectedList(name: string): string[] {
    return (this.state()[name] as string[] | undefined) ?? [];
  }

  isChecked(name: string, id: string): boolean {
    return this.selectedList(name).includes(id);
  }

  toggleVariable(name: string, id: string): void {
    this.state.update((s) => {
      const current = (s[name] as string[] | undefined) ?? [];
      const next = current.includes(id)
        ? current.filter((x) => x !== id)
        : [...current, id];
      return { ...s, [name]: next };
    });
  }

  selectAll(field: FormField): void {
    this.state.update((s) => ({ ...s, [field.name]: this.optionsFor(field).map((v) => v.id) }));
  }

  clearAll(name: string): void {
    this.state.update((s) => ({ ...s, [name]: [] }));
  }

  setValue(name: string, value: unknown): void {
    this.state.update((s) => ({ ...s, [name]: value }));
  }

  setSingleVariable(name: string, raw: string): void {
    this.setValue(name, raw === '' ? null : raw);
  }

  setNumber(name: string, raw: string): void {
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) this.setValue(name, parsed);
  }

  kindLabel(v: VariableInfo): string {
    return v.kind === 'continuous' ? '连续' : v.kind === 'binary' ? '二分类' : '分类';
  }

  variableAriaLabel(v: VariableInfo): string {
    const unit = v.unit ? `，单位 ${v.unit}` : '';
    const missing = v.n_missing > 0 ? `，缺失 ${v.n_missing} 例` : '';
    const positive = v.n_positive !== null ? `，阳性 ${v.n_positive} 例` : '';
    return `${v.label}，${this.kindLabel(v)}变量${unit}${positive}${missing}`;
  }
}
