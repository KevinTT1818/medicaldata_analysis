import {
  ChangeDetectionStrategy,
  Component,
  computed,
  forwardRef,
  input,
  output,
} from '@angular/core';

import type {
  FilterCondition,
  FilterGroup,
  FilterNode,
  FilterOp,
  OperatorMeta,
  VariableInfo,
} from '../../core/api.types';

/**
 * 条件树的一个节点。自身递归引用以渲染嵌套分组。
 *
 * 节点不可变：任何编辑都产出一个新节点并通过 changed 上抛，
 * 由父节点替换对应位置。这样整棵树只有一个真实来源，撤销/比较都简单。
 */
@Component({
  selector: 'app-cohort-node',
  imports: [forwardRef(() => CohortNode)],
  templateUrl: './cohort-node.html',
  styleUrl: './cohort-node.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CohortNode {
  readonly node = input.required<FilterNode>();
  readonly variables = input.required<VariableInfo[]>();
  readonly operators = input.required<OperatorMeta>();
  readonly depth = input(0);
  readonly removable = input(true);

  readonly changed = output<FilterNode>();
  readonly removed = output<void>();

  readonly isGroup = computed(() => this.node().kind === 'group');
  readonly asGroup = computed(() => this.node() as FilterGroup);
  readonly asCondition = computed(() => this.node() as FilterCondition);

  readonly variable = computed<VariableInfo | null>(() => {
    const condition = this.node();
    if (condition.kind !== 'condition') return null;
    return this.variables().find((v) => v.id === condition.variable) ?? null;
  });

  readonly availableOps = computed<FilterOp[]>(() => {
    const v = this.variable();
    return v ? (this.operators().allowed[v.kind] ?? []) : [];
  });

  readonly valueMode = computed<'none' | 'range' | 'list' | 'number' | 'level'>(() => {
    const condition = this.asCondition();
    const meta = this.operators();
    if (meta.no_value_ops.includes(condition.op)) return 'none';
    if (meta.range_ops.includes(condition.op)) return 'range';
    if (meta.list_ops.includes(condition.op)) return 'list';
    return this.variable()?.kind === 'continuous' ? 'number' : 'level';
  });

  readonly levels = computed<string[]>(() => {
    const v = this.variable();
    if (!v) return [];
    return v.source === 'condition' || v.source === 'outcome' ? ['否', '是'] : (v.levels ?? []);
  });

  readonly rangeValue = computed<[string, string]>(() => {
    const value = this.asCondition().value;
    return Array.isArray(value)
      ? [String(value[0] ?? ''), String(value[1] ?? '')]
      : ['', ''];
  });

  readonly listValue = computed<string[]>(() => {
    const value = this.asCondition().value;
    return Array.isArray(value) ? value.map(String) : [];
  });

  readonly scalarValue = computed(() => {
    const value = this.asCondition().value;
    return Array.isArray(value) || value === null || value === undefined ? '' : String(value);
  });

  symbol(op: FilterOp): string {
    return this.operators().symbols[op] ?? op;
  }

  /** 变量类型不同，可用运算符不同；换变量时把不兼容的运算符和取值一并重置。 */
  onVariableChange(variableId: string): void {
    const next = this.variables().find((v) => v.id === variableId);
    if (!next) return;
    const allowed = this.operators().allowed[next.kind] ?? [];
    this.changed.emit({
      kind: 'condition',
      variable: variableId,
      op: allowed[0] ?? 'eq',
      value: null,
    });
  }

  onOpChange(op: FilterOp): void {
    this.changed.emit({ ...this.asCondition(), op, value: null });
  }

  onScalarChange(raw: string): void {
    const condition = this.asCondition();
    const isNumeric = this.variable()?.kind === 'continuous';
    const parsed = isNumeric ? Number(raw) : raw;
    if (isNumeric && !Number.isFinite(parsed as number)) return;
    this.changed.emit({ ...condition, value: parsed });
  }

  onRangeChange(index: 0 | 1, raw: string): void {
    const current = this.rangeValue();
    const next: [string, string] = index === 0 ? [raw, current[1]] : [current[0], raw];
    const numbers = next.map((x) => (x === '' ? null : Number(x)));
    if (numbers.some((n) => n !== null && !Number.isFinite(n))) return;
    this.changed.emit({ ...this.asCondition(), value: numbers as (number | string)[] });
  }

  toggleLevel(level: string): void {
    const current = this.listValue();
    const next = current.includes(level)
      ? current.filter((x) => x !== level)
      : [...current, level];
    this.changed.emit({ ...this.asCondition(), value: next });
  }

  isLevelSelected(level: string): boolean {
    return this.listValue().includes(level);
  }

  // --- 分组操作 ---

  onGroupOpChange(op: 'and' | 'or'): void {
    this.changed.emit({ ...this.asGroup(), op });
  }

  addCondition(): void {
    const group = this.asGroup();
    const first = this.variables()[0];
    if (!first) return;
    const allowed = this.operators().allowed[first.kind] ?? [];
    this.changed.emit({
      ...group,
      children: [
        ...group.children,
        { kind: 'condition', variable: first.id, op: allowed[0] ?? 'eq', value: null },
      ],
    });
  }

  addGroup(): void {
    const group = this.asGroup();
    this.changed.emit({
      ...group,
      children: [...group.children, { kind: 'group', op: 'or', children: [] }],
    });
  }

  replaceChild(index: number, child: FilterNode): void {
    const group = this.asGroup();
    this.changed.emit({
      ...group,
      children: group.children.map((c, i) => (i === index ? child : c)),
    });
  }

  removeChild(index: number): void {
    const group = this.asGroup();
    this.changed.emit({ ...group, children: group.children.filter((_, i) => i !== index) });
  }
}
