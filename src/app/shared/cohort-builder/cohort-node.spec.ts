/**
 * 条件树节点。
 *
 * 节点不可变：任何编辑都产出一个新节点往上抛，由父节点替换对应位置。
 * 这里守的是编辑语义 —— 尤其是「换变量时必须重置运算符和取值」：
 * 连续变量的 between 留到分类变量上会拼出一条跑不通的查询，
 * 而错误要到点「运行」时才暴露。
 */
import { TestBed } from '@angular/core/testing';

import { CohortNode } from './cohort-node';
import { OPERATORS, VARIABLES } from '../../testing/fixtures';
import type { FilterCondition, FilterGroup, FilterNode } from '../../core/api.types';

function make(node: FilterNode) {
  const fixture = TestBed.createComponent(CohortNode);
  fixture.componentRef.setInput('node', node);
  fixture.componentRef.setInput('variables', VARIABLES);
  fixture.componentRef.setInput('operators', OPERATORS);
  fixture.detectChanges();

  const emitted: FilterNode[] = [];
  fixture.componentInstance.changed.subscribe((n) => emitted.push(n));
  return { c: fixture.componentInstance, emitted };
}

const numeric: FilterCondition = {
  kind: 'condition', variable: 'person.age_at_index', op: 'between', value: [40, 70],
};
const categorical: FilterCondition = {
  kind: 'condition', variable: 'person.race', op: 'in', value: ['A', 'B'],
};
const group: FilterGroup = { kind: 'group', op: 'and', children: [numeric] };

describe('CohortNode 取值形态', () => {
  it('按运算符决定用哪种取值控件', () => {
    expect(make(numeric).c.valueMode()).toBe('range');
    expect(make(categorical).c.valueMode()).toBe('list');
    expect(make({ ...numeric, op: 'gt', value: 50 }).c.valueMode()).toBe('number');
    expect(make({ ...categorical, op: 'eq', value: 'A' }).c.valueMode()).toBe('level');
    expect(make({ ...numeric, op: 'is_null', value: null }).c.valueMode()).toBe('none');
  });

  it('区间取值拆成两个输入框，缺一半时留空而不是显示 undefined', () => {
    expect(make(numeric).c.rangeValue()).toEqual(['40', '70']);
    expect(make({ ...numeric, value: [40, null] as never }).c.rangeValue()).toEqual(['40', '']);
    expect(make({ ...numeric, value: null }).c.rangeValue()).toEqual(['', '']);
  });

  it('标量取值遇到数组或空值时显示空串', () => {
    expect(make({ ...numeric, op: 'gt', value: 50 }).c.scalarValue()).toBe('50');
    expect(make({ ...numeric, op: 'gt', value: null }).c.scalarValue()).toBe('');
    expect(make(numeric).c.scalarValue()).toBe('');
  });

  it('诊断与结局变量的可选水平固定是「否 / 是」', () => {
    const hf = make({ kind: 'condition', variable: 'condition.hf', op: 'eq', value: null });
    expect(hf.c.levels()).toEqual(['否', '是']);
    const death = make({ kind: 'condition', variable: 'outcome.death', op: 'eq', value: null });
    expect(death.c.levels()).toEqual(['否', '是']);
  });

  it('其余变量的水平来自变量目录', () => {
    expect(make(categorical).c.levels()).toEqual(['A', 'B', 'C']);
  });

  it('连续变量没有可选水平', () => {
    expect(make({ ...numeric, op: 'gt' }).c.levels()).toEqual([]);
  });

  it('可用运算符随变量类型变化', () => {
    expect(make(numeric).c.availableOps()).toEqual(OPERATORS.allowed.continuous);
    expect(make(categorical).c.availableOps()).toEqual(OPERATORS.allowed.categorical);
  });

  it('二分类变量用分类的运算符集 —— 水平数随数据集变化，不能写死', () => {
    const node = make({ kind: 'condition', variable: 'person.gender', op: 'eq', value: null });
    expect(node.c.availableOps()).toEqual(OPERATORS.allowed.binary);
  });

  it('变量不在目录里时不崩，只是没有可用运算符', () => {
    const node = make({ kind: 'condition', variable: 'nope.gone', op: 'eq', value: null });
    expect(node.c.variable()).toBeNull();
    expect(node.c.availableOps()).toEqual([]);
  });
});

describe('CohortNode 编辑', () => {
  it('换变量时重置运算符与取值', () => {
    const { c, emitted } = make(numeric);
    c.onVariableChange('person.race');
    expect(emitted.at(-1)).toEqual({
      kind: 'condition',
      variable: 'person.race',
      op: OPERATORS.allowed.categorical[0],
      value: null,
    });
  });

  it('换到不存在的变量时什么都不做', () => {
    const { c, emitted } = make(numeric);
    c.onVariableChange('nope.gone');
    expect(emitted).toEqual([]);
  });

  it('换运算符时清掉取值 —— between 的 [40,70] 换成 gt 后没有意义', () => {
    const { c, emitted } = make(numeric);
    c.onOpChange('gt');
    expect(emitted.at(-1)).toEqual({ ...numeric, op: 'gt', value: null });
  });

  it('连续变量的标量输入解析失败时不发出变化', () => {
    const { c, emitted } = make({ ...numeric, op: 'gt', value: 50 });
    c.onScalarChange('五十');
    expect(emitted).toEqual([]);
    c.onScalarChange('60');
    expect((emitted.at(-1) as FilterCondition).value).toBe(60);
  });

  it('分类变量的标量输入按字符串原样传', () => {
    const { c, emitted } = make({ ...categorical, op: 'eq', value: 'A' });
    c.onScalarChange('B');
    expect((emitted.at(-1) as FilterCondition).value).toBe('B');
  });

  it('区间只改一端，另一端保留', () => {
    const { c, emitted } = make(numeric);
    c.onRangeChange(1, '80');
    expect((emitted.at(-1) as FilterCondition).value).toEqual([40, 80]);
  });

  it('区间清空一端得到 null，不是 0', () => {
    const { c, emitted } = make(numeric);
    c.onRangeChange(0, '');
    expect((emitted.at(-1) as FilterCondition).value).toEqual([null, 70]);
  });

  it('区间输入非数字时不发出变化', () => {
    const { c, emitted } = make(numeric);
    c.onRangeChange(0, 'abc');
    expect(emitted).toEqual([]);
  });

  it('勾选与取消勾选水平', () => {
    const { c, emitted } = make(categorical);
    c.toggleLevel('C');
    expect((emitted.at(-1) as FilterCondition).value).toEqual(['A', 'B', 'C']);
    expect(c.isLevelSelected('A')).toBe(true);
  });
});

describe('CohortNode 分组', () => {
  it('识别分组节点', () => {
    expect(make(group).c.isGroup()).toBe(true);
    expect(make(numeric).c.isGroup()).toBe(false);
  });

  it('切换与 / 或时保留子节点', () => {
    const { c, emitted } = make(group);
    c.onGroupOpChange('or');
    expect(emitted.at(-1)).toEqual({ ...group, op: 'or' });
  });

  it('新增条件时用第一个变量与它的第一个可用运算符', () => {
    const { c, emitted } = make(group);
    c.addCondition();
    const next = emitted.at(-1) as FilterGroup;
    expect(next.children.length).toBe(2);
    expect(next.children[1]).toEqual({
      kind: 'condition',
      variable: VARIABLES[0].id,
      op: OPERATORS.allowed[VARIABLES[0].kind][0],
      value: null,
    });
  });

  it('没有可选变量时不新增条件', () => {
    const fixture = TestBed.createComponent(CohortNode);
    fixture.componentRef.setInput('node', group);
    fixture.componentRef.setInput('variables', []);
    fixture.componentRef.setInput('operators', OPERATORS);
    fixture.detectChanges();
    const emitted: FilterNode[] = [];
    fixture.componentInstance.changed.subscribe((n) => emitted.push(n));
    fixture.componentInstance.addCondition();
    expect(emitted).toEqual([]);
  });

  it('新增的子分组默认用「或」—— 嵌套分组通常是为了表达并列的备选条件', () => {
    const { c, emitted } = make(group);
    c.addGroup();
    const next = emitted.at(-1) as FilterGroup;
    expect(next.children.at(-1)).toEqual({ kind: 'group', op: 'or', children: [] });
  });

  it('替换子节点只动指定位置', () => {
    const two: FilterGroup = { kind: 'group', op: 'and', children: [numeric, categorical] };
    const { c, emitted } = make(two);
    const replacement: FilterCondition = {
      kind: 'condition', variable: 'person.gender', op: 'eq', value: 'F',
    };
    c.replaceChild(1, replacement);
    const next = emitted.at(-1) as FilterGroup;
    expect(next.children).toEqual([numeric, replacement]);
  });

  it('删除子节点', () => {
    const two: FilterGroup = { kind: 'group', op: 'and', children: [numeric, categorical] };
    const { c, emitted } = make(two);
    c.removeChild(0);
    expect((emitted.at(-1) as FilterGroup).children).toEqual([categorical]);
  });

  it('编辑产出新对象，不改动传入的节点', () => {
    const original: FilterGroup = { kind: 'group', op: 'and', children: [numeric] };
    const snapshot = JSON.parse(JSON.stringify(original));
    const { c } = make(original);
    c.addCondition();
    c.onGroupOpChange('or');
    expect(original).toEqual(snapshot);
  });
});

describe('CohortNode 运算符符号', () => {
  it('用中文符号显示运算符', () => {
    const { c } = make(numeric);
    expect(c.symbol('between')).toBe('介于');
    expect(c.symbol('gte')).toBe('≥');
  });

  it('没有符号映射时退回运算符本身', () => {
    const { c } = make(numeric);
    expect(c.symbol('nope' as never)).toBe('nope');
  });
});
