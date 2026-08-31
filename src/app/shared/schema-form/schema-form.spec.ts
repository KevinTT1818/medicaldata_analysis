/**
 * JSON Schema 驱动的参数表单。
 *
 * 后端加一个新算子时前端不改代码，全靠这里把 schema 翻译成控件 —— 也就是说
 * 这段翻译错了，受影响的是所有算子。重点守三件事：
 * 控件类型的推断、x-variable-filter 的筛选语义、必填单选变量的默认值
 * （留 null 会让下拉框显示第一项而模型是空，运行按钮灰着却看不出原因）。
 */
import { TestBed } from '@angular/core/testing';

import { SchemaForm } from './schema-form';
import { VARIABLES } from '../../testing/fixtures';

const SCHEMA = {
  properties: {
    variables: {
      type: 'array', title: '纳入变量', description: '出现在表格行上的变量',
      items: { type: 'string' }, minItems: 1, maxItems: 500,
      'x-widget': 'variables', 'x-variable-filter': 'any',
    },
    group_by: {
      anyOf: [{ type: 'string' }, { type: 'null' }],
      title: '分组变量', description: '留空则只出总体列',
      'x-widget': 'variable', 'x-variable-filter': 'grouping',
    },
    outcome: {
      type: 'string', title: '结局变量',
      'x-widget': 'variable', 'x-variable-filter': 'survival',
    },
    weighting: {
      type: 'string', enum: ['auto', 'weighted', 'unweighted'], default: 'auto',
      title: '抽样权重',
      'x-enum-labels': { auto: '自动（有权重就加权）', weighted: '强制加权', unweighted: '不加权' },
    },
    conf_level: {
      type: 'number', default: 0.95, minimum: 0.5, maximum: 0.999, title: '置信水平',
    },
    m: { type: 'integer', default: 5, minimum: 2, maximum: 50, title: '插补份数' },
    exact: { type: 'boolean', default: false, title: '精确检验' },
  },
  required: ['variables', 'outcome'],
};

function make(schema: Record<string, unknown> = SCHEMA, variables = VARIABLES) {
  const fixture = TestBed.createComponent(SchemaForm);
  fixture.componentRef.setInput('schema', schema);
  fixture.componentRef.setInput('variables', variables);
  fixture.detectChanges();
  return { fixture, c: fixture.componentInstance };
}

describe('SchemaForm 控件推断', () => {
  it('按 x-widget 与 type 推断控件类型', () => {
    const kinds = Object.fromEntries(make().c.fields().map((f) => [f.name, f.kind]));
    expect(kinds).toEqual({
      variables: 'variables',
      group_by: 'variable',
      outcome: 'variable',
      weighting: 'enum',
      conf_level: 'number',
      m: 'number',
      exact: 'boolean',
    });
  });

  it('识别 anyOf 里的 null 分支为可空', () => {
    const byName = new Map(make().c.fields().map((f) => [f.name, f]));
    expect(byName.get('group_by')!.nullable).toBe(true);
    expect(byName.get('outcome')!.nullable).toBe(false);
  });

  it('必填项来自 schema 的 required', () => {
    const required = make().c.fields().filter((f) => f.required).map((f) => f.name);
    expect(required).toEqual(['variables', 'outcome']);
  });

  it('枚举取 x-enum-labels 里的中文标签，缺标签时退回原值', () => {
    const field = make().c.fields().find((f) => f.name === 'weighting')!;
    expect(field.enumOptions).toEqual([
      { value: 'auto', label: '自动（有权重就加权）' },
      { value: 'weighted', label: '强制加权' },
      { value: 'unweighted', label: '不加权' },
    ]);
  });

  it('整数字段的步进是 1，小数字段用细步进', () => {
    const byName = new Map(make().c.fields().map((f) => [f.name, f]));
    expect(byName.get('m')!.step).toBe(1);
    expect(byName.get('conf_level')!.step).toBeLessThan(1);
  });

  it('带上取值范围，让浏览器也能拦住越界输入', () => {
    const field = make().c.fields().find((f) => f.name === 'conf_level')!;
    expect(field.min).toBe(0.5);
    expect(field.max).toBe(0.999);
  });

  it('变量选择器排在前面，其余归到「更多参数」', () => {
    const { c } = make();
    expect(c.primaryFields().map((f) => f.name)).toEqual(['variables', 'group_by', 'outcome']);
    expect(c.advancedFields().map((f) => f.name))
      .toEqual(['weighting', 'conf_level', 'm', 'exact']);
  });
});

describe('SchemaForm 变量筛选', () => {
  it('grouping 排除连续变量 —— 拿连续变量分组会分出几百组', () => {
    const { c } = make();
    const field = c.fields().find((f) => f.name === 'group_by')!;
    const ids = c.optionsFor(field).map((v) => v.id);
    expect(ids).not.toContain('person.age_at_index');
    expect(ids).not.toContain('measurement.chol');
    expect(ids).toContain('person.gender');
    expect(ids).toContain('person.race');
  });

  it('survival 只留带随访时长的结局变量', () => {
    const { c } = make();
    const field = c.fields().find((f) => f.name === 'outcome')!;
    expect(c.optionsFor(field).map((v) => v.id)).toEqual(['outcome.death']);
  });

  it('没有筛选器时全部可选', () => {
    const { c } = make();
    const field = c.fields().find((f) => f.name === 'variables')!;
    expect(c.optionsFor(field).length).toBe(VARIABLES.length);
  });

  it('covariate 排除结局变量 —— 拿结局当自变量是循环论证', () => {
    const schema = {
      properties: {
        covariates: { type: 'array', 'x-widget': 'variables', 'x-variable-filter': 'covariate' },
      },
      required: ['covariates'],
    };
    const { c } = make(schema);
    const ids = c.optionsFor(c.fields()[0]).map((v) => v.id);
    expect(ids.some((id) => id.startsWith('outcome.'))).toBe(false);
    expect(ids).toContain('condition.hf');
  });

  it('binary 只留二分类变量', () => {
    const schema = {
      properties: {
        y: { type: 'string', 'x-widget': 'variable', 'x-variable-filter': 'binary' },
      },
      required: ['y'],
    };
    const { c } = make(schema);
    expect(c.optionsFor(c.fields()[0]).every((v) => v.kind === 'binary')).toBe(true);
  });
});

describe('SchemaForm 默认值', () => {
  it('必填的单选变量默认选中第一个可选项，不留 null', () => {
    // 留 null 时浏览器会跳过 disabled 的占位项显示第一项，
    // 看起来选好了但模型是空的 —— 运行按钮灰着却看不出原因
    expect(make().c.state()['outcome']).toBe('outcome.death');
  });

  it('可选的单选变量默认留空', () => {
    expect(make().c.state()['group_by']).toBeNull();
  });

  it('多选变量默认空数组', () => {
    expect(make().c.state()['variables']).toEqual([]);
  });

  it('其余字段取 schema 里的 default', () => {
    const s = make().c.state();
    expect(s['weighting']).toBe('auto');
    expect(s['conf_level']).toBe(0.95);
    expect(s['m']).toBe(5);
    expect(s['exact']).toBe(false);
  });

  it('可选变量一个都没有时不硬塞值', () => {
    const schema = {
      properties: { y: { type: 'string', 'x-widget': 'variable', 'x-variable-filter': 'survival' } },
      required: ['y'],
    };
    const { c } = make(schema, VARIABLES.filter((v) => v.source !== 'outcome'));
    expect(c.state()['y']).toBeNull();
  });

  it('换算子时参数重置，不把上一个算子的选择带过去', () => {
    const { fixture, c } = make();
    c.toggleVariable('variables', 'person.age_at_index');
    expect(c.state()['variables']).toEqual(['person.age_at_index']);

    fixture.componentRef.setInput('schema', {
      properties: { variables: { type: 'array', 'x-widget': 'variables' } },
      required: ['variables'],
    });
    fixture.detectChanges();
    expect(c.state()['variables']).toEqual([]);
  });
});

describe('SchemaForm 选择与校验', () => {
  it('勾选与取消勾选', () => {
    const { c } = make();
    c.toggleVariable('variables', 'person.gender');
    c.toggleVariable('variables', 'person.race');
    expect(c.state()['variables']).toEqual(['person.gender', 'person.race']);
    expect(c.isChecked('variables', 'person.gender')).toBe(true);

    c.toggleVariable('variables', 'person.gender');
    expect(c.state()['variables']).toEqual(['person.race']);
    expect(c.isChecked('variables', 'person.gender')).toBe(false);
  });

  it('全选只选筛选器允许的变量', () => {
    const { c } = make();
    const field = c.fields().find((f) => f.name === 'group_by')!;
    c.selectAll(field);
    expect(c.state()['group_by']).toEqual(
      VARIABLES.filter((v) => v.kind !== 'continuous').map((v) => v.id),
    );
  });

  it('清空', () => {
    const { c } = make();
    c.toggleVariable('variables', 'person.gender');
    c.clearAll('variables');
    expect(c.state()['variables']).toEqual([]);
  });

  it('必填的多选为空时整个表单无效', () => {
    const { c } = make();
    expect(c.valid()).toBe(false);
    c.toggleVariable('variables', 'person.gender');
    expect(c.valid()).toBe(true);
  });

  it('必填的单选被清空后表单无效', () => {
    const { c } = make();
    c.toggleVariable('variables', 'person.gender');
    expect(c.valid()).toBe(true);
    c.setSingleVariable('outcome', '');
    expect(c.valid()).toBe(false);
  });

  it('可选字段留空不影响有效性', () => {
    const { c } = make();
    c.toggleVariable('variables', 'person.gender');
    c.setSingleVariable('group_by', '');
    expect(c.valid()).toBe(true);
  });

  it('数字输入解析失败时保留原值，不写入 NaN', () => {
    const { c } = make();
    c.setNumber('conf_level', '0.9');
    expect(c.state()['conf_level']).toBe(0.9);
    c.setNumber('conf_level', '不是数字');
    expect(c.state()['conf_level']).toBe(0.9);
  });

  it('每次变化都上报当前参数与是否有效', () => {
    const { fixture, c } = make();
    const seen: { value: Record<string, unknown>; valid: boolean }[] = [];
    c.changed.subscribe((e) => seen.push(e));

    c.toggleVariable('variables', 'person.gender');
    fixture.detectChanges();

    const last = seen.at(-1)!;
    expect(last.valid).toBe(true);
    expect(last.value['variables']).toEqual(['person.gender']);
  });
});

describe('SchemaForm 无障碍', () => {
  it('复选框的可读名称把标签与徽章分开，不会连成「年龄连续」', () => {
    const { c } = make();
    const age = VARIABLES.find((v) => v.id === 'person.age_at_index')!;
    expect(c.variableAriaLabel(age)).toBe('年龄，连续变量，单位 岁');
  });

  it('带缺失与阳性数时一并读出', () => {
    const { c } = make();
    const hf = VARIABLES.find((v) => v.id === 'condition.hf')!;
    expect(c.variableAriaLabel(hf)).toBe('心衰，二分类变量，阳性 32 例');

    const chol = VARIABLES.find((v) => v.id === 'measurement.chol')!;
    expect(c.variableAriaLabel(chol)).toContain('缺失 7 例');
  });
});
