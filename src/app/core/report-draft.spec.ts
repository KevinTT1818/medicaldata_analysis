/**
 * 报告草稿。跨页面所以存在 localStorage 里，刷新和跳转都不能丢。
 *
 * 存的是分析的定义而不是算出来的数字 —— 保存到后端后才能重跑核对。
 * 这里也守存储不可用时的退路：隐私模式或清过站点数据时读写都会抛，
 * 页面不该因此崩掉。
 */
import { TestBed } from '@angular/core/testing';

import { ReportDraft } from './report-draft';
import { section } from '../testing/fixtures';

const KEY = 'meddata.report-draft';

function fresh(): ReportDraft {
  TestBed.resetTestingModule();
  return TestBed.inject(ReportDraft);
}

describe('ReportDraft 基本编辑', () => {
  beforeEach(() => localStorage.clear());

  it('初始为空', () => {
    const d = fresh();
    expect(d.count()).toBe(0);
    expect(d.title()).toBe('');
    expect(d.editingId()).toBeNull();
  });

  it('按加入顺序追加小节', () => {
    const d = fresh();
    d.add(section({ title: '基线' }));
    d.add(section({ title: '生存' }));
    expect(d.sections().map((s) => s.title)).toEqual(['基线', '生存']);
    expect(d.count()).toBe(2);
  });

  it('按下标删除', () => {
    const d = fresh();
    d.add(section({ title: 'A' }));
    d.add(section({ title: 'B' }));
    d.add(section({ title: 'C' }));
    d.removeAt(1);
    expect(d.sections().map((s) => s.title)).toEqual(['A', 'C']);
  });

  it('上下移动相邻交换', () => {
    const d = fresh();
    d.add(section({ title: 'A' }));
    d.add(section({ title: 'B' }));
    d.move(0, 1);
    expect(d.sections().map((s) => s.title)).toEqual(['B', 'A']);
  });

  it('移出边界时什么都不做', () => {
    const d = fresh();
    d.add(section({ title: 'A' }));
    d.add(section({ title: 'B' }));
    d.move(0, -1);
    d.move(1, 1);
    expect(d.sections().map((s) => s.title)).toEqual(['A', 'B']);
  });

  it('局部更新小节，其余字段保留', () => {
    const d = fresh();
    d.add(section({ title: 'A', note: null }));
    d.updateSection(0, { note: '这里补一句说明' });
    expect(d.sections()[0].title).toBe('A');
    expect(d.sections()[0].note).toBe('这里补一句说明');
  });

  it('清空同时清掉正在编辑的报告 id', () => {
    const d = fresh();
    d.load('rpt-1', '旧报告', '说明', [section()]);
    expect(d.editingId()).toBe('rpt-1');
    d.clear();
    expect(d.count()).toBe(0);
    expect(d.editingId()).toBeNull();
    expect(d.title()).toBe('');
  });

  it('载入已保存的报告继续编辑', () => {
    const d = fresh();
    d.load('rpt-9', '心衰队列分析', null, [section({ title: 'X' })]);
    expect(d.editingId()).toBe('rpt-9');
    expect(d.title()).toBe('心衰队列分析');
    expect(d.description()).toBe('');   // null 归一成空串，模板不用再判空
    expect(d.sections().map((s) => s.title)).toEqual(['X']);
  });
});

describe('ReportDraft 持久化', () => {
  beforeEach(() => localStorage.clear());

  it('写入后能从存储里读回来', () => {
    const d = fresh();
    d.setTitle('我的报告');
    d.add(section({ title: '基线' }));

    const raw = JSON.parse(localStorage.getItem(KEY)!);
    expect(raw.title).toBe('我的报告');
    expect(raw.sections.length).toBe(1);
  });

  it('新实例能接着上次的草稿', () => {
    const first = fresh();
    first.setTitle('续写');
    first.setDescription('说明');
    first.add(section({ title: '基线' }));

    const second = fresh();
    expect(second.title()).toBe('续写');
    expect(second.description()).toBe('说明');
    expect(second.sections().map((s) => s.title)).toEqual(['基线']);
  });

  it('存的是分析定义而不是结果 —— 这样才能重跑核对', () => {
    const d = fresh();
    d.add(section({
      dataset: 'nhanes_2017',
      analysis: 'describe.baseline_table',
      params: { variables: ['person.age_at_index'], weighting: 'weighted' },
      measurement_agg: 'last',
    }));
    const stored = JSON.parse(localStorage.getItem(KEY)!).sections[0];
    expect(stored.dataset).toBe('nhanes_2017');
    expect(stored.analysis).toBe('describe.baseline_table');
    expect(stored.params.weighting).toBe('weighted');
    expect(stored.measurement_agg).toBe('last');
  });

  it('存储里是坏 JSON 时退回空草稿而不是崩', () => {
    localStorage.setItem(KEY, '{这不是 JSON');
    const d = fresh();
    expect(d.count()).toBe(0);
    expect(d.title()).toBe('');
  });

  it('存储里缺字段时补齐默认值', () => {
    localStorage.setItem(KEY, JSON.stringify({ title: '只有标题' }));
    const d = fresh();
    expect(d.title()).toBe('只有标题');
    expect(d.sections()).toEqual([]);
    expect(d.editingId()).toBeNull();
  });

  it('读存储抛异常时（隐私模式）退回空草稿', () => {
    const original = Storage.prototype.getItem;
    Storage.prototype.getItem = () => { throw new Error('SecurityError'); };
    try {
      expect(fresh().count()).toBe(0);
    } finally {
      Storage.prototype.getItem = original;
    }
  });

  it('写存储抛异常时本次会话照常能用', () => {
    const d = fresh();
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = () => { throw new Error('QuotaExceededError'); };
    try {
      d.add(section({ title: '写不进去也要能加' }));
      expect(d.count()).toBe(1);
    } finally {
      Storage.prototype.setItem = original;
    }
  });
});
