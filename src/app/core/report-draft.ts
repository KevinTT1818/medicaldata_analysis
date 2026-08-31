import { Injectable, computed, signal } from '@angular/core';

import type { ReportSection } from './api.types';

const STORAGE_KEY = 'meddata.report-draft';

interface Draft {
  title: string;
  description: string;
  sections: ReportSection[];
  /** 正在编辑的已保存报告 id；为空表示新报告。 */
  editingId: string | null;
}

function emptyDraft(): Draft {
  return { title: '', description: '', sections: [], editingId: null };
}

/**
 * 报告草稿。
 *
 * 在工作台跑完一次分析后「加入报告」，再到报告页整理成文 —— 跨页面，所以放
 * localStorage 里，刷新和跳转都不会丢。存的是分析的定义而非算出来的数字，
 * 保存到后端后才能重跑核对。
 */
@Injectable({ providedIn: 'root' })
export class ReportDraft {
  private readonly state = signal<Draft>(this.restore());

  readonly title = computed(() => this.state().title);
  readonly description = computed(() => this.state().description);
  readonly sections = computed(() => this.state().sections);
  readonly editingId = computed(() => this.state().editingId);
  readonly count = computed(() => this.state().sections.length);

  private restore(): Draft {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return emptyDraft();
      const parsed = JSON.parse(raw) as Partial<Draft>;
      return { ...emptyDraft(), ...parsed, sections: parsed.sections ?? [] };
    } catch {
      // 存储不可用（隐私模式、清过站点数据）时退回空草稿，不该让页面崩
      return emptyDraft();
    }
  }

  private persist(next: Draft): void {
    this.state.set(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* 存不进去也不影响本次会话使用 */
    }
  }

  add(section: ReportSection): void {
    const current = this.state();
    this.persist({ ...current, sections: [...current.sections, section] });
  }

  removeAt(index: number): void {
    const current = this.state();
    this.persist({
      ...current,
      sections: current.sections.filter((_, i) => i !== index),
    });
  }

  move(index: number, delta: number): void {
    const current = this.state();
    const target = index + delta;
    if (target < 0 || target >= current.sections.length) return;
    const sections = [...current.sections];
    [sections[index], sections[target]] = [sections[target], sections[index]];
    this.persist({ ...current, sections });
  }

  updateSection(index: number, patch: Partial<ReportSection>): void {
    const current = this.state();
    this.persist({
      ...current,
      sections: current.sections.map((s, i) => (i === index ? { ...s, ...patch } : s)),
    });
  }

  setTitle(title: string): void {
    this.persist({ ...this.state(), title });
  }

  setDescription(description: string): void {
    this.persist({ ...this.state(), description });
  }

  /** 载入一份已保存的报告继续编辑。 */
  load(id: string, title: string, description: string | null,
       sections: ReportSection[]): void {
    this.persist({ title, description: description ?? '', sections, editingId: id });
  }

  clear(): void {
    this.persist(emptyDraft());
  }
}
