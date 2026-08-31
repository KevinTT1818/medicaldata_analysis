/**
 * 任务进度流。
 *
 * 用 fetch 而不是 EventSource —— EventSource 发不了自定义请求头，带密钥就
 * 只剩放进查询串这条路，而密钥不该出现在 URL 里。代价是要自己拆 SSE 帧，
 * 所以这里重点测分帧：帧可能被网络切在任意位置，切在帧中间时不能丢事件、
 * 也不能拿半个 JSON 去 parse。
 */
import { TestBed } from '@angular/core/testing';

import { API_KEY_HEADER, ApiKeyStore } from './api-key';
import { JobTracker } from './job-tracker';
import type { Job } from './api.types';

function job(patch: Partial<Job>): Job {
  return {
    id: 'j1', kind: 'analysis', status: 'running', stage: '运行中', progress: 0.5,
    error: null, warnings: [], created_at: '', finished_at: null,
    cohort_n: null, cohort_total: null, cached: false, spec: {},
    ...patch,
  } as Job;
}

/** 把若干块文本做成一个流式响应，块的切分位置由调用方决定。 */
function streamOf(chunks: string[], status = 200): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(body, { status });
}

function frame(j: Job): string {
  return `data: ${JSON.stringify(j)}\n\n`;
}

describe('JobTracker', () => {
  let tracker: JobTracker;
  let calls: { url: string; init?: RequestInit }[];

  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
    tracker = TestBed.inject(JobTracker);
    calls = [];
  });

  function stubFetch(response: Response | (() => Response)) {
    globalThis.fetch = ((url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return Promise.resolve(typeof response === 'function' ? response() : response);
    }) as typeof fetch;
  }

  it('读到终态时把最后一个任务解析出来', async () => {
    stubFetch(streamOf([
      frame(job({ progress: 0.2 })),
      frame(job({ status: 'succeeded', progress: 1 })),
    ]));
    const seen: Job[] = [];
    const final = await tracker.watch('j1', (j) => seen.push(j));
    expect(final.status).toBe('succeeded');
    expect(seen.length).toBe(2);
  });

  it('失败也是终态，正常返回而不是抛', async () => {
    stubFetch(streamOf([frame(job({ status: 'failed', error: 'boom' }))]));
    const final = await tracker.watch('j1', () => {});
    expect(final.status).toBe('failed');
    expect(final.error).toBe('boom');
  });

  it('帧被切在中间也不丢事件', async () => {
    const whole = frame(job({ progress: 0.3 })) + frame(job({ status: 'succeeded' }));
    // 切在第一帧的 JSON 中间
    const cut = Math.floor(whole.length * 0.3);
    stubFetch(streamOf([whole.slice(0, cut), whole.slice(cut)]));
    const seen: Job[] = [];
    const final = await tracker.watch('j1', (j) => seen.push(j));
    expect(seen.length).toBe(2);
    expect(final.status).toBe('succeeded');
  });

  it('一次收到多帧时逐个上报', async () => {
    stubFetch(streamOf([
      frame(job({ progress: 0.1 })) + frame(job({ progress: 0.6 }))
      + frame(job({ status: 'succeeded' })),
    ]));
    const seen: Job[] = [];
    await tracker.watch('j1', (j) => seen.push(j));
    expect(seen.map((j) => j.progress)).toEqual([0.1, 0.6, 0.5]);
  });

  it('忽略非 data 行（心跳注释、事件名）', async () => {
    stubFetch(streamOf([
      ': 心跳\n\n',
      'event: ping\n\n',
      frame(job({ status: 'succeeded' })),
    ]));
    const final = await tracker.watch('j1', () => {});
    expect(final.status).toBe('succeeded');
  });

  it('带上 API 密钥', async () => {
    TestBed.inject(ApiKeyStore).set('abc123');
    stubFetch(streamOf([frame(job({ status: 'succeeded' }))]));
    await tracker.watch('j1', () => {});
    expect((calls[0].init!.headers as Record<string, string>)[API_KEY_HEADER]).toBe('abc123');
  });

  it('密钥不进 URL —— 会落进浏览器历史与代理日志', async () => {
    TestBed.inject(ApiKeyStore).set('abc123');
    stubFetch(streamOf([frame(job({ status: 'succeeded' }))]));
    await tracker.watch('j1', () => {});
    expect(calls[0].url).toBe('/api/jobs/j1/events');
    expect(calls[0].url).not.toContain('abc123');
  });

  it('401 时给出可操作的提示', async () => {
    stubFetch(streamOf([], 401));
    await expect(tracker.watch('j1', () => {})).rejects.toThrow(/密钥/);
  });

  it('流没到终态就结束时报中断', async () => {
    stubFetch(streamOf([frame(job({ progress: 0.4 }))]));
    await expect(tracker.watch('j1', () => {})).rejects.toThrow(/中断/);
  });

  it('服务端报错时报中断', async () => {
    stubFetch(streamOf([], 500));
    await expect(tracker.watch('j1', () => {})).rejects.toThrow(/中断/);
  });
});
