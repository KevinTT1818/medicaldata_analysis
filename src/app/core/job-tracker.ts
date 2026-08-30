import { Injectable, inject } from '@angular/core';

import { API_KEY_HEADER, ApiKeyStore } from './api-key';
import type { Job } from './api.types';

/**
 * 通过 SSE 跟踪一个异步任务直到终结。
 *
 * 用 fetch 而不是 EventSource：EventSource 发不了自定义请求头，带密钥就只剩
 * 放进查询串这条路，而密钥不该出现在 URL 里（会进浏览器历史、代理日志、
 * Referer）。fetch 能读流也能带头，代价只是要自己拆 SSE 帧。
 */
@Injectable({ providedIn: 'root' })
export class JobTracker {
  private readonly apiKey = inject(ApiKeyStore);

  async watch(jobId: string, onUpdate: (job: Job) => void): Promise<Job> {
    const key = this.apiKey.key();
    const response = await fetch(`/api/jobs/${jobId}/events`, {
      headers: key ? { [API_KEY_HEADER]: key } : {},
    });

    if (response.status === 401) {
      throw new Error('服务器要求 API 密钥，请在右上角填写后重试');
    }
    if (!response.ok || !response.body) {
      throw new Error('进度流中断，请刷新后重试');
    }

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = '';
    let last: Job | null = null;

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += value;

        // SSE 以空行分帧；最后一段可能不完整，留在缓冲里等下一批
        const frames = buffer.split('\n\n');
        buffer = frames.pop() ?? '';

        for (const frame of frames) {
          const payload = frame
            .split('\n')
            .filter((line) => line.startsWith('data:'))
            .map((line) => line.slice(5).trim())
            .join('\n');
          if (!payload) continue;

          const job = JSON.parse(payload) as Job;
          last = job;
          onUpdate(job);
          if (job.status === 'succeeded' || job.status === 'failed') return job;
        }
      }
    } finally {
      await reader.cancel().catch(() => undefined);
    }

    // 流收完了却没见到终态 —— 服务端异常断开
    if (last && (last.status === 'succeeded' || last.status === 'failed')) return last;
    throw new Error('进度流中断，请刷新后重试');
  }
}
