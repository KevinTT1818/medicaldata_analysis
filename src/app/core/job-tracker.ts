import { Injectable } from '@angular/core';

import type { Job } from './api.types';

/** 通过 SSE 跟踪一个异步任务直到终结。 */
@Injectable({ providedIn: 'root' })
export class JobTracker {
  watch(jobId: string, onUpdate: (job: Job) => void): Promise<Job> {
    return new Promise<Job>((resolve, reject) => {
      const source = new EventSource(`/api/jobs/${jobId}/events`);
      let settled = false;

      const finish = (fn: () => void) => {
        settled = true;
        source.close();
        fn();
      };

      source.onmessage = (event: MessageEvent<string>) => {
        const job = JSON.parse(event.data) as Job;
        onUpdate(job);
        if (job.status === 'succeeded' || job.status === 'failed') {
          finish(() => resolve(job));
        }
      };

      // 服务端正常收流也会触发 onerror，已 settled 的忽略即可
      source.onerror = () => {
        if (!settled) {
          finish(() => reject(new Error('进度流中断，请刷新后重试')));
        }
      };
    });
  }
}
