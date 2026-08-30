import { Injectable, computed, signal } from '@angular/core';

const STORAGE_KEY = 'meddata.api-key';

/** 密钥请求头名。与后端 app/security.py 里的 HEADER_NAME 一致。 */
export const API_KEY_HEADER = 'X-API-Key';

/**
 * API 密钥。
 *
 * 后端设了 MEDDATA_API_KEY 时所有 /api/* 都要带这个头。没设时后端放行，
 * 前端也就不需要它 —— 本机单人跑的开箱流程不受影响。
 *
 * 存 localStorage：这是一台机器上的一个人的凭据，不该每次刷新都重输。
 */
@Injectable({ providedIn: 'root' })
export class ApiKeyStore {
  private readonly state = signal<string>(this.restore());

  readonly key = computed(() => this.state());
  readonly present = computed(() => this.state().length > 0);

  private restore(): string {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? '';
    } catch {
      // 隐私模式或清过站点数据时读会抛，退回空值而不是让页面崩
      return '';
    }
  }

  set(key: string): void {
    const trimmed = key.trim();
    this.state.set(trimmed);
    try {
      if (trimmed) localStorage.setItem(STORAGE_KEY, trimmed);
      else localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* 存不进去也不影响本次会话使用 */
    }
  }

  clear(): void {
    this.set('');
  }
}
