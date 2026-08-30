import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';

import { ApiKeyStore } from '../../core/api-key';
import { HealthProbe } from '../../core/health';

/**
 * 顶栏的密钥入口。只在服务器确实要求密钥时出现 —— 本机单人跑的时候
 * 多一个填不填都行的输入框只会让人犹豫。
 */
@Component({
  selector: 'app-api-key-badge',
  templateUrl: './api-key-badge.html',
  styleUrl: './api-key-badge.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ApiKeyBadge {
  private readonly store = inject(ApiKeyStore);
  private readonly health = inject(HealthProbe);

  readonly required = computed(() => this.health.authRequired());
  readonly hasKey = computed(() => this.store.present());
  readonly editing = signal(false);
  readonly draft = signal('');

  /** 要密钥却还没填 —— 这时所有请求都会 401，得让人一眼看到。 */
  readonly missing = computed(() => this.required() && !this.hasKey());

  open(): void {
    this.draft.set(this.store.key());
    this.editing.set(true);
  }

  cancel(): void {
    this.editing.set(false);
  }

  save(): void {
    this.store.set(this.draft());
    this.editing.set(false);
    // 密钥变了之后此前 401 的 httpResource 不会自己重试，刷新最省事
    location.reload();
  }

  clear(): void {
    this.store.clear();
    this.editing.set(false);
    location.reload();
  }
}
