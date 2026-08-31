import { Injectable, computed } from '@angular/core';
import { httpResource } from '@angular/common/http';

export interface Health {
  status: string;
  auth_required: boolean;
}

/**
 * 服务器状态。
 *
 * /api/health 是唯一不设防的端点 —— 前端要能在没有密钥时问出
 * 「这台服务器要不要密钥」，否则会陷入「要密钥才能知道要不要密钥」。
 */
@Injectable({ providedIn: 'root' })
export class HealthProbe {
  private readonly resource = httpResource<Health>(() => '/api/health');

  readonly authRequired = computed(() => this.resource.value()?.auth_required ?? false);
  readonly reachable = computed(() => !this.resource.error());
}
