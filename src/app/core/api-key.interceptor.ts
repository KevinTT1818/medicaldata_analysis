import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';

import { API_KEY_HEADER, ApiKeyStore } from './api-key';

/**
 * 给所有 /api 请求带上密钥。
 *
 * 放在拦截器而不是 ApiClient 里：页面还用 httpResource() 直接发请求，
 * 只改 ApiClient 会漏掉那些。健康检查故意不带 —— 它本来就不设防，
 * 前端要靠它在没有密钥时问出「这台服务器要不要密钥」。
 */
export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  if (!req.url.startsWith('/api') || req.url.startsWith('/api/health')) {
    return next(req);
  }
  const key = inject(ApiKeyStore).key();
  return next(key ? req.clone({ setHeaders: { [API_KEY_HEADER]: key } }) : req);
};
