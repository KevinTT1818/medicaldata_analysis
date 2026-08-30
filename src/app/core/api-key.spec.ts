/**
 * API 密钥的存取与随请求下发。
 *
 * 后端设了 MEDDATA_API_KEY 时所有 /api/* 都要带 X-API-Key。这里守两件事：
 * 拦截器不能漏（页面还用 httpResource() 直接发请求，只改 ApiClient 会漏掉），
 * 以及 /api/health 必须不带 —— 它是唯一不设防的端点，前端靠它问出
 * 「这台服务器要不要密钥」，带上一个错的反而可能被拒。
 */
import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { API_KEY_HEADER, ApiKeyStore } from './api-key';
import { apiKeyInterceptor } from './api-key.interceptor';

const STORAGE_KEY = 'meddata.api-key';

describe('ApiKeyStore', () => {
  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
  });

  it('初始没有密钥', () => {
    const store = TestBed.inject(ApiKeyStore);
    expect(store.key()).toBe('');
    expect(store.present()).toBe(false);
  });

  it('存下的密钥能被新实例读到', () => {
    TestBed.inject(ApiKeyStore).set('abc123');
    TestBed.resetTestingModule();
    expect(TestBed.inject(ApiKeyStore).key()).toBe('abc123');
  });

  it('去掉首尾空白 —— 粘贴时很容易带上换行', () => {
    const store = TestBed.inject(ApiKeyStore);
    store.set('  abc123\n');
    expect(store.key()).toBe('abc123');
  });

  it('清除后存储里也不留', () => {
    const store = TestBed.inject(ApiKeyStore);
    store.set('abc123');
    store.clear();
    expect(store.present()).toBe(false);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('设成空串等于清除', () => {
    const store = TestBed.inject(ApiKeyStore);
    store.set('abc123');
    store.set('   ');
    expect(store.present()).toBe(false);
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('存储不可用时退回空值而不是崩', () => {
    const original = Storage.prototype.getItem;
    Storage.prototype.getItem = () => { throw new Error('SecurityError'); };
    try {
      TestBed.resetTestingModule();
      expect(TestBed.inject(ApiKeyStore).key()).toBe('');
    } finally {
      Storage.prototype.getItem = original;
    }
  });

  it('写存储抛异常时本次会话照常能用', () => {
    const store = TestBed.inject(ApiKeyStore);
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = () => { throw new Error('QuotaExceededError'); };
    try {
      store.set('abc123');
      expect(store.key()).toBe('abc123');
    } finally {
      Storage.prototype.setItem = original;
    }
  });
});

describe('apiKeyInterceptor', () => {
  let http: HttpClient;
  let ctrl: HttpTestingController;
  let store: ApiKeyStore;

  beforeEach(() => {
    localStorage.clear();
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([apiKeyInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpClient);
    ctrl = TestBed.inject(HttpTestingController);
    store = TestBed.inject(ApiKeyStore);
  });

  afterEach(() => ctrl.verify());

  it('有密钥时给 /api 请求带上头', () => {
    store.set('abc123');
    http.get('/api/cohorts').subscribe();
    const req = ctrl.expectOne('/api/cohorts');
    expect(req.request.headers.get(API_KEY_HEADER)).toBe('abc123');
    req.flush([]);
  });

  it('没有密钥时不带头 —— 后端未启用校验时也要能用', () => {
    http.get('/api/cohorts').subscribe();
    const req = ctrl.expectOne('/api/cohorts');
    expect(req.request.headers.has(API_KEY_HEADER)).toBe(false);
    req.flush([]);
  });

  it('健康检查不带密钥', () => {
    store.set('abc123');
    http.get('/api/health').subscribe();
    const req = ctrl.expectOne('/api/health');
    expect(req.request.headers.has(API_KEY_HEADER)).toBe(false);
    req.flush({ status: 'ok', auth_required: true });
  });

  it('非 /api 的请求不带密钥', () => {
    store.set('abc123');
    http.get('/assets/logo.svg').subscribe();
    const req = ctrl.expectOne('/assets/logo.svg');
    expect(req.request.headers.has(API_KEY_HEADER)).toBe(false);
    req.flush('');
  });

  it('各种方法都带上，不只是 GET', () => {
    store.set('abc123');
    http.post('/api/analyses', {}).subscribe();
    http.delete('/api/reports/x').subscribe();
    http.put('/api/cohorts/x', {}).subscribe();
    for (const url of ['/api/analyses', '/api/reports/x', '/api/cohorts/x']) {
      const req = ctrl.expectOne(url);
      expect(req.request.headers.get(API_KEY_HEADER)).toBe('abc123');
      req.flush({});
    }
  });

  it('改了密钥之后的请求用新值', () => {
    store.set('old');
    http.get('/api/jobs').subscribe();
    const first = ctrl.expectOne('/api/jobs');
    expect(first.request.headers.get(API_KEY_HEADER)).toBe('old');
    first.flush([]);

    store.set('new');
    http.get('/api/jobs').subscribe();
    const second = ctrl.expectOne('/api/jobs');
    expect(second.request.headers.get(API_KEY_HEADER)).toBe('new');
    second.flush([]);
  });
});
