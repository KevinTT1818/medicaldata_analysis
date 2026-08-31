import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      // 顶栏的密钥徽章会探测 /api/health 判断服务器要不要密钥
      providers: [provideRouter([]), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  /** 顶栏的密钥徽章会探测 /api/health。不答复它，whenStable() 会一直等下去。 */
  async function render() {
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();
    const ctrl = TestBed.inject(HttpTestingController);
    for (const req of ctrl.match('/api/health')) {
      req.flush({ status: 'ok', auth_required: false });
    }
    await fixture.whenStable();
    return fixture;
  }

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('should render the two top-level nav links', async () => {
    const fixture = await render();
    const links = (fixture.nativeElement as HTMLElement).querySelectorAll('nav a');
    expect([...links].map((a) => a.textContent?.trim())).toEqual([
      '数据集',
      '队列构建器',
      '分析工作台',
      '报告',
    ]);
  });

  it('should expose a skip link for keyboard users', async () => {
    const fixture = await render();
    const skip = (fixture.nativeElement as HTMLElement).querySelector('.skip-link');
    expect(skip?.getAttribute('href')).toBe('#main');
  });
});
