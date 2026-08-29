import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([])],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('should render the two top-level nav links', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const links = (fixture.nativeElement as HTMLElement).querySelectorAll('nav a');
    expect([...links].map((a) => a.textContent?.trim())).toEqual([
      '数据集',
      '队列构建器',
      '分析工作台',
      '报告',
    ]);
  });

  it('should expose a skip link for keyboard users', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const skip = (fixture.nativeElement as HTMLElement).querySelector('.skip-link');
    expect(skip?.getAttribute('href')).toBe('#main');
  });
});
