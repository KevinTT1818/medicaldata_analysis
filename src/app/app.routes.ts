import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'datasets' },
  {
    path: 'datasets',
    title: '数据集',
    loadComponent: () =>
      import('./features/datasets/datasets-page').then((m) => m.DatasetsPage),
  },
  {
    path: 'cohorts',
    title: '队列构建器',
    loadComponent: () =>
      import('./features/cohorts/cohorts-page').then((m) => m.CohortsPage),
  },
  {
    path: 'workbench',
    title: '分析工作台',
    loadComponent: () =>
      import('./features/workbench/workbench-page').then((m) => m.WorkbenchPage),
  },
  {
    path: 'reports',
    title: '报告',
    loadComponent: () =>
      import('./features/reports/reports-page').then((m) => m.ReportsPage),
  },
  { path: '**', redirectTo: 'datasets' },
];
