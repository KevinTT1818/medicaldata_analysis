import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import type {
  AnalysisDescriptor,
  CohortPreview,
  FilterNode,
  Job,
  JobResultEnvelope,
  MeasurementAgg,
  ReportRun,
  ReportSection,
  SavedCohort,
  SavedReport,
} from './api.types';

@Injectable({ providedIn: 'root' })
export class ApiClient {
  private readonly http = inject(HttpClient);

  importDataset(datasetId: string): Observable<Job> {
    return this.http.post<Job>(`/api/datasets/${datasetId}/import`, {});
  }

  analysisRegistry(): Observable<AnalysisDescriptor[]> {
    return this.http.get<AnalysisDescriptor[]>('/api/analyses/registry');
  }

  submitAnalysis(
    dataset: string,
    analysis: string,
    params: unknown,
    cohortId?: string | null,
    measurementAgg?: MeasurementAgg,
  ): Observable<Job> {
    return this.http.post<Job>('/api/analyses', {
      dataset, analysis, params,
      ...(cohortId ? { cohort_id: cohortId } : {}),
      ...(measurementAgg ? { measurement_agg: measurementAgg } : {}),
    });
  }

  saveCohort(body: {
    dataset: string; name: string; description: string | null; definition: FilterNode;
  }): Observable<SavedCohort> {
    return this.http.post<SavedCohort>('/api/cohorts', body);
  }

  updateCohort(id: string, body: {
    dataset: string; name: string; description: string | null; definition: FilterNode;
  }): Observable<SavedCohort> {
    return this.http.put<SavedCohort>(`/api/cohorts/${id}`, body);
  }

  deleteCohort(id: string): Observable<void> {
    return this.http.delete<void>(`/api/cohorts/${id}`);
  }

  previewCohort(dataset: string, definition: FilterNode | null): Observable<CohortPreview> {
    return this.http.post<CohortPreview>('/api/cohorts/preview', { dataset, definition });
  }

  saveReport(body: {
    title: string; description: string | null; sections: ReportSection[];
  }): Observable<SavedReport> {
    return this.http.post<SavedReport>('/api/reports', { report: body, baseline: true });
  }

  updateReport(id: string, body: {
    title: string; description: string | null; sections: ReportSection[];
  }): Observable<SavedReport> {
    return this.http.put<SavedReport>(`/api/reports/${id}`, { report: body, baseline: true });
  }

  deleteReport(id: string): Observable<void> {
    return this.http.delete<void>(`/api/reports/${id}`);
  }

  /** 重跑整份报告并与基线比对。 */
  runReport(id: string): Observable<ReportRun> {
    return this.http.post<ReportRun>(`/api/reports/${id}/run`, {});
  }

  /** 以当前结果为准重设基线。 */
  rebaseline(id: string): Observable<SavedReport> {
    return this.http.post<SavedReport>(`/api/reports/${id}/rebaseline`, {});
  }

  /** 导出 Word。图表由前端把画布转成 PNG 一起送过去，服务端不重画。 */
  exportDocx(id: string, images: Record<string, string[]>): Observable<Blob> {
    return this.http.post(`/api/reports/${id}/export/docx`, { images },
                          { responseType: 'blob' });
  }

  job(jobId: string): Observable<Job> {
    return this.http.get<Job>(`/api/jobs/${jobId}`);
  }

  jobResult(jobId: string): Observable<JobResultEnvelope> {
    return this.http.get<JobResultEnvelope>(`/api/jobs/${jobId}/result`);
  }
}
