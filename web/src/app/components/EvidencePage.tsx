import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  FileText,
  Image as ImageIcon,
  Loader2,
  Table2,
} from 'lucide-react';
import type { AnalysisRequest, AppData, EvidenceStep, QuestionItem, UploadJobStatus } from '../types';

interface EvidencePageProps {
  data: AppData;
  question: QuestionItem;
  request: AnalysisRequest | null;
  onBack: () => void;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? '';

const roleLabels: Record<string, string> = {
  main_evidence: '主证据',
  graph_neighbor: '图邻居',
  explicit_reference: '显式引用',
  table_or_figure: '表图证据',
  visual_companion: '视觉补充',
  caption: '图注',
  context_text: '上下文',
};

function formatPercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  return `${(value * 100).toFixed(0)}%`;
}

function scoreWidth(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '0%';
  }
  return `${Math.max(4, Math.min(100, value * 100))}%`;
}

function qualityClass(status: string) {
  if (status === 'pass' || status === 'succeeded') {
    return 'border-emerald-200 bg-emerald-50 text-emerald-700';
  }
  if (status === 'warn' || status === 'running' || status === 'queued') {
    return 'border-amber-200 bg-amber-50 text-amber-700';
  }
  if (status === 'failed') {
    return 'border-rose-200 bg-rose-50 text-rose-700';
  }
  return 'border-slate-200 bg-slate-100 text-slate-600';
}

function enabledFlag(value?: number | boolean) {
  return value === true || value === 1;
}

function nodeIcon(step: EvidenceStep) {
  if (step.node_type === 'table') {
    return <Table2 size={15} />;
  }
  if (step.node_type === 'figure' || step.node_type === 'caption' || step.role === 'visual_companion') {
    return <ImageIcon size={15} />;
  }
  return <FileText size={15} />;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function EvidencePage({ data, question, request, onBack }: EvidencePageProps) {
  const uploadRequest = request?.mode === 'upload' ? request : null;
  const [jobStatus, setJobStatus] = useState<UploadJobStatus | null>(null);
  const [jobError, setJobError] = useState('');
  const [selectedNodeId, setSelectedNodeId] = useState<string>('');

  useEffect(() => {
    if (!uploadRequest) {
      return;
    }

    let cancelled = false;

    async function submitAndPoll() {
      try {
        setJobError('');
        setJobStatus(null);
        const formData = new FormData();
        formData.append('pdf', uploadRequest.file);
        formData.append('question', uploadRequest.question);
        formData.append('chunk_template', uploadRequest.chunk_template);

        const response = await fetch(`${API_BASE}/api/analyze`, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const started = (await response.json()) as { job_id: string };

        while (!cancelled) {
          const statusResponse = await fetch(`${API_BASE}/api/jobs/${started.job_id}`, { cache: 'no-store' });
          if (!statusResponse.ok) {
            throw new Error(await statusResponse.text());
          }
          const nextStatus = (await statusResponse.json()) as UploadJobStatus;
          setJobStatus(nextStatus);
          if (nextStatus.status === 'succeeded' || nextStatus.status === 'failed') {
            break;
          }
          await sleep(1400);
        }
      } catch (error) {
        if (!cancelled) {
          setJobError(error instanceof Error ? error.message : '上传 PDF 分析失败');
        }
      }
    }

    submitAndPoll();
    return () => {
      cancelled = true;
    };
  }, [uploadRequest]);

  const displayQuestion = jobStatus?.result?.question ?? question;
  const steps = uploadRequest ? jobStatus?.result?.steps ?? [] : data.chains[question.question_id] ?? [];
  const sortedSteps = useMemo(() => {
    return [...steps].sort((a, b) => b.score - a.score || a.chain_step - b.chain_step);
  }, [steps]);

  const selectedStep =
    sortedSteps.find((step) => `${step.chain_step}-${step.node_id}` === selectedNodeId) ?? sortedSteps[0] ?? null;
  const pdfName = request?.pdf_name || `${displayQuestion.doc_id}.pdf`;
  const statusLabel = uploadRequest ? jobStatus?.status ?? (jobError ? 'failed' : 'queued') : displayQuestion.quality_status;
  const isUploadRunning =
    Boolean(uploadRequest) && !jobError && jobStatus?.status !== 'succeeded' && jobStatus?.status !== 'failed';
  const pipelineVariant =
    displayQuestion.pipeline_variant || jobStatus?.pipeline_variant || (uploadRequest ? 'V5-online-quality' : 'offline');
  const candidateRetriever = displayQuestion.candidate_retriever || jobStatus?.candidate_retriever || 'multiroute';
  const rerankRetriever = displayQuestion.rerank_retriever || jobStatus?.rerank_retriever || candidateRetriever;
  const pipelineFeatures = [
    ['Context', displayQuestion.context_expansion ?? jobStatus?.context_expansion],
    ['Adaptive', displayQuestion.adaptive_rerank_boost ?? jobStatus?.adaptive_rerank_boost],
    ['GraphBoost', displayQuestion.graph_context_boost ?? jobStatus?.graph_context_boost],
    ['Guard', displayQuestion.evidence_guard ?? jobStatus?.evidence_guard],
    ['EnhancedEdges', displayQuestion.enhanced_context_edges ?? jobStatus?.enhanced_context_edges],
  ]
    .filter(([, value]) => enabledFlag(value as number | boolean | undefined))
    .map(([label]) => label as string);

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-sky-50 to-cyan-50">
      <header className="border-b border-blue-100 bg-white/90 px-4 py-4 shadow-sm backdrop-blur-sm sm:px-6">
        <div className="mx-auto max-w-[1800px]">
          <button
            onClick={onBack}
            className="mb-3 inline-flex items-center gap-2 text-blue-600 transition hover:text-blue-700"
          >
            <ArrowLeft size={20} />
            返回
          </button>

          <div className="flex items-start gap-4">
            <FileText className="mt-1 shrink-0 text-blue-500" size={24} />
            <div className="min-w-0 flex-1">
              <h1 className="truncate text-xl font-semibold text-blue-950">{pdfName}</h1>
              <p className="mt-1 leading-7 text-slate-600">问题：{displayQuestion.question}</p>
            </div>
            <div className="hidden shrink-0 rounded-full bg-blue-100 px-4 py-2 text-blue-700 sm:block">
              {sortedSteps.length} 条证据
            </div>
          </div>
        </div>
      </header>

      <section className="mx-auto flex max-w-[1800px] flex-col gap-6 p-4 sm:p-6">
        <section className="rounded-xl border border-blue-100 bg-white p-4 shadow-lg">
          <div className="mb-4 flex items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-blue-950">证据卡片</h2>
            <span className={`rounded-full border px-3 py-1 text-sm ${qualityClass(statusLabel || '')}`}>
              {statusLabel || 'pending'}
            </span>
          </div>

          {displayQuestion.card_url ? (
            <img
              src={displayQuestion.card_url}
              alt={`${displayQuestion.question_id} evidence card`}
              className="w-full rounded-lg border border-blue-100 bg-white object-contain shadow-sm"
            />
          ) : (
            <JobPlaceholder running={isUploadRunning} status={jobStatus} error={jobError} />
          )}

          <div className="mx-auto mt-4 grid max-w-3xl grid-cols-3 gap-2 text-center text-xs text-slate-500">
            <SummaryCell label="证据步骤" value={displayQuestion.num_steps || sortedSteps.length} />
            <SummaryCell label="视觉节点" value={displayQuestion.visual_node_steps} />
            <SummaryCell label="裁剪图" value={displayQuestion.crop_steps} />
          </div>

          <PipelineStatus
            variant={pipelineVariant}
            candidateRetriever={candidateRetriever}
            rerankRetriever={rerankRetriever}
            features={pipelineFeatures}
          />

          {displayQuestion.quality_issues.length > 0 && (
            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm leading-6 text-amber-800">
              {displayQuestion.quality_issues.join('；')}
            </div>
          )}
        </section>

        <section className="rounded-xl border border-blue-100 bg-white p-4 shadow-lg">
          <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-blue-950">证据相关性排序</h2>
              <p className="mt-1 text-sm text-slate-500">从高到低展示当前问题的证据节点。</p>
            </div>
            {displayQuestion.answer && sortedSteps.length > 0 ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-3 py-1 text-sm text-emerald-700">
                <CheckCircle2 size={16} />
                已生成答案
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-3 py-1 text-sm text-amber-700">
                <AlertTriangle size={16} />
                {isUploadRunning ? '正在分析' : '待运行检索'}
              </span>
            )}
          </div>

          {displayQuestion.answer && sortedSteps.length > 0 && (
            <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-semibold text-slate-950">答案</p>
              <p className="mt-2 leading-7 text-slate-600">{displayQuestion.answer}</p>
            </div>
          )}

          <div className="space-y-4">
            {sortedSteps.map((step, index) => {
              const key = `${step.chain_step}-${step.node_id}`;
              const selected = selectedStep && `${selectedStep.chain_step}-${selectedStep.node_id}` === key;
              return (
                <EvidenceRow
                  key={key}
                  step={step}
                  rank={index + 1}
                  selected={Boolean(selected)}
                  onSelect={() => setSelectedNodeId(key)}
                />
              );
            })}

            {sortedSteps.length === 0 && (
              <EmptyEvidence running={isUploadRunning} status={jobStatus} error={jobError} />
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function SummaryCell({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3">
      <div className="text-lg font-semibold text-slate-950">{value}</div>
      <div>{label}</div>
    </div>
  );
}

function PipelineStatus({
  variant,
  candidateRetriever,
  rerankRetriever,
  features,
}: {
  variant: string;
  candidateRetriever: string;
  rerankRetriever: string;
  features: string[];
}) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2 rounded-lg border border-blue-100 bg-blue-50/60 px-3 py-2 text-xs text-blue-800">
      <span className="rounded-full bg-white px-2.5 py-1 font-semibold text-blue-900">{variant}</span>
      <span>retrieve: {candidateRetriever}</span>
      <span>rerank: {rerankRetriever}</span>
      {features.map((feature) => (
        <span key={feature} className="rounded-full bg-white/80 px-2.5 py-1">
          {feature}
        </span>
      ))}
    </div>
  );
}

function JobPlaceholder({
  running,
  status,
  error,
}: {
  running: boolean;
  status: UploadJobStatus | null;
  error: string;
}) {
  if (error || status?.status === 'failed') {
    return (
      <div className="grid min-h-[520px] place-items-center rounded-lg border border-dashed border-rose-200 bg-rose-50 p-6 text-center text-rose-700">
        <div>
          <AlertTriangle className="mx-auto mb-3" size={28} />
          <p className="font-medium">后端分析失败</p>
          <p className="mt-2 text-sm leading-6">{error || status?.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-dashed border-blue-200 bg-blue-50/50 p-6 text-center text-slate-600">
      <Loader2 className={`mx-auto mb-3 text-blue-600 ${running ? 'animate-spin' : ''}`} size={30} />
      <p className="font-medium">{status?.message || '等待后端开始处理 PDF'}</p>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-blue-100">
        <div
          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all"
          style={{ width: `${Math.max(4, Math.min(100, status?.progress ?? 4))}%` }}
        />
      </div>
      <p className="mt-2 text-sm text-slate-500">{status?.stage || 'queued'}</p>
      {status?.logs?.length ? (
        <div className="mt-4 max-h-48 overflow-y-auto rounded-lg bg-white p-3 text-left text-xs leading-5 text-slate-500">
          {status.logs.slice(-8).map((line, index) => (
            <p key={`${line}-${index}`}>{line}</p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function EmptyEvidence({
  running,
  status,
  error,
}: {
  running: boolean;
  status: UploadJobStatus | null;
  error: string;
}) {
  let text = '该问题还没有可展示的证据结果。';
  if (running) {
    text = status?.message || '后端正在生成证据链，请稍等。';
  }
  if (error || status?.status === 'failed') {
    text = error || status?.error || '后端分析失败。';
  }
  return (
    <div className="grid min-h-[520px] place-items-center rounded-lg border border-dashed border-blue-200 bg-blue-50/50 p-8 text-center text-slate-500">
      <div>
        {running ? <Loader2 className="mx-auto mb-3 animate-spin text-blue-600" size={30} /> : null}
        <p>{text}</p>
      </div>
    </div>
  );
}

function EvidenceRow({
  step,
  rank,
  selected,
  onSelect,
}: {
  step: EvidenceStep;
  rank: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const imageUrl = step.crop_url || step.page_url;
  return (
    <button
      onClick={onSelect}
      className={`w-full rounded-xl border p-4 text-left transition ${
        selected ? 'border-blue-400 bg-blue-50 shadow-md' : 'border-slate-200 bg-slate-50 hover:border-blue-200'
      }`}
    >
      <div className="flex gap-4">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white">
          {rank}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1 rounded-lg bg-white px-2.5 py-1 text-sm text-blue-700">
              {nodeIcon(step)}
              {roleLabels[step.role] ?? step.role}
            </span>
            <span className="rounded-lg bg-white px-2.5 py-1 text-sm text-slate-600">第 {step.page} 页</span>
            <span className="rounded-lg bg-white px-2.5 py-1 text-sm text-slate-600">{step.node_type}</span>
            <span className="ml-auto rounded-lg bg-blue-100 px-2.5 py-1 text-sm text-blue-700">
              {formatPercent(step.score)}
            </span>
          </div>

          <div className="mt-3 flex flex-col gap-4 xl:flex-row">
            {imageUrl && (
              <img
                src={imageUrl}
                alt={step.node_id}
                className="h-32 w-full rounded-lg border border-blue-100 bg-white object-contain xl:w-44"
              />
            )}
            <div className="min-w-0 flex-1">
              <p className="line-clamp-3 leading-7 text-slate-800">
                {step.content_preview || step.visual_caption || step.visual_summary || step.node_id}
              </p>
              {(step.visual_caption || step.visual_summary) && (
                <p className="mt-2 line-clamp-2 text-sm leading-6 text-slate-500">
                  {step.visual_caption || step.visual_summary}
                </p>
              )}
            </div>
          </div>

          <div className="mt-3 flex items-center gap-3">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-blue-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                style={{ width: scoreWidth(step.score) }}
              />
            </div>
            <span className="text-sm text-slate-500">相关性</span>
          </div>
        </div>
      </div>
    </button>
  );
}
