import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type FormEvent, type ReactNode } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CreditCard,
  Download,
  ExternalLink,
  FileQuestion,
  FileSearch,
  FileText,
  FolderOpen,
  HelpCircle,
  Layers,
  Link2,
  Loader2,
  Play,
  Search,
  Settings,
  Sparkles,
  Upload,
} from 'lucide-react';
import type { AppData, ChunkTemplate, EvidenceStep, PdfItem, QuestionDetail, QuestionItem, UploadJobStatus } from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? '';

type MenuId = 'workspace' | 'documents' | 'questions' | 'upload' | 'metrics' | 'settings';
type BackendStatus = 'checking' | 'online' | 'offline';

const MENU_ITEMS: { id: MenuId; label: string; icon: typeof FileText }[] = [
  { id: 'upload', label: '上传分析', icon: Upload },
  { id: 'workspace', label: '案例复盘', icon: FileText },
  { id: 'documents', label: '文档档案', icon: FolderOpen },
  { id: 'questions', label: '参考问题', icon: HelpCircle },
  { id: 'metrics', label: '效果回看', icon: BarChart3 },
  { id: 'settings', label: '作品状态', icon: Settings },
];

const CATEGORY_OPTIONS = ['全部', '表格/数据', '跨文档', '多模态证据', '风险/条款', '技术/业务'];

const COMPANY_HINTS = [
  { label: 'Microsoft', patterns: ['Microsoft', '微软', 'MSFT', 'Azure', 'Copilot'] },
  { label: 'Apple', patterns: ['Apple', '苹果', 'AAPL', 'iPhone', 'iPad', 'Mac'] },
  { label: 'NVIDIA', patterns: ['NVIDIA', '英伟达', 'NVDA', 'GPU', 'CUDA', 'Blackwell'] },
  { label: 'Tesla', patterns: ['Tesla', '特斯拉', 'TSLA', '电池', '自动驾驶'] },
];

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function readResponseError(response: Response) {
  const text = await response.text();
  if (!text) {
    return `请求失败：HTTP ${response.status}`;
  }
  try {
    const payload = JSON.parse(text) as { detail?: string };
    return payload.detail || text;
  } catch {
    return text;
  }
}

function formatDate(value: string) {
  if (!value) {
    return '未生成';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatPercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  return `${(value * 100).toFixed(0)}%`;
}

function inferCompanies(question: QuestionItem) {
  const haystack = [
    question.doc_id,
    question.question,
    question.answer,
    question.question_type,
    question.evidence_note,
  ]
    .join(' ')
    .toLowerCase();

  const companies = COMPANY_HINTS.filter((company) =>
    company.patterns.some((pattern) => haystack.includes(pattern.toLowerCase())),
  ).map((company) => company.label);

  if (companies.length > 0) {
    return companies;
  }
  if (question.doc_id && question.doc_id !== '金融类') {
    return [question.doc_id];
  }
  return ['演示样例'];
}

function getVisiblePdfs(pdfs: PdfItem[]) {
  return pdfs.filter((pdf) => pdf.doc_id !== '金融类' || pdf.pages > 0 || pdf.node_count > 0);
}

function truncateText(value: string, maxLength = 160) {
  const clean = value.replace(/```json|```/g, '').replace(/\s+/g, ' ').trim();
  if (clean.length <= maxLength) {
    return clean;
  }
  return `${clean.slice(0, maxLength - 1)}...`;
}

function evidenceText(step: EvidenceStep, maxLength = 160) {
  return truncateText(step.reason || step.content_preview || step.visual_summary || step.visual_caption || step.node_id, maxLength);
}

function sourcePages(steps: EvidenceStep[]) {
  return Array.from(new Set(steps.map((step) => step.page).filter(Boolean))).slice(0, 6);
}

function categoryMatches(question: QuestionItem, category: string, companies: string[]) {
  if (category === '全部') {
    return true;
  }
  const text = `${question.question} ${question.answer} ${question.question_type} ${question.evidence_note}`.toLowerCase();
  if (category === '表格/数据') {
    return question.question_type.includes('表格') || question.question_type.includes('数据') || question.gold_modalities.includes('table');
  }
  if (category === '跨文档') {
    return companies.length > 1 || question.doc_id === '金融类' || text.includes('跨');
  }
  if (category === '多模态证据') {
    return /图|表|视觉|图片|页面|图表|截图|定位/.test(text) || question.gold_modalities.some((modality) => modality !== 'text');
  }
  if (category === '风险/条款') {
    return /风险|供应链|集中度|依赖|不确定|条款|政策|合规|责任|限制/.test(text);
  }
  if (category === '技术/业务') {
    return /ai|azure|gpu|数据中心|基础设施|copilot|blackwell|cuda|技术|业务|平台|产品|系统/.test(text);
  }
  return true;
}

function getQuestionDetailUrl(question: QuestionItem) {
  return question.detail_url || `/app-data/details/${encodeURIComponent(question.question_id)}.json`;
}

function getQuestionStepCount(question: QuestionItem, data?: AppData) {
  return question.num_steps || (data?.chains[question.question_id] ?? []).length;
}

function getChainReadyCount(data: AppData) {
  return data.questions.filter((question) => getQuestionStepCount(question, data) > 0).length;
}

function getQuestionStatus(question: QuestionItem, stepCount: number) {
  if (stepCount > 0 && question.card_url) {
    return 'ready';
  }
  if (stepCount > 0) {
    return 'chain';
  }
  return 'empty';
}

function formatJobStatus(status?: string, hasError = false) {
  if (hasError || status === 'failed') {
    return '分析中断';
  }
  if (status === 'succeeded') {
    return '分析完成';
  }
  if (status === 'running') {
    return '正在分析';
  }
  if (status === 'queued') {
    return '排队中';
  }
  return '准备中';
}

const LIVE_PIPELINE_STAGES = [
  { id: 'parse', label: '解析切块' },
  { id: 'visual', label: '视觉定位' },
  { id: 'graph', label: '关系图' },
  { id: 'kg', label: 'GraphRAG' },
  { id: 'retrieve', label: '召回' },
  { id: 'rerank', label: '重排' },
  { id: 'chain', label: '证据链' },
  { id: 'card', label: '卡片' },
  { id: 'done', label: '完成' },
] as const;

function liveStageIndex(stage?: string) {
  const index = LIVE_PIPELINE_STAGES.findIndex((item) => item.id === stage);
  if (stage === 'failed') {
    return LIVE_PIPELINE_STAGES.length;
  }
  return index >= 0 ? index : -1;
}

function nodeIconType(type: string) {
  if (type === 'table') {
    return '表格';
  }
  if (type === 'figure' || type === 'caption') {
    return '视觉';
  }
  return '文本';
}

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => window.matchMedia('(max-width: 760px)').matches);

  useEffect(() => {
    const media = window.matchMedia('(max-width: 760px)');
    const update = () => setIsMobile(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  return isMobile;
}

export default function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState('');
  const [questionDetails, setQuestionDetails] = useState<Record<string, QuestionDetail>>({});
  const [loadingDetailId, setLoadingDetailId] = useState('');
  const [currentMenu, setCurrentMenu] = useState<MenuId>('upload');
  const [selectedQuestionId, setSelectedQuestionId] = useState('');
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('全部');
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const isMobile = useIsMobile();

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        const response = await fetch('/app-data.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`无法读取案例数据：${response.status}`);
        }
        const nextData = (await response.json()) as AppData;
        if (!cancelled) {
          setData(nextData);
          setSelectedQuestionId(nextData.questions[0]?.question_id ?? '');
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '页面数据加载失败');
        }
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 2200);

    fetch(`${API_BASE}/api/health`, { signal: controller.signal })
      .then((response) => setBackendStatus(response.ok ? 'online' : 'offline'))
      .catch(() => setBackendStatus('offline'))
      .finally(() => window.clearTimeout(timeoutId));

    return () => {
      window.clearTimeout(timeoutId);
      controller.abort();
    };
  }, []);

  const selectedQuestion = useMemo(() => {
    if (!data) {
      return null;
    }
    return data.questions.find((question) => question.question_id === selectedQuestionId) ?? data.questions[0] ?? null;
  }, [data, selectedQuestionId]);

  useEffect(() => {
    if (!selectedQuestion) {
      return;
    }

    const qid = selectedQuestion.question_id;
    if (!qid || questionDetails[qid]) {
      return;
    }

    const controller = new AbortController();
    setLoadingDetailId(qid);

    async function loadQuestionDetail() {
      try {
        const response = await fetch(getQuestionDetailUrl(selectedQuestion), {
          cache: 'force-cache',
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`detail ${response.status}`);
        }
        const detail = (await response.json()) as QuestionDetail;
        setQuestionDetails((current) => ({ ...current, [qid]: detail }));
      } catch (err) {
        if (!controller.signal.aborted) {
          console.warn('Failed to load question detail', err);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingDetailId((current) => (current === qid ? '' : current));
        }
      }
    }

    loadQuestionDetail();
    return () => {
      controller.abort();
    };
  }, [questionDetails, selectedQuestion]);

  const selectedDetail = selectedQuestion ? questionDetails[selectedQuestion.question_id] : null;

  const selectedSteps = useMemo(() => {
    if (!data || !selectedQuestion) {
      return [];
    }
    const detailSteps = selectedDetail?.steps ?? data.chains[selectedQuestion.question_id] ?? [];
    return [...detailSteps].sort(
      (a, b) => b.score - a.score || a.chain_step - b.chain_step,
    );
  }, [data, selectedDetail, selectedQuestion]);

  if (error && !data) {
    return <StartupUpload backendStatus={backendStatus} dataError={error} />;
  }

  if (!data) {
    return <StartupUpload backendStatus={backendStatus} />;
  }

  if (isMobile) {
    return (
      <MobileApp
        data={data}
        selectedQuestion={selectedQuestion}
        selectedSteps={selectedSteps}
        selectedQuestionId={selectedQuestion?.question_id ?? ''}
        onSelectQuestion={setSelectedQuestionId}
        backendStatus={backendStatus}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <TopNavBar data={data} backendStatus={backendStatus} />
      <div className="flex items-start">
        <Sidebar currentMenu={currentMenu} onMenuChange={setCurrentMenu} />
        <main className="min-w-0 flex-1" aria-busy={Boolean(loadingDetailId)}>
          {currentMenu === 'workspace' && (
            <WorkspaceView
              data={data}
              selectedQuestion={selectedQuestion}
              selectedSteps={selectedSteps}
              selectedQuestionId={selectedQuestion?.question_id ?? ''}
              query={query}
              category={category}
              onQueryChange={setQuery}
              onCategoryChange={setCategory}
              onSelectQuestion={setSelectedQuestionId}
              onOpenUpload={() => setCurrentMenu('upload')}
            />
          )}
          {currentMenu === 'documents' && (
            <SimplePage title="PDF 文档档案" subtitle="查看已纳入作品的 PDF、结构化片段与多模态证据。">
              <DocumentPanel data={data} question={selectedQuestion} steps={selectedSteps} />
            </SimplePage>
          )}
          {currentMenu === 'questions' && (
            <SimplePage title="参考问题" subtitle="从复杂文档案例中沉淀的参考问题，可用于演示、对照和答辩复盘。">
              <QuestionList
                data={data}
                selectedQuestionId={selectedQuestion?.question_id ?? ''}
                query={query}
                category={category}
                onQueryChange={setQuery}
                onCategoryChange={setCategory}
                onSelectQuestion={setSelectedQuestionId}
              />
            </SimplePage>
          )}
          {currentMenu === 'upload' && <UploadAnalysis data={data} onOpenWorkspace={() => setCurrentMenu('workspace')} />}
          {currentMenu === 'metrics' && (
            <SimplePage title="效果回看" subtitle="回看复杂文档问答的召回、排序、证据覆盖与视觉定位表现。">
              <MetricsView data={data} />
            </SimplePage>
          )}
          {currentMenu === 'settings' && (
            <SimplePage title="作品状态" subtitle="确认在线分析、案例资料和复盘内容是否准备妥当。">
              <SettingsView backendStatus={backendStatus} />
            </SimplePage>
          )}
        </main>
      </div>
    </div>
  );
}

function StartupUpload({ backendStatus, dataError }: { backendStatus: BackendStatus; dataError?: string }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white px-6 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 text-lg font-bold text-white">
            M
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-normal text-slate-950">复杂文档证据问答</h1>
            <p className="text-xs text-slate-500">案例数据仍在加载时，也可以先上传 PDF 分析。</p>
          </div>
        </div>
        <span
          className={cn(
            'inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium',
            backendStatus === 'online'
              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
              : backendStatus === 'checking'
                ? 'border-blue-200 bg-blue-50 text-blue-700'
                : 'border-rose-200 bg-rose-50 text-rose-700',
          )}
        >
          <Activity size={14} />
          {backendStatus === 'online' ? '在线分析就绪' : backendStatus === 'checking' ? '正在连接后端' : '后端未连接'}
        </span>
      </header>
      {dataError && (
        <div className="mx-6 mt-4 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-800">
          历史案例数据暂时没有加载成功：{dataError}。上传分析功能仍可继续使用。
        </div>
      )}
      <UploadAnalysis />
    </div>
  );
}

function TopNavBar({ data, backendStatus }: { data: AppData; backendStatus: BackendStatus }) {
  const statusClass =
    backendStatus === 'online'
      ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
      : backendStatus === 'checking'
        ? 'border-blue-200 bg-blue-50 text-blue-700'
        : 'border-rose-200 bg-rose-50 text-rose-700';

  return (
    <header className="sticky top-0 z-20 flex h-16 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-6 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 text-lg font-bold text-white">
          M
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-normal text-slate-950">复杂文档证据问答</h1>
          <p className="text-xs text-slate-500">让 PDF 里的文字、表格、图像与版面结构互相作证</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <span className={cn('inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium', statusClass)}>
          <Activity size={14} />
          {backendStatus === 'online' ? '在线分析就绪' : backendStatus === 'checking' ? '正在连接' : '等待连接'}
        </span>
        <span className="hidden text-xs text-slate-500 lg:block">更新：{formatDate(data.generated_at)}</span>
        <a
          href="/app-data.json"
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-9 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 transition hover:border-blue-300 hover:text-blue-700"
        >
          <Download size={16} />
          案例快照
        </a>
      </div>
    </header>
  );
}

function Sidebar({ currentMenu, onMenuChange }: { currentMenu: MenuId; onMenuChange: (menu: MenuId) => void }) {
  return (
    <aside className="sticky top-16 flex h-[calc(100vh-4rem)] w-64 shrink-0 flex-col overflow-y-auto border-r border-slate-200 bg-white">
      <nav className="flex-1 space-y-1 p-4">
        {MENU_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = currentMenu === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onMenuChange(item.id)}
              className={cn(
                'flex w-full items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition',
                isActive
                  ? 'bg-blue-600 text-white shadow-sm shadow-blue-200'
                  : 'text-slate-600 hover:bg-blue-50 hover:text-blue-700',
              )}
            >
              <Icon size={18} />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div className="border-t border-slate-200 p-4 text-xs leading-5 text-slate-500">
        当前流程：<span className="font-medium text-slate-700">上传 PDF → 提出问题 → 查看证据</span>
      </div>
    </aside>
  );
}

function WorkspaceView({
  data,
  selectedQuestion,
  selectedSteps,
  selectedQuestionId,
  query,
  category,
  onQueryChange,
  onCategoryChange,
  onSelectQuestion,
  onOpenUpload,
}: {
  data: AppData;
  selectedQuestion: QuestionItem | null;
  selectedSteps: EvidenceStep[];
  selectedQuestionId: string;
  query: string;
  category: string;
  onQueryChange: (value: string) => void;
  onCategoryChange: (value: string) => void;
  onSelectQuestion: (questionId: string) => void;
  onOpenUpload: () => void;
}) {
  return (
    <div className="space-y-6 p-6">
      <DataHealthBar data={data} />
      <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-6 py-4">
          <div>
            <h2 className="text-xl font-semibold text-slate-950">案例复盘</h2>
            <p className="mt-1 text-sm text-slate-500">按问题回看答案、出处、证据卡片和相关 PDF 文档。</p>
          </div>
          <button
            onClick={onOpenUpload}
            className="inline-flex h-10 items-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-cyan-500 px-4 text-sm font-medium text-white shadow-sm transition hover:from-blue-700 hover:to-cyan-600"
          >
            <Upload size={16} />
            上传新 PDF
          </button>
        </div>

        <div className="grid grid-cols-12 gap-0">
          <div className="col-span-4 border-r border-slate-200 bg-slate-50/70 p-4">
            <QuestionList
              data={data}
              selectedQuestionId={selectedQuestionId}
              query={query}
              category={category}
              onQueryChange={onQueryChange}
              onCategoryChange={onCategoryChange}
              onSelectQuestion={onSelectQuestion}
            />
          </div>
          <div className="col-span-8 min-w-0 bg-slate-50/60 p-5">
            <CaseReviewPanel data={data} question={selectedQuestion} steps={selectedSteps} />
          </div>
        </div>
      </section>
    </div>
  );
}

function DataHealthBar({ data }: { data: AppData }) {
  const totalQuestions = data.questions.length;
  const visiblePdfs = getVisiblePdfs(data.pdfs);
  const chainReady = getChainReadyCount(data);
  const cardCount = data.questions.filter((question) => question.card_url).length || data.corpus.num_cards;
  const hasNoEvidence = totalQuestions > 0 && (chainReady === 0 || cardCount === 0);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-4">
        <StatCard icon={<FileText size={20} />} label="PDF 文档" value={visiblePdfs.length} />
        <StatCard icon={<FileQuestion size={20} />} label="案例问题" value={totalQuestions} />
        <StatCard
          icon={<Link2 size={20} />}
          label="证据链"
          value={`${chainReady} / ${totalQuestions}`}
          valueClass={chainReady === 0 ? 'text-amber-600' : 'text-emerald-600'}
        />
        <StatCard
          icon={<CreditCard size={20} />}
          label="证据卡片"
          value={cardCount}
          valueClass={cardCount === 0 ? 'text-amber-600' : 'text-emerald-600'}
        />
      </div>

      {hasNoEvidence && (
        <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-800">
          <AlertTriangle className="mt-0.5 shrink-0 text-amber-600" size={18} />
          <div>
            当前案例资料还在整理中。完成分析后，这里会呈现可追溯的证据结果和卡片。
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  valueClass = 'text-slate-950',
}: {
  icon: ReactNode;
  label: string;
  value: string | number;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="grid h-10 w-10 place-items-center rounded-lg bg-blue-50 text-blue-600">{icon}</div>
      <div>
        <p className="text-xs text-slate-500">{label}</p>
        <p className={cn('mt-0.5 text-2xl font-semibold', valueClass)}>{value}</p>
      </div>
    </div>
  );
}

function QuestionList({
  data,
  selectedQuestionId,
  query,
  category,
  onQueryChange,
  onCategoryChange,
  onSelectQuestion,
}: {
  data: AppData;
  selectedQuestionId: string;
  query: string;
  category: string;
  onQueryChange: (value: string) => void;
  onCategoryChange: (value: string) => void;
  onSelectQuestion: (questionId: string) => void;
}) {
  const filteredQuestions = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return data.questions.filter((question) => {
      const companies = inferCompanies(question);
      const searchable = [
        question.question_id,
        question.question,
        question.answer,
        question.question_type,
        question.evidence_note,
        companies.join(' '),
      ]
        .join(' ')
        .toLowerCase();
      return (!keyword || searchable.includes(keyword)) && categoryMatches(question, category, companies);
    });
  }, [category, data.questions, query]);

  return (
    <div className="flex h-[calc(100vh-6rem)] min-h-[460px] max-h-[900px] flex-col rounded-lg border border-slate-200 bg-white lg:sticky lg:top-20">
      <div className="border-b border-slate-200 p-4">
        <div className="mb-3 flex items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-950">问题索引</h3>
            <p className="mt-1 text-xs text-slate-500">只在这里滚动切换问题</p>
          </div>
          <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-500">{filteredQuestions.length} 条</span>
        </div>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            className="h-10 w-full rounded-lg border border-slate-200 bg-slate-50 pl-9 pr-3 text-sm outline-none transition focus:border-blue-400 focus:bg-white focus:ring-2 focus:ring-blue-100"
            placeholder="搜索问题、主题、指标"
          />
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {CATEGORY_OPTIONS.map((option) => (
            <button
              key={option}
              onClick={() => onCategoryChange(option)}
              className={cn(
                'rounded-full border px-3 py-1 text-xs font-medium transition',
                category === option
                  ? 'border-blue-600 bg-blue-600 text-white'
                  : 'border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700',
              )}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {filteredQuestions.map((question) => {
          const companies = inferCompanies(question);
          const stepCount = getQuestionStepCount(question, data);
          const status = getQuestionStatus(question, stepCount);
          return (
            <button
              key={question.question_id}
              onClick={() => onSelectQuestion(question.question_id)}
              className={cn(
                'mb-2 w-full rounded-lg border p-3 text-left transition',
                selectedQuestionId === question.question_id
                  ? 'border-blue-400 bg-blue-50 shadow-sm'
                  : 'border-slate-200 bg-white hover:border-blue-200 hover:bg-slate-50',
              )}
            >
              <div className="mb-2 flex items-start justify-between gap-2">
                <span className="rounded bg-blue-50 px-2 py-1 font-mono text-xs font-semibold text-blue-700">
                  {question.question_id}
                </span>
                <QuestionStatusBadge status={status} />
              </div>
              <p className="line-clamp-2 text-sm font-medium leading-6 text-slate-800">
                {truncateText(question.question, 72)}
              </p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {companies.slice(0, 3).map((company) => (
                  <span key={company} className="rounded-md bg-cyan-50 px-2 py-0.5 text-xs font-medium text-cyan-700">
                    {company}
                  </span>
                ))}
              </div>
            </button>
          );
        })}

        {filteredQuestions.length === 0 && (
          <div className="grid min-h-48 place-items-center rounded-lg border border-dashed border-slate-200 p-6 text-center text-sm text-slate-500">
            没有匹配的问题。
          </div>
        )}
      </div>
    </div>
  );
}

function QuestionStatusBadge({ status }: { status: 'ready' | 'chain' | 'empty' }) {
  if (status === 'ready') {
    return <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700">已完成</span>;
  }
  if (status === 'chain') {
    return <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700">有证据链</span>;
  }
  return <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700">待生成</span>;
}

function CaseReviewPanel({ data, question, steps }: { data: AppData; question: QuestionItem | null; steps: EvidenceStep[] }) {
  if (!question) {
    return <EmptyCard icon={<HelpCircle size={34} />} title="暂无问题" description="请先选择一个案例问题。" />;
  }

  const pages = sourcePages(steps);
  const relatedDocs = getRelatedDocs(data, question);

  return (
    <div className="space-y-5">
      <section className="overflow-hidden rounded-lg border border-blue-100 bg-white shadow-sm">
        <div className="border-b border-blue-100 bg-gradient-to-r from-blue-600 via-sky-500 to-cyan-500 px-5 py-4 text-white">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded bg-white/15 px-2 py-1 font-mono text-xs font-semibold">{question.question_id}</span>
            <span className="rounded-full border border-white/30 bg-white/10 px-2.5 py-1 text-xs">{question.question_type || '未分类'}</span>
            <span className="ml-auto rounded-full bg-white/15 px-2.5 py-1 text-xs">{steps.length} 条证据线索</span>
          </div>
          <h3 className="mt-4 text-xl font-semibold leading-8 tracking-normal">{question.question}</h3>
        </div>

        <div className="grid gap-0 lg:grid-cols-[1.4fr_0.9fr]">
          <div className="p-5">
            <div className="mb-3 flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-lg bg-blue-50 text-blue-600">
                <Sparkles size={18} />
              </div>
              <div>
                <h4 className="font-semibold text-slate-950">答案摘录</h4>
                <p className="text-xs text-slate-500">先看结论，再向下看证据卡片</p>
              </div>
            </div>
            <p className="text-sm leading-7 text-slate-700">
              {question.answer || '该问题暂未提供参考答案。'}
            </p>
          </div>

          <div className="border-t border-blue-50 bg-slate-50/70 p-5 lg:border-l lg:border-t-0">
            <h4 className="text-sm font-semibold text-slate-950">关联文档</h4>
            <div className="mt-3 space-y-2">
              {relatedDocs.slice(0, 3).map((pdf) => (
                <div key={pdf.doc_id} className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
                  <FileText className="shrink-0 text-blue-500" size={16} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-slate-700">{pdf.file_name}</p>
                    <p className="text-[11px] text-slate-400">{pdf.pages || '-'} 页 · {pdf.node_count} 片段</p>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {pages.length > 0 ? (
                pages.map((page) => (
                  <span key={page} className="rounded-full bg-blue-50 px-2.5 py-1 text-xs text-blue-700">
                    第 {page} 页
                  </span>
                ))
              ) : (
                <span className="rounded-full bg-amber-50 px-2.5 py-1 text-xs text-amber-700">等待证据页码</span>
              )}
            </div>
          </div>
        </div>
      </section>

      <EvidenceInsightCard question={question} steps={steps} />

      <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h4 className="font-semibold text-slate-950">证据脉络</h4>
            <p className="mt-1 text-xs text-slate-500">向下查看每条证据的页码、截图和摘要。</p>
          </div>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">{steps.length} 条</span>
        </div>
        {steps.length > 0 ? (
          <div className="grid gap-3 xl:grid-cols-2">
            {steps.map((step, index) => (
              <EvidenceStepRow key={`${step.chain_step}-${step.node_id}-${index}`} step={step} rank={index + 1} />
            ))}
          </div>
        ) : (
          <EmptyCard
            icon={<FileSearch size={34} />}
            title="暂无证据结果"
            description="完成分析后，这里会展示出处页码、文档片段和视觉线索。"
          />
        )}
      </section>
    </div>
  );
}

function getRelatedDocs(data: AppData, question: QuestionItem) {
  const companies = inferCompanies(question);
  const visiblePdfs = getVisiblePdfs(data.pdfs);
  const matches = visiblePdfs.filter((pdf) =>
    companies.some((company) => `${pdf.doc_id} ${pdf.file_name}`.toLowerCase().includes(company.toLowerCase())),
  );
  return matches.length > 0 ? matches : visiblePdfs;
}

function SummaryCell({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-blue-100 bg-white/80 p-3 text-center">
      <div className="text-lg font-semibold text-slate-950">{value}</div>
      <div className="mt-0.5 text-[11px] text-slate-500">{label}</div>
    </div>
  );
}

function EvidenceInsightCard({ question, steps, compact = false }: { question: QuestionItem | null; steps: EvidenceStep[]; compact?: boolean }) {
  if (!question) {
    return (
      <div className="grid min-h-56 place-items-center rounded-lg border border-dashed border-slate-200 bg-white p-6 text-center">
        <p className="text-sm text-slate-500">选择问题后生成证据卡片。</p>
      </div>
    );
  }

  const topSteps = steps.slice(0, compact ? 2 : 3);
  const pages = sourcePages(steps);
  const leadStep = topSteps[0];

  return (
    <section className="overflow-hidden rounded-lg border border-blue-100 bg-white shadow-sm">
      <div className="bg-slate-950 px-5 py-4 text-white">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-blue-500/20 px-3 py-1 text-xs font-medium text-cyan-100">可追溯证据卡</span>
          <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-200">{question.question_id}</span>
          <span className="ml-auto rounded-full bg-emerald-400/15 px-3 py-1 text-xs text-emerald-100">
            {question.quality_status === 'pass' ? '证据完整' : '等待完善'}
          </span>
        </div>
        <p className="mt-3 line-clamp-2 text-base font-semibold leading-7">{question.question}</p>
      </div>

      <div className="grid gap-0 lg:grid-cols-[1fr_1.25fr]">
        <div className="border-b border-blue-50 bg-gradient-to-br from-blue-50 to-white p-5 lg:border-b-0 lg:border-r">
          <p className="text-xs font-semibold tracking-wide text-blue-600">答案</p>
          <p className={cn('mt-3 text-sm leading-7 text-slate-700', compact ? 'line-clamp-5' : '')}>
            {question.answer || '暂无答案。'}
          </p>

          <div className="mt-5 grid grid-cols-3 gap-2">
            <SummaryCell label="证据" value={steps.length} />
            <SummaryCell label="视觉" value={question.visual_node_steps || 0} />
            <SummaryCell label="裁剪" value={question.crop_steps || 0} />
          </div>
        </div>

        <div className="p-5">
          {leadStep ? (
            <div className="grid gap-4 sm:grid-cols-[160px_1fr]">
              {leadStep.crop_url || leadStep.page_url ? (
                <img
                  src={leadStep.crop_url || leadStep.page_url}
                  alt={leadStep.node_id}
                  className="h-36 w-full rounded-lg border border-blue-100 bg-slate-50 object-contain"
                />
              ) : (
                <div className="grid h-36 place-items-center rounded-lg border border-dashed border-blue-100 bg-blue-50 text-blue-300">
                  <Layers size={30} />
                </div>
              )}
              <div className="min-w-0">
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700">
                    {nodeIconType(leadStep.node_type)}
                  </span>
                  <span className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600">第 {leadStep.page || '-'} 页</span>
                  <span className="rounded-md bg-emerald-50 px-2 py-1 text-xs text-emerald-700">{formatPercent(leadStep.score)}</span>
                </div>
                <p className="mt-3 text-sm leading-6 text-slate-700">{evidenceText(leadStep, compact ? 110 : 170)}</p>
              </div>
            </div>
          ) : (
            <div className="grid min-h-36 place-items-center rounded-lg border border-dashed border-slate-200 bg-slate-50 p-6 text-center text-sm text-slate-500">
              完成分析后会在这里绘制证据截图与摘要。
            </div>
          )}

          {topSteps.length > 1 && (
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              {topSteps.slice(1).map((step, index) => (
                <div key={`${step.chain_step}-${step.node_id}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="rounded-full bg-white px-2 py-0.5 text-xs font-medium text-slate-500">证据 {index + 2}</span>
                    <span className="text-xs text-blue-700">第 {step.page || '-'} 页</span>
                  </div>
                  <p className="line-clamp-3 text-xs leading-5 text-slate-600">{evidenceText(step, 96)}</p>
                </div>
              ))}
            </div>
          )}

          <div className="mt-4 flex flex-wrap gap-2">
            {pages.map((page) => (
              <span key={page} className="rounded-full border border-blue-100 bg-blue-50 px-2.5 py-1 text-xs text-blue-700">
                第 {page} 页
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function QuestionAnswerPanel({ question, steps }: { question: QuestionItem | null; steps: EvidenceStep[] }) {
  if (!question) {
    return <EmptyCard icon={<HelpCircle size={34} />} title="暂无问题" description="请先在左侧选择一个问题。" />;
  }

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="rounded bg-blue-50 px-2 py-1 font-mono text-xs font-semibold text-blue-700">
            {question.question_id}
          </span>
          <span className="rounded-full border border-blue-100 bg-blue-50 px-2.5 py-1 text-xs text-blue-700">
            {question.question_type || '未分类'}
          </span>
        </div>
        <h3 className="text-lg font-semibold leading-8 text-slate-950">{question.question}</h3>
      </section>

      <section className="rounded-lg border border-blue-100 bg-gradient-to-br from-blue-50 to-cyan-50 p-5">
        <div className="mb-3 flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg bg-blue-600 text-white">
            <Sparkles size={18} />
          </div>
          <div>
            <h4 className="font-semibold text-slate-950">答案摘录</h4>
            <p className="text-xs text-slate-500">来自案例问题集的对照答案</p>
          </div>
        </div>
        <p className="text-sm leading-7 text-slate-700">{question.answer || '该问题暂未提供参考答案。'}</p>
      </section>

      <EvidenceInsightCard question={question} steps={steps} compact />

      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h4 className="font-semibold text-slate-950">证据脉络</h4>
            <p className="mt-1 text-xs text-slate-500">按相关性整理出处页码、视觉线索和片段摘要。</p>
          </div>
          <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">{steps.length} 条证据</span>
        </div>

        {steps.length > 0 ? (
          <div className="space-y-3">
            {steps.map((step, index) => (
              <EvidenceStepRow key={`${step.chain_step}-${step.node_id}-${index}`} step={step} rank={index + 1} />
            ))}
          </div>
        ) : (
          <div className="grid min-h-72 place-items-center rounded-lg border border-dashed border-slate-200 bg-slate-50 p-8 text-center">
            <div>
              <div className="mx-auto mb-4 grid h-20 w-20 place-items-center rounded-2xl bg-gradient-to-br from-slate-100 to-blue-100 text-slate-400">
                <FileSearch size={38} />
              </div>
              <p className="font-medium text-slate-700">暂无证据结果</p>
              <p className="mt-2 text-sm leading-6 text-slate-500">
                当前案例只有问题与答案对照。完成分析后即可展示出处页码、文档片段和可视化线索。
              </p>
              <div className="mt-5 flex justify-center gap-3">
                <button className="inline-flex h-9 items-center gap-2 rounded-lg bg-blue-600 px-3 text-sm font-medium text-white opacity-70">
                  <Play size={15} />
                  生成证据
                </button>
                <a
                  href="/app-data.json"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex h-9 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 hover:border-blue-200 hover:text-blue-700"
                >
                  <ExternalLink size={15} />
                  查看案例快照
                </a>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function EvidenceStepRow({ step, rank }: { step: EvidenceStep; rank: number }) {
  const imageUrl = step.crop_url || step.page_url;
  return (
    <article className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-blue-600 text-sm font-semibold text-white">
          {rank}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-md bg-white px-2 py-1 text-xs text-blue-700">{nodeIconType(step.node_type)}</span>
            <span className="rounded-md bg-white px-2 py-1 text-xs text-slate-600">第 {step.page || '-'} 页</span>
            <span className="rounded-md bg-blue-50 px-2 py-1 text-xs text-blue-700">{formatPercent(step.score)}</span>
          </div>
          <div className="mt-3 flex gap-3">
            {imageUrl && (
              <img
                src={imageUrl}
                alt={step.node_id}
                className="h-24 w-32 shrink-0 rounded-lg border border-slate-200 bg-white object-contain"
              />
            )}
            <p className="text-sm leading-6 text-slate-700">
              {step.content_preview || step.visual_caption || step.visual_summary || step.reason || step.node_id}
            </p>
          </div>
        </div>
      </div>
    </article>
  );
}

function DocumentPanel({
  data,
  question,
  steps,
}: {
  data: AppData;
  question: QuestionItem | null;
  steps: EvidenceStep[];
}) {
  const companies = question ? inferCompanies(question) : [];
  const visiblePdfs = useMemo(() => getVisiblePdfs(data.pdfs), [data.pdfs]);
  const relatedDocs = useMemo(() => {
    if (!question) {
      return visiblePdfs;
    }
    const matches = visiblePdfs.filter((pdf) =>
      companies.some((company) => `${pdf.doc_id} ${pdf.file_name}`.toLowerCase().includes(company.toLowerCase())),
    );
    return matches.length > 0 ? matches : visiblePdfs;
  }, [companies, question, visiblePdfs]);

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-slate-200 bg-white p-5">
        <h3 className="mb-4 text-sm font-semibold text-slate-950">相关文档</h3>
        <div className="space-y-3">
          {relatedDocs.map((pdf) => (
            <DocumentRow key={pdf.doc_id} pdf={pdf} />
          ))}
        </div>
      </section>

      <EvidenceInsightCard question={question} steps={steps} compact />
    </div>
  );
}

function DocumentRow({ pdf }: { pdf: PdfItem }) {
  const parsed = pdf.node_count > 0 || pdf.pages > 0;
  return (
    <div className="flex items-start gap-3 rounded-lg border border-slate-200 p-3 transition hover:border-blue-300 hover:bg-blue-50">
      <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-blue-100 text-blue-600">
        <FileText size={19} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-slate-950">{pdf.file_name}</p>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span>{pdf.pages || '-'} 页</span>
          <span>{pdf.node_count} 片段</span>
          <span className={cn('inline-flex items-center gap-1', parsed ? 'text-emerald-600' : 'text-amber-600')}>
            <CheckCircle2 size={12} />
            {parsed ? '已解析' : '待解析'}
          </span>
        </div>
      </div>
    </div>
  );
}

function UploadAnalysis({ data, onOpenWorkspace }: { data?: AppData; onOpenWorkspace?: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState('');
  const [chunkTemplate, setChunkTemplate] = useState<ChunkTemplate>('auto');
  const [jobStatus, setJobStatus] = useState<UploadJobStatus | null>(null);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeRunRef = useRef(0);

  const canSubmit = Boolean(file) && question.trim().length > 0 && !isSubmitting;

  useEffect(() => {
    return () => {
      activeRunRef.current += 1;
    };
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    selectFile(nextFile);
  };

  const selectFile = (nextFile: File | null) => {
    activeRunRef.current += 1;
    setFile(nextFile);
    setError('');
    setJobStatus(null);
    setIsSubmitting(false);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    const nextFile = event.dataTransfer.files?.[0] ?? null;
    if (nextFile && nextFile.type !== 'application/pdf' && !nextFile.name.toLowerCase().endsWith('.pdf')) {
      setError('只支持上传 PDF 文件');
      return;
    }
    selectFile(nextFile);
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || !question.trim()) {
      return;
    }

    const runId = activeRunRef.current + 1;
    activeRunRef.current = runId;
    const isCurrentRun = () => activeRunRef.current === runId;

    try {
      setIsSubmitting(true);
      setError('');
      setJobStatus(null);

      const formData = new FormData();
      formData.append('pdf', file);
      formData.append('question', question.trim());
      formData.append('chunk_template', chunkTemplate);
      formData.append('profile', 'live_fullchain');

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await readResponseError(response));
      }
      const started = (await response.json()) as { job_id: string };

      while (isCurrentRun()) {
        const statusResponse = await fetch(`${API_BASE}/api/jobs/${started.job_id}`, { cache: 'no-store' });
        if (!statusResponse.ok) {
          throw new Error(await readResponseError(statusResponse));
        }
        const nextStatus = (await statusResponse.json()) as UploadJobStatus;
        if (!isCurrentRun()) {
          return;
        }
        setJobStatus(nextStatus);
        if (nextStatus.status === 'succeeded' || nextStatus.status === 'failed') {
          break;
        }
        await sleep(1400);
      }
    } catch (err) {
      if (isCurrentRun()) {
        setError(err instanceof Error ? err.message : '上传分析失败');
      }
    } finally {
      if (isCurrentRun()) {
        setIsSubmitting(false);
      }
    }
  };

  const uploadSteps = jobStatus?.result?.steps ?? [];
  const uploadQuestion = jobStatus?.result?.question;
  const uploadPdfItem = uploadQuestion
    ? buildUploadPdfItem(uploadQuestion, file?.name ?? jobStatus?.pdf_name ?? '上传文档.pdf', uploadSteps)
    : null;
  const uploadData = uploadPdfItem ? { ...emptyAppData(), pdfs: [uploadPdfItem] } : emptyAppData();
  const currentStageIndex = liveStageIndex(jobStatus?.stage);

  return (
    <div className="space-y-6 p-6">
      <section className="overflow-hidden rounded-lg border border-blue-100 bg-white shadow-sm">
        <div className="border-b border-blue-100 bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 px-6 py-6 text-white">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm font-medium text-blue-100">文档洞察工作台</p>
              <h2 className="mt-1 text-2xl font-semibold tracking-normal">把 PDF 变成可以追问的证据链</h2>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-blue-50">
                上传复杂 PDF 文档，提出你关心的问题，页面会把答案、出处页码和视觉线索整理成一份清爽的分析结果。
              </p>
            </div>
            <span className="w-fit rounded-full border border-white/30 bg-white/15 px-3 py-1 text-xs font-medium text-white">
              证据可追溯
            </span>
          </div>
        </div>

        <div className="p-6">
          <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-950">新建分析</h3>
            <p className="mt-1 text-sm text-slate-500">建议优先从这里开始。上传文档并输入问题，稍后即可看到答案与证据。</p>
          </div>
          <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">现场全链路</span>
          </div>

            <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
          <label
            className="block cursor-pointer rounded-lg border-2 border-dashed border-blue-300 bg-blue-50/40 p-8 text-center transition hover:bg-blue-50"
            onDragOver={(event) => event.preventDefault()}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              className="hidden"
              type="file"
              accept="application/pdf,.pdf"
              onChange={handleFileChange}
            />
            <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-full bg-blue-100 text-blue-600">
              <Upload size={32} />
            </div>
            <p className="font-medium text-slate-950">{file ? file.name : '点击上传或拖拽 PDF 文件'}</p>
            <p className="mt-1 text-xs text-slate-500">
              现场演示建议使用 1-8 页、20MB 以内的小 PDF，系统会完整跑解析、视觉定位、召回、重排和证据链。
            </p>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 rounded-lg bg-white px-4 py-2 text-sm font-medium text-blue-700 shadow-sm ring-1 ring-blue-100 transition hover:ring-blue-300"
            >
              选择 PDF
            </button>
          </label>

          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-900">你想问什么</label>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                className="min-h-28 w-full resize-none rounded-lg border border-slate-200 bg-white p-3 text-sm leading-6 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                placeholder="例如：这份文档中的关键结论、图表或条款说明了什么？"
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-900">分析侧重</label>
              <select
                value={chunkTemplate}
                onChange={(event) => setChunkTemplate(event.target.value as ChunkTemplate)}
                className="h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
              >
                <option value="auto">自动识别</option>
                <option value="finance">报表/表格分析</option>
                <option value="ai">AI 技术投资</option>
                <option value="general">通用文档问答</option>
                <option value="math">数学</option>
                <option value="medical">医学</option>
              </select>
            </div>
            <div className="flex gap-3">
              <button
                type="submit"
                disabled={!canSubmit}
                className={cn(
                  'inline-flex h-10 flex-1 items-center justify-center gap-2 rounded-lg text-sm font-medium text-white transition',
                  canSubmit ? 'bg-blue-600 hover:bg-blue-700' : 'cursor-not-allowed bg-slate-300',
                )}
              >
                {isSubmitting ? <Loader2 className="animate-spin" size={16} /> : <Sparkles size={16} />}
                开始分析
              </button>
              <button
                type="button"
                onClick={() => {
                  activeRunRef.current += 1;
                  setQuestion('');
                  setFile(null);
                  if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                  }
                  setJobStatus(null);
                  setError('');
                  setIsSubmitting(false);
                }}
                className="h-10 rounded-lg border border-slate-200 bg-white px-4 text-sm text-slate-700 transition hover:border-blue-300 hover:text-blue-700"
              >
                清空
              </button>
            </div>
          </div>
          </form>
        </div>
      </section>

      {(jobStatus || error) && (
        <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <h3 className="font-semibold text-slate-950">分析进度</h3>
              <p className="mt-1 text-sm text-slate-500">{jobStatus?.message || jobStatus?.stage || '正在准备文档分析'}</p>
            </div>
            <span
              className={cn(
                'rounded-full border px-3 py-1 text-xs font-medium',
                jobStatus?.status === 'succeeded'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                  : jobStatus?.status === 'failed' || error
                    ? 'border-rose-200 bg-rose-50 text-rose-700'
                    : 'border-blue-200 bg-blue-50 text-blue-700',
              )}
            >
              {formatJobStatus(jobStatus?.status, Boolean(error))}
            </span>
          </div>

          {error ? (
            <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm leading-6 text-rose-700">{error}</div>
          ) : (
            <>
              <div className="h-2 overflow-hidden rounded-full bg-blue-100">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-600 to-cyan-500 transition-all"
                  style={{ width: `${Math.max(4, Math.min(100, jobStatus?.progress ?? 4))}%` }}
                />
              </div>
              <div className="mt-4 grid gap-2 sm:grid-cols-3 lg:grid-cols-9">
                {LIVE_PIPELINE_STAGES.map((stage, index) => {
                  const done = currentStageIndex >= index || jobStatus?.status === 'succeeded';
                  const active = jobStatus?.stage === stage.id;
                  return (
                    <div
                      key={stage.id}
                      className={cn(
                        'rounded-lg border px-2.5 py-2 text-center text-xs transition',
                        done
                          ? 'border-blue-200 bg-blue-50 text-blue-700'
                          : 'border-slate-200 bg-slate-50 text-slate-400',
                        active && 'ring-2 ring-blue-200',
                      )}
                    >
                      {stage.label}
                    </div>
                  );
                })}
              </div>
              {jobStatus?.pdf_pages || jobStatus?.file_size_mb ? (
                <p className="mt-3 text-xs text-slate-500">
                  当前文档：{jobStatus?.pdf_pages || '-'} 页，约 {jobStatus?.file_size_mb || '-'} MB；模式：
                  {jobStatus?.profile || 'live_fullchain'}
                </p>
              ) : null}
            </>
          )}

          {jobStatus?.logs?.length ? (
            <div className="mt-4 max-h-36 overflow-y-auto rounded-lg border border-slate-100 bg-slate-50 p-3 text-xs leading-5 text-slate-500">
              {jobStatus.logs.slice(-8).map((line, index) => (
                <p key={`${line}-${index}`}>{line}</p>
              ))}
            </div>
          ) : null}
        </section>
      )}

      {uploadQuestion && (
        <section className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
          <QuestionAnswerPanel question={uploadQuestion} steps={uploadSteps} />
          <DocumentPanel data={uploadData} question={uploadQuestion} steps={uploadSteps} />
        </section>
      )}

      {data && <OfflineDataPreview data={data} onOpenWorkspace={onOpenWorkspace} />}
    </div>
  );
}

function OfflineDataPreview({ data, onOpenWorkspace }: { data: AppData; onOpenWorkspace?: () => void }) {
  const totalQuestions = data.questions.length;
  const visiblePdfs = getVisiblePdfs(data.pdfs);
  const chainReady = getChainReadyCount(data);
  const parsedDocs = visiblePdfs.filter((pdf) => pdf.pages > 0 || pdf.node_count > 0).length;
  const sampleQuestions = data.questions.slice(0, 3);

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">示例与复盘</p>
          <h3 className="mt-1 text-lg font-semibold text-slate-950">案例库与历史结果</h3>
          <p className="mt-1 text-sm leading-6 text-slate-500">
            项目内置的演示问题和历史分析会沉淀在这里，适合做演示对照、效果回看和答辩复盘。
          </p>
        </div>
        <button
          type="button"
          onClick={onOpenWorkspace}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 text-sm font-medium text-blue-700 transition hover:border-blue-300 hover:bg-blue-100"
        >
          <FileText size={16} />
          查看案例工作台
        </button>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <InfoCard
          title="案例文档解析"
          value={`${parsedDocs} / ${visiblePdfs.length}`}
          description="文档页数、片段数量和结构信息会展示在文档档案中。"
        />
        <InfoCard
          title="预设问题"
          value={`${totalQuestions} 条`}
          description="覆盖表格数据、跨文档对比、风险说明、技术业务和视觉证据等场景。"
        />
        <InfoCard
          title="证据链"
          value={`${chainReady} / ${totalQuestions}`}
          description="可在案例复盘中查看答案依据和证据卡片。"
        />
      </div>

      {sampleQuestions.length > 0 && (
        <div className="mt-5 grid gap-3 lg:grid-cols-3">
          {sampleQuestions.map((question) => (
            <div key={question.question_id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="rounded bg-blue-50 px-2 py-1 font-mono text-xs font-semibold text-blue-700">
                  {question.question_id}
                </span>
                <span className="rounded-full bg-white px-2 py-0.5 text-xs text-slate-500">{question.question_type}</span>
              </div>
              <p className="line-clamp-2 text-sm leading-6 text-slate-700">{question.question}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function MetricBar({ label, value, tone = 'blue' }: { label: string; value: number; tone?: 'blue' | 'emerald' | 'amber' }) {
  const width = value > 0 ? `${Math.max(2, Math.min(100, value * 100))}%` : '0%';
  const barClass =
    tone === 'emerald'
      ? 'bg-emerald-500'
      : tone === 'amber'
        ? 'bg-amber-500'
        : 'bg-blue-600';

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-slate-500">{label}</span>
        <span className="font-medium text-slate-700">{formatPercent(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={cn('h-full rounded-full', barClass)} style={{ width }} />
      </div>
    </div>
  );
}

function MetricsView({ data }: { data: AppData }) {
  const metrics = data.metrics;

  if (!metrics.length) {
    return (
      <EmptyCard
        icon={<BarChart3 size={34} />}
        title="暂无效果指标"
        description="当前还没有可展示的效果回看数据。"
      />
    );
  }

  const reviewQuestions = Math.round(metrics[0]?.num_questions || data.questions.length);
  const bestMetric = metrics.reduce((best, metric) =>
    (metric.evidence_hit || 0) > (best.evidence_hit || 0) ? metric : best,
  metrics[0]);
  const chainReady = bestMetric.evidence_chain_ready
    ? Math.round(bestMetric.evidence_chain_ready * reviewQuestions)
    : Math.min(reviewQuestions, getChainReadyCount(data));
  const cardReady = Math.min(reviewQuestions, data.questions.filter((question) => question.card_url).length || data.corpus.num_cards);
  const bestVisual = metrics.reduce((best, metric) => Math.max(best, metric.visual_grounding_hit || 0), 0);

  return (
    <div className="space-y-5">
      <div className="grid gap-4 lg:grid-cols-4">
        <StatCard icon={<FileQuestion size={20} />} label="回看样例" value={reviewQuestions} />
        <StatCard icon={<Link2 size={20} />} label="证据链覆盖" value={`${chainReady} / ${reviewQuestions}`} valueClass="text-emerald-600" />
        <StatCard icon={<CreditCard size={20} />} label="证据卡片" value={cardReady} valueClass="text-emerald-600" />
        <StatCard icon={<FileSearch size={20} />} label="视觉定位" value={formatPercent(bestVisual)} valueClass="text-blue-600" />
      </div>

      <div className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm leading-6 text-blue-800">
        当前展示的是复杂文档 QA 的离线回看集表现。金融年报只是预置演示材料，用来展示跨文档、表格数据、视觉定位和证据链能力，系统本身面向合同、教材、论文、说明书等复杂 PDF。
      </div>

      <div className="grid gap-4 xl:grid-cols-5">
        {metrics.map((metric) => (
          <article key={metric.method} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-950">{metric.method}</h3>
                <p className="text-xs text-slate-500">{metric.num_questions || reviewQuestions} 个样例</p>
              </div>
              <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs text-blue-700">
                {metric.avg_rerank_time_ms ? `${(metric.avg_rerank_time_ms / 1000).toFixed(1)}s` : '快速'}
              </span>
            </div>
            <div className="space-y-3">
              <MetricBar label="召回@5" value={metric.recall_at_5 || 0} tone="blue" />
              <MetricBar label="排序 MRR" value={metric.mrr || 0} tone="emerald" />
              <MetricBar label="证据覆盖" value={metric.evidence_hit || 0} tone="amber" />
              <MetricBar label="视觉定位" value={metric.visual_grounding_hit || 0} tone="blue" />
            </div>
          </article>
        ))}
      </div>

      <section className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-200 px-5 py-4">
          <h3 className="font-semibold text-slate-950">指标明细</h3>
          <p className="mt-1 text-sm text-slate-500">保留核心指标，方便答辩时说明不同方案在召回、排序和证据组织上的差异。</p>
        </div>
        <table className="w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500">
            <tr>
              <th className="px-4 py-3">方案</th>
              <th className="px-4 py-3">Recall@5</th>
              <th className="px-4 py-3">MRR</th>
              <th className="px-4 py-3">证据覆盖</th>
              <th className="px-4 py-3">视觉定位</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => (
              <tr key={metric.method} className="border-t border-slate-100">
                <td className="px-4 py-3 font-medium text-slate-950">{metric.method}</td>
                <td className="px-4 py-3 text-slate-600">{formatPercent(metric.recall_at_5)}</td>
                <td className="px-4 py-3 text-slate-600">{formatPercent(metric.mrr)}</td>
                <td className="px-4 py-3 text-slate-600">{formatPercent(metric.evidence_hit)}</td>
                <td className="px-4 py-3 text-slate-600">{formatPercent(metric.visual_grounding_hit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function SettingsView({ backendStatus }: { backendStatus: BackendStatus }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <InfoCard title="在线分析" value={backendStatus === 'online' ? '已就绪' : '等待连接'} description="用于处理上传文档、整理证据并生成分析结果。" />
      <InfoCard
        title="案例资料"
        value="已同步"
        description="案例复盘所需的问题、文档、证据卡片和评估结果已准备到前端。"
      />
      <InfoCard
        title="复盘内容"
        value="证据链 / 证据卡片 / 评估指标"
        description="案例库会集中展示答案依据、证据卡片和效果回看。"
      />
      <InfoCard
        title="主流程"
        value="上传 PDF 分析"
        description="新建分析会展示处理进度，并在完成后呈现答案和证据。"
      />
    </div>
  );
}

function SimplePage({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-950">{title}</h2>
        <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
      </div>
      {children}
    </div>
  );
}

function InfoCard({ title, value, description }: { title: string; value: string; description: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5">
      <p className="text-sm font-semibold text-slate-950">{title}</p>
      <p className="mt-2 rounded bg-slate-50 px-3 py-2 text-sm font-semibold text-blue-700">{value}</p>
      <p className="mt-3 text-sm leading-6 text-slate-500">{description}</p>
    </div>
  );
}

function EmptyCard({ icon, title, description }: { icon: ReactNode; title: string; description: string }) {
  return (
    <div className="grid min-h-72 place-items-center rounded-lg border border-dashed border-slate-200 bg-white p-8 text-center">
      <div>
        <div className="mx-auto mb-4 grid h-16 w-16 place-items-center rounded-2xl bg-blue-50 text-blue-400">{icon}</div>
        <p className="font-medium text-slate-700">{title}</p>
        <p className="mt-2 text-sm leading-6 text-slate-500">{description}</p>
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="grid min-h-screen place-items-center bg-slate-950 text-white">
      <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-white/5 px-5 py-4">
        <Loader2 className="animate-spin text-cyan-300" size={22} />
        <span>正在整理证据资料...</span>
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <div className="grid min-h-screen place-items-center bg-slate-950 p-6 text-white">
      <div className="max-w-xl rounded-lg border border-red-300/30 bg-red-950/40 p-6 shadow-xl">
        <div className="flex items-center gap-3 text-red-100">
          <AlertTriangle size={24} />
          <h1 className="text-xl">页面数据没有准备好</h1>
        </div>
        <p className="mt-3 text-sm leading-6 text-red-50/80">{error}。请先完成案例数据准备后刷新页面。</p>
      </div>
    </div>
  );
}

function MobileApp({
  data,
  selectedQuestion,
  selectedSteps,
  selectedQuestionId,
  onSelectQuestion,
  backendStatus,
}: {
  data: AppData;
  selectedQuestion: QuestionItem | null;
  selectedSteps: EvidenceStep[];
  selectedQuestionId: string;
  onSelectQuestion: (questionId: string) => void;
  backendStatus: BackendStatus;
}) {
  const [tab, setTab] = useState<'upload' | 'questions' | 'answer' | 'documents'>('upload');
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('全部');

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="border-b border-slate-200 bg-white px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 text-sm font-bold text-white">
            M
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold text-slate-950">复杂文档证据问答</h1>
            <p className="text-xs text-slate-500">PDF 证据追踪 · {backendStatus === 'online' ? '在线' : '未连接'}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 border-b border-slate-200 bg-white px-2">
        {[
          ['upload', '上传'],
          ['questions', '问题'],
          ['answer', '答案'],
          ['documents', '文档'],
        ].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setTab(id as 'upload' | 'questions' | 'answer' | 'documents')}
            className={cn(
              'h-11 border-b-2 text-sm font-medium',
              tab === id ? 'border-blue-600 text-blue-700' : 'border-transparent text-slate-500',
            )}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="p-4">
        {tab === 'questions' && (
          <QuestionList
            data={data}
            selectedQuestionId={selectedQuestionId}
            query={query}
            category={category}
            onQueryChange={setQuery}
            onCategoryChange={setCategory}
            onSelectQuestion={(questionId) => {
              onSelectQuestion(questionId);
              setTab('answer');
            }}
          />
        )}
        {tab === 'answer' && <QuestionAnswerPanel question={selectedQuestion} steps={selectedSteps} />}
        {tab === 'documents' && <DocumentPanel data={data} question={selectedQuestion} steps={selectedSteps} />}
        {tab === 'upload' && <UploadAnalysis data={data} onOpenWorkspace={() => setTab('questions')} />}
      </div>
    </div>
  );
}

function buildUploadPdfItem(question: QuestionItem, fileName: string, steps: EvidenceStep[]): PdfItem {
  const modalities = steps.reduce<Record<string, number>>((counts, step) => {
    const key = step.node_type || 'text';
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
  return {
    doc_id: question.doc_id || fileName,
    file_name: fileName,
    pages: sourcePages(steps).length,
    question_count: 1,
    node_count: steps.length,
    modalities,
  };
}

function emptyAppData(): AppData {
  return {
    generated_at: '',
    corpus: {
      num_pdfs: 0,
      num_questions: 0,
      num_chain_steps: 0,
      num_cards: 0,
      quality_pass: 0,
      quality_warn: 0,
      quality_fail: 0,
    },
    pdfs: [],
    questions: [],
    chains: {},
    rankings: {},
    metrics: [],
  };
}
