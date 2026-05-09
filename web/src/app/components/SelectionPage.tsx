import { useMemo, useState, type ChangeEvent } from 'react';
import {
  ArrowLeft,
  BrainCircuit,
  Calculator,
  ChevronRight,
  FileText,
  HeartPulse,
  Landmark,
  Layers3,
  Search,
  Sparkles,
  Upload,
  type LucideIcon,
} from 'lucide-react';
import type { AnalysisRequest, AppData, ChunkTemplate, PdfItem, QuestionItem } from '../types';

interface SelectionPageProps {
  data: AppData;
  onStartAnalysis: (request: AnalysisRequest) => void;
}

type Step = 'document' | 'question';
type SourceMode = 'system' | 'upload';

const CHUNK_TEMPLATE_OPTIONS: { value: ChunkTemplate; label: string; icon: LucideIcon }[] = [
  { value: 'auto', label: 'Auto', icon: Sparkles },
  { value: 'general', label: '通用', icon: Layers3 },
  { value: 'ai', label: 'AI', icon: BrainCircuit },
  { value: 'math', label: '数学', icon: Calculator },
  { value: 'finance', label: '金融', icon: Landmark },
  { value: 'medical', label: '医学', icon: HeartPulse },
];

function questionTypeClass(type: string) {
  if (type.includes('表格')) {
    return 'border-violet-200 bg-violet-50 text-violet-700';
  }
  if (type.includes('图') || type.includes('视觉') || type.includes('跨模态')) {
    return 'border-cyan-200 bg-cyan-50 text-cyan-700';
  }
  return 'border-blue-200 bg-blue-50 text-blue-700';
}

export function SelectionPage({ data, onStartAnalysis }: SelectionPageProps) {
  const [step, setStep] = useState<Step>('document');
  const [sourceMode, setSourceMode] = useState<SourceMode>('system');
  const [selectedDocId, setSelectedDocId] = useState<string>(data.pdfs[0]?.doc_id ?? '');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [selectedQuestionId, setSelectedQuestionId] = useState('');
  const [customQuestion, setCustomQuestion] = useState('');
  const [selectedChunkTemplate, setSelectedChunkTemplate] = useState<ChunkTemplate>('auto');
  const [query, setQuery] = useState('');

  const selectedPdf = data.pdfs.find((pdf) => pdf.doc_id === selectedDocId) ?? data.pdfs[0] ?? null;

  const presetQuestions = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return data.questions.filter((question) => {
      if (question.doc_id !== selectedDocId) {
        return false;
      }
      if (!keyword) {
        return true;
      }
      return (
        question.question.toLowerCase().includes(keyword) ||
        question.answer.toLowerCase().includes(keyword) ||
        question.question_id.toLowerCase().includes(keyword)
      );
    });
  }, [data.questions, query, selectedDocId]);

  const selectedQuestion =
    data.questions.find((question) => question.question_id === selectedQuestionId) ?? presetQuestions[0] ?? null;

  const selectedPdfName = sourceMode === 'upload' ? uploadedFile?.name ?? '' : selectedPdf?.file_name ?? '';
  const canNext = sourceMode === 'upload' ? Boolean(uploadedFile) : Boolean(selectedPdf);
  const canStart = sourceMode === 'upload' ? customQuestion.trim().length > 0 : Boolean(selectedQuestion);

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (!file) {
      return;
    }
    setUploadedFile(file);
    setSourceMode('upload');
    setSelectedQuestionId('');
    setCustomQuestion('');
  };

  const handleSystemSelect = (pdf: PdfItem) => {
    setSourceMode('system');
    setSelectedDocId(pdf.doc_id);
    setUploadedFile(null);
    setSelectedQuestionId('');
    setCustomQuestion('');
  };

  const handleNext = () => {
    if (canNext) {
      setStep('question');
      setQuery('');
      setSelectedQuestionId('');
    }
  };

  const handleStart = () => {
    if (!canStart) {
      return;
    }
    if (sourceMode === 'upload') {
      if (!uploadedFile) {
        return;
      }
      onStartAnalysis({
        mode: 'upload',
        pdf_name: selectedPdfName,
        question: customQuestion.trim(),
        file: uploadedFile,
        chunk_template: selectedChunkTemplate,
      });
      return;
    }
    if (selectedQuestion && selectedPdf) {
      onStartAnalysis({
        mode: 'system',
        doc_id: selectedPdf.doc_id,
        pdf_name: selectedPdf.file_name,
        question_id: selectedQuestion.question_id,
      });
    }
  };

  return (
    <main className="min-h-screen p-4 sm:p-6 lg:p-8">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] max-w-5xl flex-col justify-center">
        <header className="mb-8 text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-white/75 px-3 py-1 text-sm text-cyan-800 shadow-sm">
            <FileText size={16} />
            多模态 RAG 证据检测
          </div>
          <h1 className="mt-4 text-3xl font-semibold tracking-normal text-slate-950 sm:text-4xl">
            {step === 'document' ? '选择要检测的 PDF' : '选择或输入问题'}
          </h1>
        </header>

        {step === 'document' ? (
          <section className="rounded-xl border border-white/70 bg-white/85 p-5 shadow-sm backdrop-blur sm:p-7">
            <ChunkTemplatePicker value={selectedChunkTemplate} onChange={setSelectedChunkTemplate} />

            <label className="block">
              <input className="hidden" type="file" accept="application/pdf,.pdf" onChange={handleUpload} />
              <div
                className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center transition ${
                  sourceMode === 'upload' && uploadedFile
                    ? 'border-cyan-400 bg-cyan-50'
                    : 'border-slate-200 bg-slate-50 hover:border-cyan-300 hover:bg-cyan-50/50'
                }`}
              >
                <Upload className="mx-auto text-cyan-700" size={34} />
                <p className="mt-3 font-medium text-slate-950">
                  {uploadedFile ? uploadedFile.name : '上传本地 PDF'}
                </p>
                <p className="mt-1 text-sm text-slate-500">
                  {uploadedFile ? '已选择上传文档' : '点击选择 PDF 文件'}
                </p>
              </div>
            </label>

            <div className="my-6 flex items-center gap-4 text-sm text-slate-400">
              <div className="h-px flex-1 bg-slate-200" />
              <span>或选择系统文档</span>
              <div className="h-px flex-1 bg-slate-200" />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {data.pdfs.map((pdf) => (
                <DocumentButton
                  key={pdf.doc_id}
                  pdf={pdf}
                  selected={sourceMode === 'system' && selectedDocId === pdf.doc_id}
                  onSelect={() => handleSystemSelect(pdf)}
                />
              ))}
            </div>

            <div className="mt-7 flex justify-center">
              <button
                onClick={handleNext}
                disabled={!canNext}
                className={`inline-flex h-11 items-center justify-center gap-2 rounded-lg px-6 text-white shadow-sm transition ${
                  canNext ? 'bg-slate-950 hover:bg-slate-800' : 'cursor-not-allowed bg-slate-300'
                }`}
              >
                下一步
                <ChevronRight size={18} />
              </button>
            </div>
          </section>
        ) : (
          <section className="rounded-xl border border-white/70 bg-white/85 p-5 shadow-sm backdrop-blur sm:p-7">
            <button
              onClick={() => setStep('document')}
              className="mb-5 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 transition hover:border-cyan-300 hover:text-cyan-700"
            >
              <ArrowLeft size={16} />
              返回选择文档
            </button>

            <div className="mb-5 rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm text-slate-500">当前文档</p>
              <p className="mt-1 font-semibold text-slate-950">{selectedPdfName}</p>
              {sourceMode === 'upload' && (
                <p className="mt-2 text-sm text-cyan-700">
                  论文领域：{CHUNK_TEMPLATE_OPTIONS.find((option) => option.value === selectedChunkTemplate)?.label}
                </p>
              )}
            </div>

            {sourceMode === 'upload' ? (
              <CustomQuestion
                value={customQuestion}
                onChange={setCustomQuestion}
                onStart={handleStart}
                canStart={canStart}
              />
            ) : (
              <PresetQuestions
                questions={presetQuestions}
                selectedQuestion={selectedQuestion}
                query={query}
                onQueryChange={setQuery}
                onSelect={setSelectedQuestionId}
                onStart={handleStart}
                canStart={canStart}
              />
            )}
          </section>
        )}
      </div>
    </main>
  );
}

function ChunkTemplatePicker({
  value,
  onChange,
}: {
  value: ChunkTemplate;
  onChange: (value: ChunkTemplate) => void;
}) {
  return (
    <div className="mb-5">
      <div className="mb-2 text-sm font-medium text-slate-700">论文领域</div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {CHUNK_TEMPLATE_OPTIONS.map((option) => {
          const Icon = option.icon;
          const selected = value === option.value;
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => onChange(option.value)}
              className={`inline-flex h-11 items-center justify-center gap-2 rounded-lg border px-3 text-sm transition ${
                selected
                  ? 'border-cyan-400 bg-cyan-50 text-cyan-800 shadow-sm'
                  : 'border-slate-200 bg-white text-slate-600 hover:border-cyan-200 hover:text-cyan-700'
              }`}
            >
              <Icon size={16} />
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function DocumentButton({
  pdf,
  selected,
  onSelect,
}: {
  pdf: PdfItem;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={`rounded-lg border p-4 text-left transition ${
        selected
          ? 'border-cyan-400 bg-cyan-50 shadow-sm'
          : 'border-slate-200 bg-white hover:border-cyan-200 hover:bg-slate-50'
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-slate-950 text-cyan-200">
          <FileText size={20} />
        </div>
        <div className="min-w-0">
          <p className="truncate font-medium text-slate-950">{pdf.file_name}</p>
          <p className="mt-1 text-sm text-slate-500">
            {pdf.pages || '-'} 页 / {pdf.question_count} 个预设问题
          </p>
        </div>
      </div>
    </button>
  );
}

function CustomQuestion({
  value,
  onChange,
  onStart,
  canStart,
}: {
  value: string;
  onChange: (value: string) => void;
  onStart: () => void;
  canStart: boolean;
}) {
  return (
    <div>
      <label className="text-sm font-medium text-slate-700">自定义问题</label>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-2 min-h-36 w-full resize-none rounded-lg border border-slate-200 bg-white p-4 leading-7 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-100"
        placeholder="请输入你想问这个 PDF 的问题..."
        autoFocus
      />
      <StartButton onClick={onStart} disabled={!canStart} label="开始分析" />
    </div>
  );
}

function PresetQuestions({
  questions,
  selectedQuestion,
  query,
  onQueryChange,
  onSelect,
  onStart,
  canStart,
}: {
  questions: QuestionItem[];
  selectedQuestion: QuestionItem | null;
  query: string;
  onQueryChange: (value: string) => void;
  onSelect: (questionId: string) => void;
  onStart: () => void;
  canStart: boolean;
}) {
  return (
    <div>
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          className="h-11 w-full rounded-lg border border-slate-200 bg-white pl-10 pr-3 text-sm outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-100"
          placeholder="搜索预设问题"
        />
      </div>

      <div className="max-h-[430px] space-y-3 overflow-y-auto pr-1">
        {questions.map((question) => (
          <button
            key={question.question_id}
            onClick={() => onSelect(question.question_id)}
            className={`w-full rounded-lg border p-4 text-left transition ${
              selectedQuestion?.question_id === question.question_id
                ? 'border-cyan-400 bg-cyan-50 shadow-sm'
                : 'border-slate-200 bg-white hover:border-cyan-200 hover:bg-slate-50'
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-semibold text-slate-950">{question.question_id}</span>
              <span className={`rounded-full border px-2 py-0.5 text-xs ${questionTypeClass(question.question_type)}`}>
                {question.question_type}
              </span>
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-700">{question.question}</p>
          </button>
        ))}
      </div>

      <StartButton onClick={onStart} disabled={!canStart} label="查看证据" />
    </div>
  );
}

function StartButton({ onClick, disabled, label }: { onClick: () => void; disabled: boolean; label: string }) {
  return (
    <div className="mt-6 flex justify-center">
      <button
        onClick={onClick}
        disabled={disabled}
        className={`inline-flex h-11 items-center justify-center gap-2 rounded-lg px-6 text-white shadow-sm transition ${
          disabled ? 'cursor-not-allowed bg-slate-300' : 'bg-slate-950 hover:bg-slate-800'
        }`}
      >
        {label}
        <ChevronRight size={18} />
      </button>
    </div>
  );
}
