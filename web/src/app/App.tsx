import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { SelectionPage } from './components/SelectionPage';
import { EvidencePage } from './components/EvidencePage';
import type { AnalysisRequest, AppData, QuestionItem } from './types';

type Page = 'selection' | 'evidence';

export default function App() {
  const [data, setData] = useState<AppData | null>(null);
  const [error, setError] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<Page>('selection');
  const [analysisRequest, setAnalysisRequest] = useState<AnalysisRequest | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        const response = await fetch('/app-data.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`无法读取 app-data.json：${response.status}`);
        }
        const nextData = (await response.json()) as AppData;
        if (!cancelled) {
          setData(nextData);
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

  const selectedQuestion = useMemo(() => {
    if (!data || !analysisRequest) {
      return null;
    }
    if (analysisRequest.mode === 'upload') {
      return {
        question_id: 'CUSTOM',
        doc_id: analysisRequest.pdf_name,
        question: analysisRequest.question,
        answer: '上传文档尚未生成证据链结果。',
        question_type: '自定义问题',
        gold_node_ids: [],
        gold_pages: [],
        gold_modalities: [],
        evidence_note: '',
        card_url: '',
        num_steps: 0,
        quality_status: 'pending',
        quality_issues: [],
        visual_required: 0,
        visual_node_steps: 0,
        crop_steps: 0,
        existing_crop_steps: 0,
        qwen_caption_steps: 0,
        source_pages: [],
      } satisfies QuestionItem;
    }
    return data.questions.find((question) => question.question_id === analysisRequest.question_id) ?? null;
  }, [analysisRequest, data]);

  const handleStartAnalysis = (request: AnalysisRequest) => {
    setAnalysisRequest(request);
    setCurrentPage('evidence');
  };

  const handleBack = () => {
    setCurrentPage('selection');
  };

  if (error) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-6">
        <div className="max-w-xl rounded-lg border border-red-300/30 bg-red-950/40 p-6 shadow-xl">
          <div className="flex items-center gap-3 text-red-100">
            <AlertTriangle size={24} />
            <h1 className="text-xl">页面数据没有准备好</h1>
          </div>
          <p className="mt-3 text-sm leading-6 text-red-50/80">
            {error}。请先在项目根目录运行
            <code className="mx-1 rounded bg-white/10 px-1.5 py-0.5">python scripts/13_export_frontend_data.py</code>
            生成前端数据。
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
        <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-white/5 px-5 py-4">
          <Loader2 className="animate-spin text-cyan-300" size={22} />
          <span>正在加载多模态证据数据...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#e0f2fe_0,#f8fafc_36%,#eef2ff_100%)] text-slate-950">
      {currentPage === 'selection' || !selectedQuestion ? (
        <SelectionPage data={data} onStartAnalysis={handleStartAnalysis} />
      ) : (
        <EvidencePage data={data} question={selectedQuestion} request={analysisRequest} onBack={handleBack} />
      )}
    </div>
  );
}
