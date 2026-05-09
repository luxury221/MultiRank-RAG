export interface PdfItem {
  doc_id: string;
  file_name: string;
  pages: number;
  question_count: number;
  node_count: number;
  modalities: Record<string, number>;
}

export interface QuestionItem {
  question_id: string;
  doc_id: string;
  question: string;
  answer: string;
  question_type: string;
  gold_node_ids: string[];
  gold_pages: string[];
  gold_modalities: string[];
  evidence_note: string;
  card_url: string;
  num_steps: number;
  quality_status: string;
  quality_issues: string[];
  visual_required: number;
  visual_node_steps: number;
  crop_steps: number;
  existing_crop_steps: number;
  qwen_caption_steps: number;
  source_pages: string[];
}

export interface EvidenceStep {
  chain_step: number;
  role: string;
  node_id: string;
  node_type: string;
  page: number;
  relation: string;
  score: number;
  visual_score: number;
  source_ref: string;
  crop_url: string;
  page_url: string;
  visual_summary: string;
  visual_caption: string;
  reason: string;
  content_preview: string;
}

export interface RankingItem {
  rank: number;
  node_id: string;
  node_type: string;
  page: number;
  score: number;
  sim_score: number;
  bridge_score: number;
  ref_score: number;
  visual_score: number;
  has_visual_crop: number;
  has_visual_caption: number;
  source_ref: string;
  content_preview: string;
  crop_url: string;
}

export interface MetricRow {
  method: string;
  num_questions: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr: number;
  ndcg_at_5: number;
  evidence_hit: number;
  modality_hit: number;
  citation_correct: number;
  visual_required_questions: number;
  visual_grounding_hit: number;
  visual_caption_hit: number;
  evidence_chain_ready: number;
  avg_rerank_time_ms: number;
}

export interface AppData {
  generated_at: string;
  corpus: {
    num_pdfs: number;
    num_questions: number;
    num_chain_steps: number;
    num_cards: number;
    quality_pass: number;
    quality_warn: number;
    quality_fail: number;
  };
  pdfs: PdfItem[];
  questions: QuestionItem[];
  chains: Record<string, EvidenceStep[]>;
  rankings: Record<string, Record<string, RankingItem[]>>;
  metrics: MetricRow[];
}

export type ChunkTemplate = 'auto' | 'general' | 'ai' | 'math' | 'finance' | 'medical';

export type AnalysisRequest =
  | {
      mode: 'system';
      doc_id: string;
      pdf_name: string;
      question_id: string;
    }
  | {
      mode: 'upload';
      pdf_name: string;
      question: string;
      file: File;
      chunk_template: ChunkTemplate;
    };

export interface UploadJobResult {
  question: QuestionItem;
  steps: EvidenceStep[];
  rankings: Record<string, RankingItem[]>;
  chunk_report?: Record<string, string | number>;
}

export interface UploadJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  stage: string;
  progress: number;
  pdf_name: string;
  question: string;
  chunk_template?: ChunkTemplate;
  message?: string;
  logs?: string[];
  error?: string;
  result?: UploadJobResult;
}
