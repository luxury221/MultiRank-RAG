from __future__ import annotations

from multirank_rag.legacy import load_numbered_script


_legacy = load_numbered_script("01_parse_pdf.py", "multirank_rag_legacy_parse_pdf")

PaperChunkTemplate = _legacy.PaperChunkTemplate
TemplateDecision = _legacy.TemplateDecision
PAPER_CHUNK_TEMPLATES = _legacy.PAPER_CHUNK_TEMPLATES

pdf_to_nodes = _legacy.pdf_to_nodes
mineru_pdf_to_nodes = _legacy.mineru_pdf_to_nodes
select_chunk_template = _legacy.select_chunk_template
chunk_template_decision = _legacy.chunk_template_decision
extract_document_refs = _legacy.extract_document_refs
bbox_to_json = _legacy.bbox_to_json
json_to_bbox = _legacy.json_to_bbox
