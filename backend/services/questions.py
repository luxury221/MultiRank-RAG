from __future__ import annotations

from pathlib import Path

from pipeline_common import clean_text, write_csv


def infer_question_type(question: str) -> str:
    text = question.lower()
    if any(term in text for term in ["table", "表格", "表 ", "表中"]):
        return "表格问答"
    if any(term in text for term in ["figure", "fig.", "图", "图片", "曲线", "趋势"]):
        return "图表理解"
    if any(term in text for term in ["跨模态", "结合", "图文", "多模态"]):
        return "跨模态综合"
    return "自定义问题"


def write_single_question(job: Path, question: dict[str, str]) -> Path:
    path = job / "questions.csv"
    fields = [
        "question_id",
        "doc_id",
        "question",
        "answer",
        "question_type",
        "gold_node_ids",
        "gold_pages",
        "gold_modalities",
        "evidence_note",
    ]
    write_csv(path, [question], fields)
    return path
