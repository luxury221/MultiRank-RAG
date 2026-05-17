from __future__ import annotations

import argparse
import csv
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from pipeline_common import ensure_project_dirs, resolve_path


RUN_NOTES = {
    "openragbench_100_embedding_fast_20260517": (
        "基础消融 V0-V4",
        "原始 embedding 召回下的 V0-V4 消融；V4 作为早期完整 MultiRank-RAG 基线。",
    ),
    "openragbench_100_A_context_20260517": (
        "A：上下文扩展",
        "加入同页/同节/同上下文的图表文本候选扩展。",
    ),
    "openragbench_100_AB_context_adaptive_20260517": (
        "AB：上下文扩展 + 自适应重排",
        "在 A 基础上加入按问题类型调整的多模态/表格重排权重。",
    ),
    "openragbench_100_ABE_context_adaptive_graph_20260517": (
        "ABE-v1",
        "早期 GraphRAG/context graph 增强版本，后续发现图边过强。",
    ),
    "openragbench_100_ABE_context_adaptive_graph_v2_20260517": (
        "ABE-v2",
        "GraphRAG 权重调试版本，仍有表格/视觉噪声。",
    ),
    "openragbench_100_ABE_context_adaptive_graph_v3_20260517": (
        "ABE-v3",
        "稳定 GraphRAG 版本；在 A/AB 基础上保留图结构收益。",
    ),
    "openragbench_100_ABECD_table_vision_20260517": (
        "ABE+CD",
        "加入 C 表格结构化与 D Doubao 视觉摘要/OCR/qa_evidence。",
    ),
    "openragbench_100_ABECD_table_vision_guard_20260517": (
        "ABE+CD+Guard 复用重排",
        "复用 ABE+CD 重排结果，只单独验证 evidence guard 对证据链的影响。",
    ),
    "openragbench_100_ABECD_guard_20260517": (
        "ABE+CD+Guard 主线",
        "当前最推荐主线：CD 后加入证据链 guard，平衡检索和证据链质量。",
    ),
    "openragbench_100_multiroute_ABECD_guard_20260517": (
        "多路召回 balanced",
        "多路召回第一版：embedding/BM25/lexical/visual/table/kg/reference/section RRF 融合。覆盖和 nDCG 提升，但 Top1/MRR 略降。",
    ),
    "openragbench_100_multiroute_tuned_ABECD_guard_20260517": (
        "多路召回 precision",
        "多路召回 precision 调权版；Recall@10 更高，但 MRR 与证据链下降，不建议作为默认主线。",
    ),
    "openragbench_100_self_correct_ABECD_multiroute_20260517": (
        "自我修正 replace-v1",
        "基于主线与多路召回结果做二次验证；仅替换整套证据时证据链分数提升，但检索稳定性下降。",
    ),
    "openragbench_100_self_correct_merge_ABECD_multiroute_20260517": (
        "自我修正 merge-v2",
        "当前最佳主线：优先保留主线排序，只在证据缺模态或低置信时合并/替换多路召回证据。",
    ),
    "openragbench_100_v0v4_20260517": (
        "早期 V0-V4 小跑",
        "早期对照实验，可能不完整，仅作历史参考。",
    ),
}

RUN_ORDER = [
    "openragbench_100_embedding_fast_20260517",
    "openragbench_100_A_context_20260517",
    "openragbench_100_AB_context_adaptive_20260517",
    "openragbench_100_ABE_context_adaptive_graph_v3_20260517",
    "openragbench_100_ABECD_table_vision_20260517",
    "openragbench_100_ABECD_guard_20260517",
    "openragbench_100_multiroute_ABECD_guard_20260517",
    "openragbench_100_multiroute_tuned_ABECD_guard_20260517",
    "openragbench_100_self_correct_ABECD_multiroute_20260517",
    "openragbench_100_self_correct_merge_ABECD_multiroute_20260517",
]

RETRIEVAL_FIELDS = [
    "method",
    "num_questions",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "evidence_hit",
    "modality_hit",
    "citation_correct",
    "visual_required_questions",
    "visual_grounding_hit",
    "visual_caption_hit",
    "evidence_chain_ready",
    "avg_rerank_time_ms",
]

CHAIN_FIELDS = [
    "chain_present",
    "avg_step_count",
    "gold_node_coverage",
    "gold_page_hit",
    "gold_modality_coverage",
    "visual_grounding_hit",
    "cross_modal_hit",
    "relation_support",
    "evidence_chain_score",
]

GROUP_FIELDS = [
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_5",
    "visual_grounding_hit",
    "visual_caption_hit",
    "evidence_chain_ready",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_number(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else value


def avg(rows: list[dict[str, str]], field: str) -> float | str:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(field) or 0.0))
        except (TypeError, ValueError):
            pass
    return round(sum(values) / len(values), 6) if values else ""


def run_label(run_name: str) -> str:
    return RUN_NOTES.get(run_name, (run_name, ""))[0]


def run_note(run_name: str) -> str:
    return RUN_NOTES.get(run_name, ("", ""))[1]


def recommended_role(run_name: str) -> str:
    if run_name == "openragbench_100_ABECD_guard_20260517":
        return "当前主线/推荐默认"
    if run_name == "openragbench_100_multiroute_ABECD_guard_20260517":
        return "高召回扩展/自我修正候选"
    if run_name == "openragbench_100_multiroute_tuned_ABECD_guard_20260517":
        return "不推荐默认，仅作调参参考"
    return ""


def priority(run_name: str, variant: str) -> int:
    base = RUN_ORDER.index(run_name) if run_name in RUN_ORDER else 100
    try:
        variant_num = int(variant.replace("V", ""))
    except ValueError:
        variant_num = 9
    return base * 10 + variant_num


def collect_rows(exp_root: Path) -> dict[str, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    chain_route_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []

    for variant_dir in sorted(exp_root.glob("*/V*")):
        if not variant_dir.is_dir():
            continue
        run_name = variant_dir.parent.name
        variant = variant_dir.name

        summary_path = variant_dir / "metrics/summary_metrics.csv"
        for row in read_csv(summary_path):
            out: dict[str, Any] = {
                "experiment_label": run_label(run_name),
                "run_name": run_name,
                "variant": variant,
                "recommended_role": recommended_role(run_name),
                "note": run_note(run_name),
                "variant_dir": str(variant_dir),
            }
            for field in RETRIEVAL_FIELDS:
                out[field] = to_number(row.get(field, ""))
            summary_rows.append(out)

        chain_path = variant_dir / "evidence_chains/chain_eval_summary.csv"
        for row in read_csv(chain_path):
            out = {
                "experiment_label": run_label(run_name),
                "run_name": run_name,
                "variant": variant,
                "route": row.get("route", ""),
                "num_questions": to_number(row.get("num_questions", "")),
                "note": run_note(run_name),
            }
            for field in CHAIN_FIELDS:
                out[field] = to_number(row.get(field, ""))
            chain_route_rows.append(out)
            if row.get("route") == "all":
                chain_rows.append(dict(out))

        per_question_rows = read_csv(variant_dir / "metrics/per_question_metrics.csv")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in per_question_rows:
            grouped[row.get("question_type") or "unknown"].append(row)
        for question_type, rows in sorted(grouped.items()):
            out = {
                "experiment_label": run_label(run_name),
                "run_name": run_name,
                "variant": variant,
                "question_type": question_type,
                "num_questions": len(rows),
            }
            for field in GROUP_FIELDS:
                out[field] = avg(rows, field)
            group_rows.append(out)

        candidate_rows = read_csv(variant_dir / "candidates.csv")
        if candidate_rows and any(row.get("source_routes") for row in candidate_rows):
            route_counts: Counter[str] = Counter()
            top10_counts: Counter[str] = Counter()
            combo_counts: Counter[str] = Counter()
            for row in candidate_rows:
                routes = [item for item in (row.get("source_routes") or "").split("|") if item]
                combo_counts["|".join(routes) if routes else "(context/none)"] += 1
                try:
                    rank = int(float(row.get("rank") or 999))
                except ValueError:
                    rank = 999
                for route in routes or ["(context/none)"]:
                    route_counts[route] += 1
                    if rank <= 10:
                        top10_counts[route] += 1
            for route, count in route_counts.most_common():
                route_rows.append(
                    {
                        "experiment_label": run_label(run_name),
                        "run_name": run_name,
                        "variant": variant,
                        "stat_type": "route_count",
                        "name": route,
                        "count": count,
                        "top10_count": top10_counts.get(route, 0),
                        "share": round(count / max(1, len(candidate_rows)), 6),
                    }
                )
            for combo, count in combo_counts.most_common(20):
                route_rows.append(
                    {
                        "experiment_label": run_label(run_name),
                        "run_name": run_name,
                        "variant": variant,
                        "stat_type": "route_combo_top20",
                        "name": combo,
                        "count": count,
                        "top10_count": "",
                        "share": round(count / max(1, len(candidate_rows)), 6),
                    }
                )

    notes_rows = [
        {"experiment_label": label, "run_name": run_name, "description": note}
        for run_name, (label, note) in RUN_NOTES.items()
    ]

    chain_by_key = {(row["run_name"], row["variant"]): row for row in chain_rows}
    overview_rows: list[dict[str, Any]] = []
    for row in sorted(summary_rows, key=lambda item: priority(item["run_name"], item["variant"])):
        chain = chain_by_key.get((row["run_name"], row["variant"]), {})
        overview = {
            "experiment_label": row["experiment_label"],
            "run_name": row["run_name"],
            "variant": row["variant"],
            "recommended_role": row.get("recommended_role", ""),
            "recall_at_1": row.get("recall_at_1", ""),
            "recall_at_3": row.get("recall_at_3", ""),
            "recall_at_5": row.get("recall_at_5", ""),
            "recall_at_10": row.get("recall_at_10", ""),
            "mrr": row.get("mrr", ""),
            "ndcg_at_5": row.get("ndcg_at_5", ""),
            "visual_grounding_hit": row.get("visual_grounding_hit", ""),
            "evidence_chain_ready": row.get("evidence_chain_ready", ""),
            "chain_score": chain.get("evidence_chain_score", ""),
            "chain_gold_node_coverage": chain.get("gold_node_coverage", ""),
            "chain_gold_page_hit": chain.get("gold_page_hit", ""),
            "chain_gold_modality_coverage": chain.get("gold_modality_coverage", ""),
            "chain_cross_modal_hit": chain.get("cross_modal_hit", ""),
            "note": row.get("note", ""),
        }
        overview_rows.append(overview)

    main = next(
        (
            row
            for row in overview_rows
            if row["run_name"] == "openragbench_100_ABECD_guard_20260517" and row["variant"] == "V4"
        ),
        None,
    )
    if main:
        for row in overview_rows:
            for field in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_5", "chain_score"]:
                try:
                    row[f"delta_{field}_vs_main"] = round(float(row.get(field) or 0) - float(main.get(field) or 0), 6)
                except (TypeError, ValueError):
                    row[f"delta_{field}_vs_main"] = ""

    return {
        "Overview": overview_rows,
        "RetrievalMetrics": sorted(summary_rows, key=lambda item: priority(item["run_name"], item["variant"])),
        "EvidenceChainAll": sorted(chain_rows, key=lambda item: priority(item["run_name"], item["variant"])),
        "EvidenceChainByRoute": sorted(
            chain_route_rows,
            key=lambda item: (priority(item["run_name"], item["variant"]), item.get("route", "")),
        ),
        "ByQuestionType": sorted(
            group_rows,
            key=lambda item: (priority(item["run_name"], item["variant"]), item.get("question_type", "")),
        ),
        "RouteStats": sorted(
            route_rows,
            key=lambda item: (priority(item["run_name"], item["variant"]), item.get("stat_type", ""), -int(item.get("count") or 0)),
        ),
        "RunNotes": notes_rows,
    }


def headers_for(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    headers: list[str] = []
    for header in preferred or []:
        if any(header in row for row in rows) and header not in headers:
            headers.append(header)
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    return headers


def col_name(number: int) -> str:
    text = ""
    while number:
        number, rem = divmod(number - 1, 26)
        text = chr(65 + rem) + text
    return text


def cell_xml(cell_ref: str, value: Any) -> str:
    if isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value or ""))}</t></is></c>'


def sheet_xml(rows: list[dict[str, Any]], headers: list[str]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ',
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" ',
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        "<cols>",
    ]
    for index in range(1, len(headers) + 1):
        width = 28 if index <= 4 else 16
        parts.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
    parts.append("</cols><sheetData>")
    all_rows = [dict(zip(headers, headers))] + rows
    for row_index, row in enumerate(all_rows, start=1):
        parts.append(f'<row r="{row_index}">')
        for col_index, header in enumerate(headers, start=1):
            parts.append(cell_xml(f"{col_name(col_index)}{row_index}", row.get(header, "")))
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def write_xlsx(path: Path, sheets: dict[str, tuple[list[dict[str, Any]], list[str]]]) -> None:
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ',
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ',
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" ',
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>',
    ]
    rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index, sheet_name in enumerate(sheets, start=1):
        workbook.append(f'<sheet name="{escape(sheet_name[:31])}" sheetId="{index}" r:id="rId{index}"/>')
        rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook.append("</sheets></workbook>")
    rels.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    rels.append("</Relationships>")

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "".join(content_types))
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", "".join(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", "".join(rels))
        archive.writestr("xl/styles.xml", styles)
        for index, (_, (rows, headers)) in enumerate(sheets.items(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", sheet_xml(rows, headers))


def write_csv_overview(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment metrics to an Excel-compatible .xlsx file.")
    parser.add_argument("--experiment-root", default="outputs/experiments/open_ragbench_100")
    parser.add_argument("--output", default="outputs/reports/openragbench_experiment_summary_20260517.xlsx")
    parser.add_argument("--overview-csv", default="outputs/reports/openragbench_experiment_overview_20260517.csv")
    args = parser.parse_args()

    ensure_project_dirs()
    exp_root = resolve_path(args.experiment_root)
    rows_by_sheet = collect_rows(exp_root)
    preferred_overview = [
        "experiment_label",
        "variant",
        "recommended_role",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "visual_grounding_hit",
        "evidence_chain_ready",
        "chain_score",
        "delta_mrr_vs_main",
        "delta_ndcg_at_5_vs_main",
        "delta_chain_score_vs_main",
        "note",
    ]
    sheets: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
    for sheet_name, rows in rows_by_sheet.items():
        preferred = preferred_overview if sheet_name == "Overview" else None
        sheets[sheet_name] = (rows, headers_for(rows, preferred))

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(output_path, sheets)
    write_csv_overview(resolve_path(args.overview_csv), rows_by_sheet["Overview"])
    print(f"Wrote Excel report: {output_path}")
    print(f"Wrote overview CSV: {resolve_path(args.overview_csv)}")
    for sheet_name, rows in rows_by_sheet.items():
        print(f"{sheet_name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
