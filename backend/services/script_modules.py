from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from backend.config import SCRIPTS_DIR


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parse_pdf = load_script_module("parse_pdf_script", SCRIPTS_DIR / "01_parse_pdf.py")
build_graph_edges = load_script_module("build_graph_script", SCRIPTS_DIR / "02_build_graph.py")
visual_evidence = load_script_module("visual_evidence_script", SCRIPTS_DIR / "10_build_visual_evidence.py")
chain_builder = load_script_module("chain_builder_script", SCRIPTS_DIR / "09_build_evidence_chains.py")
card_builder = load_script_module("card_builder_script", SCRIPTS_DIR / "11_build_evidence_cards.py")
chunk_reporter = load_script_module("chunk_reporter_script", SCRIPTS_DIR / "14_chunk_quality_report.py")
