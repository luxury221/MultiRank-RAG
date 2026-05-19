from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from multirank_rag.common import ROOT, SCRIPTS_DIR


def ensure_scripts_path() -> None:
    root = str(ROOT)
    scripts = str(SCRIPTS_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def import_legacy_module(module_name: str) -> ModuleType:
    ensure_scripts_path()
    return importlib.import_module(module_name)


def load_numbered_script(filename: str, module_name: str | None = None) -> ModuleType:
    ensure_scripts_path()
    path = SCRIPTS_DIR / filename
    if module_name is None:
        module_name = f"multirank_rag_legacy_{Path(filename).stem.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
