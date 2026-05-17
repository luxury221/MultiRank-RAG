from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("23_build_graphrag.py")
    spec = importlib.util.spec_from_file_location("build_kg_impl", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load KG builder: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
