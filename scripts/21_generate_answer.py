from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("21_generate_competition_submission_llm.py")
    spec = importlib.util.spec_from_file_location("generate_answer_impl", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load answer generator: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
