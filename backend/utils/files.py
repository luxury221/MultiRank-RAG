from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


def safe_filename(filename: str) -> str:
    name = Path(filename or "upload.pdf").name
    stem = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", Path(name).stem).strip("_") or "upload"
    return f"{stem}.pdf"


def write_csv_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
