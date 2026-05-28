from __future__ import annotations

from pathlib import Path


def count_pdf_pages(pdf_path: Path) -> int:
    """Return a best-effort PDF page count without making parsing mandatory."""
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        pass

    try:
        from PyPDF2 import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        pass

    try:
        import fitz

        with fitz.open(str(pdf_path)) as doc:
            return len(doc)
    except Exception:
        return 0
