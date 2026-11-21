"""PDF text extraction. Keeps page numbers so answers can cite them."""

from pathlib import Path

import pymupdf


class ExtractionError(Exception):
    pass


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that contain text. Page numbers start at 1."""
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # corrupt or non-PDF file
        raise ExtractionError(f"Could not open PDF: {exc}") from exc

    pages: list[tuple[int, str]] = []
    with doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((i, text))

    if not pages:
        raise ExtractionError("No text found. The PDF may be scanned images; run OCR first.")
    return pages
