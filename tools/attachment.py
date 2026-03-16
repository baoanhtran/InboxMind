"""
tools/attachment.py

Extracts text and structured content from email attachments.

Supported types:
  - PDF         → PyMuPDF (fitz)
  - Excel/CSV   → openpyxl / csv
  - Images      → GPT-4o Vision via LangChain
  - Plain text  → direct decode
"""

from __future__ import annotations

import base64
import csv
import io
import json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_attachment_text(
    raw_bytes: bytes,
    filename: str,
    mime_type: str,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """
    Route attachment bytes to the appropriate extractor.
    Returns extracted text (may be long — callers should truncate if needed).
    """
    mime_lower = mime_type.lower()
    name_lower = filename.lower()

    if "pdf" in mime_lower or name_lower.endswith(".pdf"):
        return _extract_pdf(raw_bytes)

    if any(x in mime_lower for x in ("spreadsheet", "excel", "xlsx", "xls")) or \
       name_lower.endswith((".xlsx", ".xls")):
        return _extract_excel(raw_bytes)

    if name_lower.endswith(".csv"):
        return _extract_csv(raw_bytes)

    if any(x in mime_lower for x in ("image/", "jpeg", "png", "gif", "webp")):
        if llm is not None:
            return _extract_image_vision(raw_bytes, mime_type, llm)
        return "[Image attachment — GPT-4o Vision not available, no LLM provided]"

    if "text" in mime_lower or name_lower.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="replace")

    return f"[Unsupported attachment type: {mime_type} / {filename}]"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pdf(raw_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages: list[str] = []
        for page_num in range(min(doc.page_count, 30)):   # cap at 30 pages
            page = doc[page_num]
            pages.append(f"--- Page {page_num + 1} ---\n{page.get_text()}")
        doc.close()
        return "\n".join(pages)

    except ImportError:
        return "[PyMuPDF not installed — cannot extract PDF. Run: pip install pymupdf]"
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


# ---------------------------------------------------------------------------
# Excel extraction
# ---------------------------------------------------------------------------

def _extract_excel(raw_bytes: bytes) -> str:
    """
    Extract content from .xlsx files using openpyxl.
    Returns a tab-separated text representation of each sheet.
    """
    try:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True)
        output: list[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            output.append(f"=== Sheet: {sheet_name} ===")
            rows: list[str] = []
            for row in ws.iter_rows(max_row=200, values_only=True):   # cap rows
                cells = [str(c) if c is not None else "" for c in row]
                rows.append("\t".join(cells))
            output.extend(rows)

        return "\n".join(output)

    except ImportError:
        return "[openpyxl not installed — cannot extract Excel. Run: pip install openpyxl]"
    except Exception as exc:
        return f"[Excel extraction error: {exc}]"


# ---------------------------------------------------------------------------
# CSV extraction
# ---------------------------------------------------------------------------

def _extract_csv(raw_bytes: bytes) -> str:
    """Decode and return CSV content as plain text."""
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = ["\t".join(row) for row in reader]
        return "\n".join(rows[:500])    # cap at 500 rows
    except Exception as exc:
        return f"[CSV extraction error: {exc}]"


# ---------------------------------------------------------------------------
# Image → GPT-4o Vision
# ---------------------------------------------------------------------------

def _extract_image_vision(raw_bytes: bytes, mime_type: str, llm: ChatOpenAI) -> str:
    """
    Send an image to GPT-4o Vision and ask for a detailed description.
    Uses LangChain's multimodal message format.
    """
    try:
        b64_image = base64.standard_b64encode(raw_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. If it contains text, tables, "
                        "charts, or data, extract and transcribe all of it. "
                        "Focus on information that would be relevant in a business email context."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                        "detail": "high",
                    },
                },
            ]
        )

        response = llm.invoke([message])
        return response.content

    except Exception as exc:
        return f"[GPT-4o Vision error: {exc}]"
