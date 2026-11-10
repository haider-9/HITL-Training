#!/usr/bin/env python3
"""Validate the generated full-document OCR outputs.

Checks:
  * validation JSON page coverage and exact schema/alignment
  * details report page coverage and required fields
  * missed report structure and page indexes
  * annotated PDF page count, order and dimensions against the source PDF

This script is read-only and does not modify any output.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf as fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_pdf_ocr"
PDF_PATH = PROJECT_ROOT / "source" / "pdfs" / "5ee2d592b0fb4656bc69b46fa759c999pre.pdf"

REQUIRED_PAGE_KEYS = {"used_bboxes", "unused_bboxes", "valid_outer_keynotes", "invalid_outer_keynotes"}
REQUIRED_BBOX_BLOCK_KEYS = {"bbox", "text", "shape", "is_fallback_used"}
REQUIRED_KEYNOTE_BLOCK_KEYS = {"bbox", "text", "shape"}
REQUIRED_DETAIL_FIELDS = {
    "text",
    "normalized_text",
    "ocr_confidence",
    "bbox",
    "source",
    "category",
    "matched_reference",
    "already_present_in_old_json",
    "visually_detected",
    "reason",
}


def is_bbox(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for name in (
        "detected_text_validation.json",
        "detected_text_details.json",
        "missed_text.json",
        "annotated_full_document.pdf",
    ):
        if not (OUTPUT_DIR / name).exists():
            errors.append(f"missing output: {name}")
    if errors:
        print("\n".join(errors))
        return 1

    validation = json.loads((OUTPUT_DIR / "detected_text_validation.json").read_text())
    details = json.loads((OUTPUT_DIR / "detected_text_details.json").read_text())
    missed_doc = json.loads((OUTPUT_DIR / "missed_text.json").read_text())

    with fitz.open(PDF_PATH) as source:
        source_pages = [(float(p.rect.width), float(p.rect.height)) for p in source]
    with fitz.open(OUTPUT_DIR / "annotated_full_document.pdf") as annotated:
        annotated_pages = [(float(p.rect.width), float(p.rect.height)) for p in annotated]

    expected_pages = {str(i) for i in range(len(source_pages))}
    if set(validation) != expected_pages:
        errors.append(
            f"validation page keys do not match 0..{len(source_pages) - 1} "
            f"(missing={sorted(expected_pages - set(validation))[:5]}, extra={sorted(set(validation) - expected_pages)[:5]})"
        )
    detail_pages = [p.get("page_index") for p in details.get("pages", [])]
    if detail_pages != list(range(len(source_pages))):
        errors.append("details pages are not exactly 0..page_count-1 in order")

    used_total = unused_total = valid_key_total = invalid_key_total = 0
    pages_with_used = pages_with_unused = 0

    for page_key in sorted(validation, key=lambda k: int(k)):
        page = validation[page_key]
        if set(page) != REQUIRED_PAGE_KEYS:
            errors.append(f"page {page_key}: wrong page-level keys")
            continue
        for block_name in ("used_bboxes", "unused_bboxes"):
            block = page[block_name]
            if set(block) != REQUIRED_BBOX_BLOCK_KEYS:
                errors.append(f"page {page_key}: {block_name} has wrong keys")
                continue
            groups = [block["bbox"], block["text"], block["shape"], block["is_fallback_used"]]
            if len({len(g) for g in groups}) != 1:
                errors.append(f"page {page_key}: {block_name} group counts are not aligned")
                continue
            for gi in range(len(groups[0])):
                if len({len(g[gi]) for g in groups}) != 1:
                    errors.append(f"page {page_key}: {block_name}[{gi}] arrays are not aligned")
                for bbox in groups[0][gi]:
                    if not is_bbox(bbox):
                        errors.append(f"page {page_key}: {block_name} bad bbox {bbox!r}")
            if block_name == "used_bboxes":
                n = sum(len(g) for g in block["text"])
                used_total += n
                pages_with_used += 1 if n else 0
            else:
                n = sum(len(g) for g in block["text"])
                unused_total += n
                pages_with_unused += 1 if n else 0
        for block_name in ("valid_outer_keynotes", "invalid_outer_keynotes"):
            block = page[block_name]
            if set(block) != REQUIRED_KEYNOTE_BLOCK_KEYS:
                errors.append(f"page {page_key}: {block_name} has wrong keys")
                continue
            if not (len(block["bbox"]) == len(block["text"]) == len(block["shape"])):
                errors.append(f"page {page_key}: {block_name} arrays are not aligned")
            for bbox in block["bbox"]:
                if not is_bbox(bbox):
                    errors.append(f"page {page_key}: {block_name} bad bbox {bbox!r}")
            if block_name == "valid_outer_keynotes":
                valid_key_total += len(block["text"])
            else:
                invalid_key_total += len(block["text"])

    for page in details.get("pages", []):
        pi = page.get("page_index")
        if page.get("processing_error"):
            errors.append(f"details page {pi}: processing error {page['processing_error']}")
        for det in page.get("detections", []):
            missing = REQUIRED_DETAIL_FIELDS - set(det)
            if missing:
                errors.append(f"details page {pi}: missing fields {sorted(missing)}")
                break
            if not is_bbox(det.get("bbox")):
                errors.append(f"details page {pi}: bad bbox {det.get('bbox')!r}")
        for item in page.get("reference_only_items", []):
            missing = REQUIRED_DETAIL_FIELDS - set(item)
            if missing:
                errors.append(f"details page {pi}: reference-only item missing fields {sorted(missing)}")
                break
            if not is_bbox(item.get("bbox")):
                errors.append(f"details page {pi}: reference-only bad bbox {item.get('bbox')!r}")

    for item in missed_doc.get("missed", []) + missed_doc.get("uncertain", []):
        pi = item.get("page_index")
        if not isinstance(pi, int) or not (0 <= pi < len(source_pages)):
            errors.append(f"missed/uncertain item has invalid page index {pi!r}")
        if not is_bbox(item.get("bbox")):
            errors.append(f"missed/uncertain item on page {pi} has bad bbox")

    if len(annotated_pages) != len(source_pages):
        errors.append(
            f"annotated PDF has {len(annotated_pages)} pages, source has {len(source_pages)}"
        )
    else:
        for i, (src, ann) in enumerate(zip(source_pages, annotated_pages)):
            if abs(src[0] - ann[0]) > 0.5 or abs(src[1] - ann[1]) > 0.5:
                errors.append(
                    f"annotated page {i} size {ann} does not match source size {src}"
                )

    missed_summary = missed_doc.get("summary", {})
    confirmed_misses = len(missed_doc.get("missed", []))
    uncertain = len(missed_doc.get("uncertain", []))
    if missed_summary.get("confirmed_misses") != confirmed_misses:
        errors.append("missed summary count does not match missed list length")
    if missed_summary.get("uncertain_detections") != uncertain:
        errors.append("uncertain summary count does not match uncertain list length")

    category_counts = Counter(
        det.get("category")
        for page in details.get("pages", [])
        for det in page.get("detections", [])
    )
    source_counts = Counter(
        det.get("source")
        for page in details.get("pages", [])
        for det in page.get("detections", [])
    )

    print("=== OUTPUT VERIFICATION ===")
    print(f"source pages:                 {len(source_pages)}")
    print(f"validation pages:             {len(validation)}")
    print(f"details pages:                {len(details.get('pages', []))}")
    print(f"annotated PDF pages:          {len(annotated_pages)}")
    print(f"validation used_bboxes:       {used_total} (on {pages_with_used} pages)")
    print(f"validation unused_bboxes:     {unused_total} (on {pages_with_unused} pages)")
    print(f"valid_outer_keynotes:         {valid_key_total}")
    print(f"invalid_outer_keynotes:       {invalid_key_total}")
    print(f"confirmed misses:             {confirmed_misses} (on {missed_summary.get('pages_with_misses')} pages)")
    print(f"uncertain detections:         {uncertain} (on {missed_summary.get('pages_with_uncertain')} pages)")
    print(f"detail detections by category: {dict(sorted(category_counts.items()))}")
    print(f"detail detections by source:   {dict(sorted(source_counts.items()))}")
    print()
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for warning in warnings[:20]:
            print(f"  - {warning}")
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for error in errors[:50]:
            print(f"  - {error}")
        return 1
    print("OK: all structural checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
