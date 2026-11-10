#!/usr/bin/env python3
"""Run the learned OCR detector for exactly one page and save review artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "source" / "pdfs" / "5ee2d592b0fb4656bc69b46fa759c999pre.pdf"
PATTERNS_PATH = PROJECT_ROOT / "outputs" / "pattern_learning" / "learned_patterns.json"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs" / "page_review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("page", type=int, help="Zero-based PDF page index")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--patterns", type=Path, default=PATTERNS_PATH)
    args = parser.parse_args()

    if args.page < 0:
        raise SystemExit("page must be a zero-based non-negative index")
    patterns = args.patterns.resolve()
    if not patterns.exists():
        raise SystemExit(f"learned patterns are missing; run tools/learn_patterns.py first: {patterns}")
    learned = json.loads(patterns.read_text())
    if learned.get("schema_version") not in {1, 2}:
        raise SystemExit(f"unsupported learned pattern schema: {learned.get('schema_version')!r}")

    page_dir = args.artifacts.resolve() / f"page_{args.page:04d}"
    temp_dir = page_dir / "work"
    output_dir = page_dir / "detector_output"
    page_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "full_pdf_ocr_pipeline.py"),
        "--pdf",
        str(args.pdf.resolve()),
        "--pages",
        str(args.page),
        "--workers",
        "1",
        "--dpi",
        str(args.dpi),
        "--output-dir",
        str(output_dir),
        "--temp-dir",
        str(temp_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    source_path = page_dir / f"page_{args.page:04d}_source.png"
    annotated_path = page_dir / f"page_{args.page:04d}_annotated.jpg"
    with fitz.open(args.pdf.resolve()) as document:
        if args.page >= len(document):
            raise SystemExit(f"page {args.page} is outside the PDF ({len(document)} pages)")
        pixmap = document[args.page].get_pixmap(dpi=args.dpi, alpha=False)
        pixmap.save(str(source_path))

    generated_annotated = temp_dir / f"page_{args.page:04d}.jpg"
    if not generated_annotated.exists():
        raise SystemExit(f"annotated image was not generated: {generated_annotated}")
    shutil.copy2(generated_annotated, annotated_path)

    record = temp_dir / f"page_{args.page:04d}.json"
    shutil.copy2(record, page_dir / f"page_{args.page:04d}_detections.json")
    (page_dir / "learned_patterns_used.json").write_text(json.dumps(learned, indent=2) + "\n")
    print(json.dumps({
        "page_index": args.page,
        "artifact_dir": str(page_dir),
        "source_image": str(source_path),
        "annotated_image": str(annotated_path),
        "detections": str(page_dir / f"page_{args.page:04d}_detections.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
