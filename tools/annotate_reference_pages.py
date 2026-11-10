#!/usr/bin/env python3
"""Run the legacy OCR/annotation pipeline on referenced discipline pages only."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from build_reference_index import build_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "source" / "pdfs" / "5ee2d592b0fb4656bc69b46fa759c999pre.pdf"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "referenced_pages_ocr"
TEMP_DIR = PROJECT_ROOT / ".cache" / "referenced_pages_ocr_work"


def referenced_pages() -> list[int]:
    return sorted(build_index())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, required=True, help="One zero-based referenced PDF page index")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pages = referenced_pages()
    if args.page not in pages:
        raise SystemExit(f"page {args.page} is not referenced by the discipline JSON files")
    if args.workers != 1:
        raise SystemExit("one-page performance mode requires --workers 1")

    selected_page = args.page
    output_dir = OUTPUT_DIR / f"page_{selected_page:04d}"
    temp_dir = TEMP_DIR / f"page_{selected_page:04d}"

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "full_pdf_ocr_pipeline.py"),
        "--pdf", str(PDF_PATH),
        "--pages", str(selected_page),
        "--workers", str(args.workers),
        "--dpi", str(args.dpi),
        "--output-dir", str(output_dir),
        "--temp-dir", str(temp_dir),
    ]
    if args.resume:
        command.append("--resume")

    print(json.dumps({
        "page_index": selected_page,
        "referenced_page_count": len(pages),
        "output_dir": str(output_dir),
    }, indent=2), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    image_dir = output_dir / "annotated_pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    source = temp_dir / f"page_{selected_page:04d}.jpg"
    if source.exists():
        shutil.copy2(source, image_dir / source.name)
        copied += 1
    (output_dir / "referenced_page.json").write_text(json.dumps({
        "source_files": [
            "source/json/mechanical_data_out_5ee2d592b0fb4656bc69b46fa759c999.json",
            "source/json/plumbing_data_out_5ee2d592b0fb4656bc69b46fa759c999.json",
            "source/json/panel_data_out_5ee2d592b0fb4656bc69b46fa759c999.json",
        ],
        "page_indexes": [selected_page],
        "annotated_images": copied,
    }, indent=2) + "\n")
    print(f"Copied {copied}/1 annotated page image to {image_dir}")


if __name__ == "__main__":
    main()
