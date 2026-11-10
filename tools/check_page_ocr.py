#!/usr/bin/env python3
"""OCR one PDF page and create a thin-box visual review image."""

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import pymupdf as fitz
except ImportError as exc:
    raise SystemExit("Install PyMuPDF first: python -m pip install PyMuPDF") from exc

try:
    import pytesseract
except ImportError as exc:
    raise SystemExit("Install pytesseract first: python -m pip install pytesseract") from exc


TITLE_RE = re.compile(r"(?<![A-Z0-9])(?:C|AF|CD|CEF|CR|DB|EG|EV|FC|HP|IF|IV|L|PF|SG|TD)(?:[- ]\d+[A-Z]?)?(?![A-Z0-9])")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_title(value: str) -> str:
    value = value.upper().replace(" ", "-")
    value = re.sub(r"--+", "-", value)
    return value


def titles_from_text(text: str) -> set[str]:
    return {normalize_title(match.group(0)) for match in TITLE_RE.finditer(text.upper())}


def load_expected_titles(project_root: Path, page_index: int) -> set[str]:
    source = next((project_root / "source" / "json").glob("mechanical_data_out_*.json"))
    data = json.loads(source.read_text())
    page = data.get(str(page_index), {})
    titles = set()
    for key in ("used_mechanical", "unused_mechanical"):
        titles.update(normalize_title(item) for item in page.get(key, {}).get("texts", []) if item)

    checklist = project_root / "review" / "human_checked.json"
    if checklist.exists():
        review = json.loads(checklist.read_text())
        for record in review.get("records", []):
            if record.get("page") == page_index and record.get("discipline") == "mechanical":
                titles.update(normalize_title(item) for item in record.get("titles_to_check", []))
                titles.update(titles_from_text(record.get("notes", "")))
    return titles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "source" / "pdfs" / "5ee2d592b0fb4656bc69b46fa759c999pre.pdf")
    parser.add_argument("--page-index", type=int, default=143)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "page_143_ocr_labeled.png")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "outputs" / "page_143_ocr_report.json")
    args = parser.parse_args()

    with fitz.open(args.pdf) as document:
        if args.page_index < 0 or args.page_index >= len(document):
            raise SystemExit(f"Page index {args.page_index} is outside 0..{len(document) - 1}")
        page = document[args.page_index]
        scale = args.dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    ocr = pytesseract.image_to_data(image, config="--psm 11", output_type=pytesseract.Output.DICT)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detections = []
    found_titles = set()

    for index, raw_text in enumerate(ocr["text"]):
        text = raw_text.strip().upper()
        titles = titles_from_text(text)
        if not titles:
            continue
        left, top = int(ocr["left"][index]), int(ocr["top"][index])
        width, height = int(ocr["width"][index]), int(ocr["height"][index])
        box = [left, top, left + width, top + height]
        found_titles.update(titles)
        draw.rectangle(box, outline=(220, 30, 30), width=2)
        draw.text((left, max(0, top - 12)), text, fill=(220, 30, 30), font=font)
        detections.append({"text": text, "titles": sorted(titles), "bbox": box, "confidence": ocr["conf"][index]})

    expected_titles = load_expected_titles(PROJECT_ROOT, args.page_index)
    used_bboxes = {
        "bbox": [[item["bbox"] for item in detections]],
        "text": [[item["text"] for item in detections]],
        "shape": [["Text" for _ in detections]],
        "is_fallback_used": [[False for _ in detections]],
    }
    missing_titles = sorted(expected_titles - found_titles)
    unused_bboxes = {
        "bbox": [[]],
        "text": [missing_titles],
        "shape": [["Text" for _ in missing_titles]],
        "is_fallback_used": [[True for _ in missing_titles]],
    }
    empty_keynotes = {"bbox": [], "text": [], "shape": []}
    report = {
        str(args.page_index): {
            "used_bboxes": used_bboxes,
            "unused_bboxes": unused_bboxes,
            "valid_outer_keynotes": empty_keynotes,
            "invalid_outer_keynotes": {"bbox": [], "text": [], "shape": []},
        }
    }
    image.save(args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved labeled image: {args.output}")
    print(f"Saved OCR report: {args.report}")
    print("Expected titles not found by OCR:", ", ".join(missing_titles) or "none")


if __name__ == "__main__":
    main()