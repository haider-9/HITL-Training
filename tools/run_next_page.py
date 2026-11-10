#!/usr/bin/env python3
"""Process the next page represented in the legacy JSON references.

Each invocation processes one page, writes a Pillow-annotated review image, and
updates cumulative validation-shaped JSON outputs. Run the command again for
the next referenced page.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from build_reference_index import build_index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "source" / "pdfs" / "5ee2d592b0fb4656bc69b46fa759c999pre.pdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "one_page_ocr"
PATTERNS_PATH = PROJECT_ROOT / "outputs" / "pattern_learning" / "learned_patterns.json"
HUMAN_FILE_NAME = "human_verified.json"


def referenced_pages() -> list[int]:
    """Return pages backed by source/json files, excluding review-only pages."""
    source_dir = PROJECT_ROOT / "source" / "json"
    source_names = {path.name for path in source_dir.glob("*.json")}
    index = build_index()
    return sorted(
        page
        for page, bucket in index.items()
        if any(source in source_names for source in bucket.get("sources", []))
    )


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def ensure_human_file(output: Path) -> Path:
    path = output / HUMAN_FILE_NAME
    if not path.exists():
        path.write_text(json.dumps({
            "schema_version": 1,
            "instructions": "After checking the annotated page, edit pages.<page>. Set reviewed=true and fill accepted, rejected, or added.",
            "pages": {},
        }, indent=2) + "\n")
    return path


def normalize_set(values: object) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip().upper() for value in values if str(value).strip()}


def apply_human_review(
    page: int,
    page_record: dict,
    human_page: dict,
) -> tuple[dict, list[dict]]:
    accepted = normalize_set(human_page.get("accepted", human_page.get("accepted_text", [])))
    rejected = normalize_set(human_page.get("rejected", human_page.get("rejected_text", [])))
    added = human_page.get("added", [])
    feedback = []

    for detection in page_record.get("detections", []):
        text = str(detection.get("text", "")).strip().upper()
        if text in accepted:
            detection["human_verified"] = True
            detection["human_decision"] = "accepted"
            detection["confirmed"] = True
            detection["is_uncertain"] = False
            detection["important"] = True
            detection["already_present_in_old_json"] = True
            detection["reason"] = "Accepted during human verification."
        elif text in rejected:
            detection["human_verified"] = True
            detection["human_decision"] = "rejected"
            detection["confirmed"] = False
            detection["important"] = False
            detection["reason"] = "Rejected during human verification."
        feedback.append({
            "text": text,
            "decision": detection.get("human_decision", "unchanged"),
            "category": detection.get("category", "other"),
            "page": page,
        })

    for item in added if isinstance(added, list) else []:
        if not isinstance(item, dict) or not item.get("text") or not isinstance(item.get("bbox"), list):
            continue
        added_detection = {
            "text": str(item["text"]).strip().upper(),
            "normalized_text": str(item["text"]).strip().upper(),
            "bbox": item["bbox"],
            "source": "human",
            "category": item.get("category", "other"),
            "pattern": "human_added",
            "ocr_confidence": 1.0,
            "confirmed": True,
            "is_uncertain": False,
            "important": True,
            "matched_reference": None,
            "already_present_in_old_json": False,
            "visually_detected": True,
            "human_verified": True,
            "human_decision": "added",
            "reason": "Added during human verification.",
            "reference_bbox": None,
            "reference_block": None,
            "reference_file": None,
            "related_old_json_files": [],
            "count_as_text": False,
            "used_as_keynote": False,
        }
        page_record.setdefault("detections", []).append(added_detection)
        feedback.append({
            "text": added_detection["text"],
            "decision": "added",
            "category": added_detection["category"],
            "page": page,
        })

    page_record["human_verified"] = bool(human_page.get("reviewed", False))
    page_record["human_notes"] = human_page.get("notes", "")
    return page_record, feedback


def validation_page_from_record(page_record: dict) -> dict:
    used = [
        item for item in page_record.get("detections", [])
        if item.get("important") and item.get("confirmed") and not item.get("is_uncertain")
    ]
    unused = [
        item for item in page_record.get("detections", [])
        if item.get("human_decision") == "rejected"
    ] + page_record.get("reference_only_items", [])

    def block(items: list[dict]) -> dict:
        return {
            "bbox": [[item.get("bbox") for item in items if isinstance(item.get("bbox"), list)]],
            "text": [[item.get("text", "") for item in items if isinstance(item.get("bbox"), list)]],
            "shape": [[item.get("shape", "Text") for item in items if isinstance(item.get("bbox"), list)]],
            "is_fallback_used": [[bool(item.get("is_fallback_used", False)) for item in items if isinstance(item.get("bbox"), list)]],
        }

    keynotes = [item for item in page_record.get("keynotes", []) if item.get("matched_reference")]
    invalid_keynotes = [item for item in page_record.get("invalid_outer_keynotes", [])]
    return {
        "used_bboxes": block(used),
        "unused_bboxes": block(unused),
        "valid_outer_keynotes": {
            "bbox": [item["bbox"] for item in keynotes],
            "text": [item["text"] for item in keynotes],
            "shape": [item.get("shape", "Circle") for item in keynotes],
        },
        "invalid_outer_keynotes": {
            "bbox": [item["bbox"] for item in invalid_keynotes],
            "text": [item["text"] for item in invalid_keynotes],
            "shape": [item.get("shape", "Circle") for item in invalid_keynotes],
        },
    }


def recolor_human_image(image_path: Path, page_record: dict, dpi: int) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not image_path.exists():
        return
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    scale = dpi / 72.0
    for detection in page_record.get("detections", []):
        if not detection.get("human_verified"):
            continue
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        color = (0, 80, 220) if detection.get("human_decision") in {"accepted", "added"} else (210, 30, 30)
        box = [int(value * scale) for value in bbox]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1] - 12)), detection.get("text", ""), fill=color, font=font)
    image.save(image_path, "JPEG", quality=82, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reset", action="store_true", help="Start again from the first referenced page")
    parser.add_argument("--human", action="store_true", help="Apply human_verified.json to a page and relearn feedback")
    parser.add_argument("--page", type=int, help="Page to review with --human")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    human_path = ensure_human_file(output)
    state_path = output / "progress.json"
    state = read_json(state_path, {"processed_pages": []})
    if not isinstance(state, dict):
        raise SystemExit(f"invalid progress file: {state_path}")
    processed = set(state.get("processed_pages", [])) if not args.reset else set()
    pages = referenced_pages()
    if args.human and args.page is None:
        raise SystemExit("--human requires --page PAGE")
    if args.human and args.page not in pages:
        raise SystemExit(f"page {args.page} is not represented in the legacy source JSONs")
    remaining = [page for page in pages if page not in processed]

    if not args.human and not remaining:
        print(json.dumps({
            "status": "complete",
            "referenced_pages": len(pages),
            "processed_pages": len(processed),
            "output_dir": str(output),
        }, indent=2))
        return

    page = args.page if args.human else remaining[0]
    if not PATTERNS_PATH.exists():
        raise SystemExit(f"learned patterns are missing; run tools/learn_patterns.py first: {PATTERNS_PATH}")
    page_dir = output / f"page_{page:04d}"
    detector_output = page_dir / "detector_output"
    temp_dir = page_dir / "work"
    detector_output.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "full_pdf_ocr_pipeline.py"),
        "--pdf", str(args.pdf.resolve()),
        "--pages", str(page),
        "--workers", "1",
        "--dpi", str(args.dpi),
        "--output-dir", str(detector_output),
        "--temp-dir", str(temp_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    validation = read_json(detector_output / "detected_text_validation.json", {})
    details = read_json(detector_output / "detected_text_details.json", {"pages": []})
    missed = read_json(detector_output / "missed_text.json", {"missed": [], "uncertain": []})
    page_key = str(page)

    if args.human:
        human_data = read_json(human_path, {})
        human_page = human_data.get("pages", {}).get(page_key, {}) if isinstance(human_data, dict) else {}
        if not isinstance(human_page, dict) or not human_page.get("reviewed"):
            raise SystemExit(
                f"Mark pages.{page_key}.reviewed=true in {human_path} after checking the image, then rerun."
            )
        if not isinstance(details, dict) or not details.get("pages"):
            raise SystemExit("one-page detector did not produce page details")
        reviewed_record, feedback = apply_human_review(page, details["pages"][0], human_page)
        details["pages"][0] = reviewed_record
        validation = {page_key: validation_page_from_record(reviewed_record)}
        feedback_path = output / "human_learning.json"
        old_feedback = read_json(feedback_path, {"feedback": []})
        feedback_rows = old_feedback.get("feedback", []) if isinstance(old_feedback, dict) else []
        feedback_rows = [item for item in feedback_rows if item.get("page") != page]
        feedback_rows.extend(feedback)
        feedback_path.write_text(json.dumps({"feedback": feedback_rows}, indent=2) + "\n")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "learn_patterns.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
        recolor_human_image(temp_dir / f"page_{page:04d}.jpg", reviewed_record, args.dpi)

    cumulative_validation = read_json(output / "detected_text_validation.json", {})
    if not isinstance(cumulative_validation, dict):
        cumulative_validation = {}
    if isinstance(validation, dict) and page_key in validation:
        cumulative_validation[page_key] = validation[page_key]
    (output / "detected_text_validation.json").write_text(
        json.dumps(dict(sorted(cumulative_validation.items(), key=lambda item: int(item[0]))), indent=2) + "\n"
    )

    cumulative_details = read_json(output / "detected_text_details.json", {"pages": []})
    if not isinstance(cumulative_details, dict):
        cumulative_details = {"pages": []}
    detail_pages = [item for item in cumulative_details.get("pages", []) if item.get("page_index") != page]
    if isinstance(details, dict):
        detail_pages.extend(details.get("pages", []))
    cumulative_details.update({
        "document": {
            "pdf_path": str(args.pdf.resolve().relative_to(PROJECT_ROOT)),
            "page_index_base": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Machine-generated OCR/PDF-text comparison. Not human-verified.",
        },
        "pages": sorted(detail_pages, key=lambda item: item.get("page_index", -1)),
    })
    (output / "detected_text_details.json").write_text(json.dumps(cumulative_details, indent=2) + "\n")

    cumulative_missed = read_json(output / "missed_text.json", {"missed": [], "uncertain": []})
    if not isinstance(cumulative_missed, dict):
        cumulative_missed = {"missed": [], "uncertain": []}
    for key in ("missed", "uncertain"):
        old_items = [item for item in cumulative_missed.get(key, []) if item.get("page_index") != page]
        if isinstance(missed, dict):
            old_items.extend(missed.get(key, []))
        cumulative_missed[key] = old_items
    cumulative_missed["summary"] = {
        "confirmed_misses": len(cumulative_missed["missed"]),
        "uncertain_detections": len(cumulative_missed["uncertain"]),
        "pages_with_misses": len({item.get("page_index") for item in cumulative_missed["missed"]}),
        "pages_with_uncertain": len({item.get("page_index") for item in cumulative_missed["uncertain"]}),
    }
    (output / "missed_text.json").write_text(json.dumps(cumulative_missed, indent=2) + "\n")

    human_data = read_json(human_path, {"schema_version": 1, "pages": {}})
    if not isinstance(human_data, dict):
        human_data = {"schema_version": 1, "pages": {}}
    human_data.setdefault("schema_version", 1)
    human_data.setdefault("pages", {})
    human_data["pages"].setdefault(page_key, {
        "reviewed": False,
        "accepted": [],
        "rejected": [],
        "added": [],
        "notes": "",
    })
    human_path.write_text(json.dumps(human_data, indent=2) + "\n")

    processed.add(page)
    state = {
        "referenced_pages": pages,
        "processed_pages": sorted(processed),
        "remaining_pages": [item for item in pages if item not in processed],
        "last_processed_page": page,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({
        "status": "processed",
        "page_index": page,
        "remaining_pages": len(state["remaining_pages"]),
        "validation_json": str(output / "detected_text_validation.json"),
        "annotated_image": str(temp_dir / f"page_{page:04d}.jpg"),
        "page_details": str(detector_output / "detected_text_details.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
