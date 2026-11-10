#!/usr/bin/env python3
"""Build a per-page reference index from the existing JSON files.

The legacy discipline JSONs mix source pages and target pages.  For the
``verified_*`` blocks the ``target_pages`` array tells us which page each
value/bbox pair actually belongs to.  ``used_*`` / ``unused_*`` blocks are
already page-local.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = PROJECT_ROOT / "source" / "json"

DISCIPLINE_FILES = {
    "mechanical": next(JSON_DIR.glob("mechanical_data_out_*.json")),
    "plumbing": next(JSON_DIR.glob("plumbing_data_out_*.json")),
    "panel": next(JSON_DIR.glob("panel_data_out_*.json")),
}
VALIDATION_FILE = next(JSON_DIR.glob("validation_dict_*.json"))


def norm_text(value: str) -> str:
    value = value.upper().strip()
    value = value.replace("\u2014", "-").replace("\u2013", "-").replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip(".,;:()[]{}<>\"'")


def match_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _pair(texts: Any, bboxes: Any, targets: Any, page_idx: int) -> list[dict]:
    out: list[dict] = []
    if not isinstance(texts, list):
        return out
    for i, raw in enumerate(texts):
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        bbox = None
        if isinstance(bboxes, list) and i < len(bboxes):
            candidate = bboxes[i]
            if (
                isinstance(candidate, (list, tuple))
                and len(candidate) == 4
                and all(isinstance(v, (int, float)) for v in candidate)
            ):
                bbox = [float(v) for v in candidate]
        target = page_idx
        if isinstance(targets, list) and i < len(targets):
            candidate = targets[i]
            if isinstance(candidate, int):
                target = candidate
            elif isinstance(candidate, str) and candidate.isdigit():
                target = int(candidate)
        out.append(
            {
                "text": text,
                "normalized": norm_text(text),
                "match_key": match_key(text),
                "bbox": bbox,
                "page": target,
            }
        )
    return out


def build_index() -> dict[int, dict]:
    pages: dict[int, dict] = {}

    def page_bucket(page: int) -> dict:
        return pages.setdefault(
            page,
            {
                "expected": [],
                "keynotes_expected": [],
                "invalid_keynotes_expected": [],
                "plan_areas": [],
                "sources": set(),
            },
        )

    for discipline, path in DISCIPLINE_FILES.items():
        data = json.loads(path.read_text())
        for page_key, page_value in data.items():
            if not isinstance(page_value, dict):
                continue
            try:
                page_idx = int(page_key)
            except ValueError:
                continue
            bucket = page_bucket(page_idx)
            bucket["sources"].add(path.name)
            for key, value in page_value.items():
                if key == "plan_area_bboxes":
                    if isinstance(value, list):
                        for bbox in value:
                            if (
                                isinstance(bbox, (list, tuple))
                                and len(bbox) == 4
                                and all(isinstance(v, (int, float)) for v in bbox)
                            ):
                                bucket["plan_areas"].append([float(v) for v in bbox])
                    continue
                if not isinstance(value, dict):
                    continue
                texts = value.get("texts")
                if texts is None:
                    texts = value.get("value")
                bboxes = value.get("bboxes")
                if bboxes is None:
                    bboxes = value.get("bbox")
                targets = value.get("target_pages")
                items = _pair(texts, bboxes, targets, page_idx)
                for item in items:
                    item["discipline"] = discipline
                    item["reference_block"] = key
                    item["reference_file"] = path.name
                    page_bucket(item["page"])["expected"].append(item)
                    page_bucket(item["page"])["sources"].add(path.name)

    # Validation dictionary: page-local used/unused/keynotes.
    #
    # The legacy validation dictionary is not fully consistent: some pages list
    # non-numeric tags inside ``valid_outer_keynotes`` and numeric keynotes
    # inside ``unused_bboxes``.  Route an item by its actual text/shape rather
    # than by the block name it happens to live in.
    if VALIDATION_FILE.exists():
        data = json.loads(VALIDATION_FILE.read_text())
        for page_key, page_value in data.items():
            if not isinstance(page_value, dict):
                continue
            try:
                page_idx = int(page_key)
            except ValueError:
                continue
            bucket = page_bucket(page_idx)
            bucket["sources"].add(VALIDATION_FILE.name)

            def add_item(text: str, bbox: Any, shape: Any, block_name: str) -> None:
                text = text.strip()
                if not text:
                    return
                bbox_list = None
                if (
                    isinstance(bbox, (list, tuple))
                    and len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)
                ):
                    bbox_list = [float(v) for v in bbox]
                shape_name = shape if isinstance(shape, str) and shape else "Circle"
                item = {
                    "text": text,
                    "normalized": norm_text(text),
                    "match_key": match_key(text),
                    "bbox": bbox_list,
                    "shape": shape_name,
                    "page": page_idx,
                    "discipline": "validation",
                    "reference_file": VALIDATION_FILE.name,
                    "reference_block": block_name,
                }
                numeric = bool(re.fullmatch(r"\d{1,3}", text))
                if block_name in {"used_bboxes", "unused_bboxes"}:
                    # These blocks are text-level validation entries.  Numeric
                    # keynotes that appear here are kept as unused/used text so
                    # the output mirrors the legacy layout instead of being
                    # duplicated as invalid keynotes.
                    bucket["expected"].append(item)
                elif block_name == "valid_outer_keynotes":
                    if numeric:
                        bucket["keynotes_expected"].append(item)
                    else:
                        bucket["expected"].append(item)
                elif block_name == "invalid_outer_keynotes":
                    if numeric:
                        bucket["invalid_keynotes_expected"].append(item)
                    else:
                        bucket["expected"].append(item)
                else:
                    bucket["expected"].append(item)

            for block_name in ("used_bboxes", "unused_bboxes"):
                block = page_value.get(block_name, {})
                if not isinstance(block, dict):
                    continue
                texts_groups = block.get("text", [])
                bbox_groups = block.get("bbox", [])
                shape_groups = block.get("shape", [])
                if not isinstance(texts_groups, list):
                    continue
                for gi, texts in enumerate(texts_groups):
                    if not isinstance(texts, list):
                        continue
                    bboxes = bbox_groups[gi] if isinstance(bbox_groups, list) and gi < len(bbox_groups) else []
                    shapes = shape_groups[gi] if isinstance(shape_groups, list) and gi < len(shape_groups) else []
                    for i, raw in enumerate(texts):
                        if not isinstance(raw, str):
                            continue
                        bbox = bboxes[i] if isinstance(bboxes, list) and i < len(bboxes) else None
                        shape = shapes[i] if isinstance(shapes, list) and i < len(shapes) else "Circle"
                        add_item(raw, bbox, shape, block_name)
            for block_name in ("valid_outer_keynotes", "invalid_outer_keynotes"):
                block = page_value.get(block_name, {})
                if not isinstance(block, dict):
                    continue
                texts = block.get("text", [])
                bboxes = block.get("bbox", [])
                shapes = block.get("shape", [])
                for i, raw in enumerate(texts if isinstance(texts, list) else []):
                    if not isinstance(raw, str):
                        continue
                    bbox = bboxes[i] if isinstance(bboxes, list) and i < len(bboxes) else None
                    shape = shapes[i] if isinstance(shapes, list) and i < len(shapes) else "Circle"
                    add_item(raw, bbox, shape, block_name)

    # Human review notes are additional expected titles, never authoritative bboxes.
    review_file = PROJECT_ROOT / "review" / "human_checked.json"
    if review_file.exists():
        review = json.loads(review_file.read_text())
        for record in review.get("records", []):
            page_idx = record.get("page")
            if not isinstance(page_idx, int):
                continue
            bucket = page_bucket(page_idx)
            bucket["sources"].add(review_file.name)
            titles = record.get("titles_to_check") or []
            for title in titles:
                if not isinstance(title, str) or not title.strip():
                    continue
                bucket["expected"].append(
                    {
                        "text": title.strip(),
                        "normalized": norm_text(title),
                        "match_key": match_key(title),
                        "bbox": None,
                        "page": page_idx,
                        "discipline": record.get("discipline", "review"),
                        "reference_block": "human_checked_titles_to_check",
                        "reference_file": review_file.name,
                    }
                )
            # Pull tag-shaped tokens out of the free-text notes.
            notes = record.get("notes") or ""
            for token in re.findall(r"\b[A-Z]{1,5}[- ]?\d{1,3}[A-Z]?\b", notes.upper()):
                bucket["expected"].append(
                    {
                        "text": token,
                        "normalized": norm_text(token),
                        "match_key": match_key(token),
                        "bbox": None,
                        "page": page_idx,
                        "discipline": record.get("discipline", "review"),
                        "reference_block": "human_checked_notes",
                        "reference_file": review_file.name,
                    }
                )

    # Deduplicate expected items per page by match key.  Keep the first bbox
    # that exists (prefer a real bbox over a null one).
    for page_idx, bucket in pages.items():
        deduped: dict[str, dict] = {}
        for item in bucket["expected"]:
            key = item["match_key"]
            if not key:
                continue
            if key not in deduped:
                deduped[key] = item
            elif deduped[key].get("bbox") is None and item.get("bbox") is not None:
                deduped[key] = item
        bucket["expected"] = list(deduped.values())

        # Keynotes are location-specific, so only collapse exact duplicates.
        for field in ("keynotes_expected", "invalid_keynotes_expected"):
            seen: set[tuple] = set()
            unique: list[dict] = []
            for item in bucket[field]:
                if not item.get("match_key"):
                    continue
                if item.get("bbox"):
                    sig = (item["match_key"], tuple(round(float(v), 1) for v in item["bbox"]))
                else:
                    sig = (item["match_key"], None)
                if sig in seen:
                    continue
                seen.add(sig)
                unique.append(item)
            bucket[field] = unique

        bucket["sources"] = sorted(bucket["sources"])
    return pages


if __name__ == "__main__":
    index = build_index()
    total = sum(len(v["expected"]) for v in index.values())
    print("pages with reference:", len(index))
    print("total unique expected items:", total)
    for page in sorted(index):
        bucket = index[page]
        if bucket["expected"] or bucket["keynotes_expected"]:
            print(
                page,
                "expected", len(bucket["expected"]),
                "keynotes", len(bucket["keynotes_expected"]),
                "invalid", len(bucket["invalid_keynotes_expected"]),
                "plans", len(bucket["plan_areas"]),
                "sources", ",".join(bucket["sources"]),
            )
