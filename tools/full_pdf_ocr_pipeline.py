#!/usr/bin/env python3
"""Full-document OCR + reference comparison pipeline.

This script renders every page of the source PDF, runs Tesseract OCR, also
reads the embedded PDF text layer, merges the two evidence sources, compares
the result with the legacy JSON reference files, and writes:

  outputs/full_pdf_ocr/detected_text_validation.json
  outputs/full_pdf_ocr/detected_text_details.json
  outputs/full_pdf_ocr/missed_text.json
  outputs/full_pdf_ocr/annotated_full_document.pdf
  outputs/full_pdf_ocr/summary.json
  outputs/full_pdf_ocr/validation_report.json

The output is machine-generated.  It is not human-verified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep Tesseract from oversubscribing the CPU when we run page workers.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import pymupdf as fitz  # noqa: E402
import pytesseract  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from build_reference_index import build_index, match_key, norm_text  # noqa: E402

PDF_PATH = PROJECT_ROOT / "source" / "pdfs" / "5ee2d592b0fb4656bc69b46fa759c999pre.pdf"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_pdf_ocr"
# Keep intermediate page records inside the project so a machine restart or a
# /tmp cleanup cannot discard a long run.  They are removed after verification.
TEMP_DIR = PROJECT_ROOT / ".cache" / "full_pdf_ocr_work"
PATTERNS_PATH = PROJECT_ROOT / "outputs" / "pattern_learning" / "learned_patterns.json"

DISCIPLINE_FILES = {
    "mechanical": next((PROJECT_ROOT / "source" / "json").glob("mechanical_data_out_*.json")),
    "plumbing": next((PROJECT_ROOT / "source" / "json").glob("plumbing_data_out_*.json")),
    "panel": next((PROJECT_ROOT / "source" / "json").glob("panel_data_out_*.json")),
}
VALIDATION_FILE = next((PROJECT_ROOT / "source" / "json").glob("validation_dict_*.json"))

# ---------------------------------------------------------------------------
# Text patterns
# ---------------------------------------------------------------------------

MECH_PREFIXES = {
    "AF", "CD", "CEF", "CR", "DB", "EG", "EV", "FC", "HP", "IF", "IV", "L",
    "PF", "SG", "TD", "SF", "EF", "RF", "UF", "ERV", "HRV", "WSHP", "RTU",
    "AHU", "AC", "VAV", "FCU", "CU", "HX", "CH", "BO", "KEF", "MUA", "VD",
    "SD", "BDD", "DG", "RAG", "EAG", "POC", "FLA", "MCA", "MOCP", "MBH",
    "OA", "RL", "T", "P", "V", "M", "B", "D", "F", "G", "H", "J", "K",
    "N", "Q", "R", "S", "U", "W", "X", "Y", "Z", "CN", "SWE", "VTR", "SPD",
}

PLUMB_PREFIXES = {
    "WC", "FD", "L", "WHP", "WT", "WH", "HB", "HH", "IB", "PC", "P", "PM",
    "S", "SS", "ST", "TP", "TS", "WB", "WD", "WR", "WM", "BT", "ET", "FV",
    "FS", "CO", "FCO", "WCO", "LT", "MT", "RWH", "G", "CW", "HW", "HWR",
    "TW", "PRV", "TV", "AV", "BK", "CP", "DP", "EP", "GP", "RPZ", "SD",
    "SV", "VB", "VP", "WV", "W", "K", "D", "F", "R", "U", "V", "X", "Y", "Z",
}

PANEL_PREFIXES = {"MSB", "FACP", "SEC", "MPOE", "MDP", "DP", "LP", "PP", "RP", "SWB", "ATS", "GEN", "XFMR", "MCC"}

BARE_TAGS = {
    "TD", "FCO", "WCO", "BT", "ET", "CO", "HB", "HH", "IB", "PC", "PM", "SS",
    "ST", "TP", "TS", "WB", "WD", "WR", "WM", "WC", "WHP", "WT", "WH", "FD",
    "FV", "FS", "CEF", "PF", "IF", "IV", "AF", "DB", "EV", "FC", "HP", "CR",
    "CD", "SG", "EG", "PRV", "TV", "AV", "SD", "VD", "RAG", "EAG", "POC",
}

SCHEDULE_WORDS = {
    "SCHEDULE", "SCHEDULES", "MARK", "DESCRIPTION", "SYMBOL", "ABBREVIATION",
    "EQUIPMENT", "FIXTURE", "PANEL", "VOLTAGE", "PHASE", "FLA", "MCA", "MOCP",
    "WEIGHT", "CAPACITY", "AIRFLOW", "ACCESSORIES", "OPTIONS", "NOTES",
    "MAKE", "MODEL", "TYPE", "SIZE", "QUANTITY", "REMARKS", "TOTAL",
}

LEGEND_WORDS = {
    "LEGEND", "KEY", "SYMBOL", "ABBREVIATION", "DESCRIPTION", "GENERAL",
    "NOTES", "GRAPHIC", "STANDARD", "TYPICAL", "DETAIL", "DETAILS",
}

TAG_RE = re.compile(r"^(?P<prefix>[A-Z]{1,5})-?(?P<num>\d{1,3})(?P<suffix>[A-Z]?)$")
PANEL_RE = re.compile(r"^(?:H\d{1,2}[A-C]?|\d{1,2}[A-C])$")
PANEL_CIRCUIT_RE = re.compile(r"^H\d{1,2}[A-C]?-\d+(?:/\d+)?$")
PANEL_SPECIAL_RE = re.compile(r"^(?:MSB|FACP|SEC|MPOE|MDP|DP|LP|PP|RP|SWB|ATS|GEN|XFMR|MCC)\d*$")
DOOR_RE = re.compile(r"^(?:\d{3}[A-Z]|D-?\d{1,3})$")
NUMERIC_RE = re.compile(r"^\d{1,3}$")
SHEET_REF_RE = re.compile(r"^[A-Z]{1,2}\d+\.\d+[A-Z]?$")
CIRCUIT_RE = re.compile(r"^[A-Z]{1,3}-?\d+(?:/\d+)?$")

COMMON_WORDS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "ARE", "NOT", "ALL",
    "SEE", "NOTE", "NOTES", "TYP", "TYPICAL", "MIN", "MAX", "TOTAL", "NEW",
    "EXISTING", "PROVIDE", "INSTALL", "ABOVE", "BELOW", "HIGH", "LOW",
    "NORTH", "SOUTH", "EAST", "WEST", "ROOM", "FLOOR", "WALL", "DOOR",
    "WINDOW", "PLAN", "SECTION", "DETAIL", "SCALE", "DATE", "SHEET",
    "PROJECT", "DRAWN", "CHECKED", "REV", "MEMO", "CAD", "FILE", "THESE",
    "DRAWINGS", "DESIGNS", "PROPERTY", "ARCHITECT", "WORK", "CONSENT",
    "WRITTEN", "COPY", "USED", "CONNECTION", "OTHER", "WITHOUT", "PART",
    "SHALL", "BE", "COPIED", "OR", "IN", "OF", "TO", "AT", "ON", "BY",
    "A", "I", "IS", "IT", "AS", "AN", "IF", "NO", "UP", "DN", "EQ",
}


BAD_CHARS = set(".=@()/\\|%\u00b0&+*#_'\"")


def has_bad_chars(text: str) -> bool:
    """Reject anything that is not a plain tag token (letters, digits, hyphen)."""
    stripped = text.strip().upper()
    if stripped.startswith("-") or stripped.endswith("-"):
        return True
    return not bool(re.fullmatch(r"[A-Z0-9-]+", stripped))


def is_tag_like(text: str) -> bool:
    if has_bad_chars(text):
        return False
    key = match_key(text)
    if not key or len(key) > 12:
        return False
    if SHEET_REF_RE.match(text.upper()):
        return False
    if TAG_RE.match(key) or PANEL_RE.match(key) or PANEL_CIRCUIT_RE.match(key) or PANEL_SPECIAL_RE.match(key):
        return True
    return False


def is_candidate_text(text: str) -> bool:
    """Broad gate used before reference matching.

    We keep every plausible tag/keynote here so that a legacy reference entry
    can still be matched even when its shape is unusual (for example the room
    tags on architectural sheets).  Strict filtering happens afterwards.
    """
    if not text or len(text) > 14:
        return False
    if has_bad_chars(text):
        return False
    key = match_key(text)
    if not key:
        return False
    if SHEET_REF_RE.match(text.upper()):
        return False
    return bool(
        TAG_RE.match(key)
        or PANEL_RE.match(key)
        or PANEL_CIRCUIT_RE.match(key)
        or PANEL_SPECIAL_RE.match(key)
        or DOOR_RE.match(key)
        or NUMERIC_RE.match(key)
    )


def classify_text(text: str, page_index: int, bbox: list[float], page_rect: tuple[float, float]) -> tuple[str, str]:
    """Return (category, pattern_name).

    Discipline is derived from the tag prefix, not forced by page range.  A
    panel schedule can legitimately list a plumbing tag such as ``PC-1``, and
    a mechanical detail sheet can contain electrical circuit numbers.
    """
    raw = text.strip().upper().strip(".,;:()[]{}<>\"'")
    key = match_key(raw)
    if not key:
        return "other", "empty"
    if has_bad_chars(raw):
        return "other", "invalid_chars"

    if PANEL_SPECIAL_RE.match(key):
        return "panel", "panel_special"
    if PANEL_CIRCUIT_RE.match(key):
        return "panel", "panel_circuit"
    if PANEL_RE.match(key):
        return "panel", "panel_tag"
    if DOOR_RE.match(key):
        return "door", "door_reference"
    if NUMERIC_RE.match(key):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= 45 and h <= 35 and (w * h) <= 1400:
            return "keynote", "keynote_number"
        return "other", "number"

    m = TAG_RE.match(key)
    if m:
        prefix = m.group("prefix")
        learned_panel = _LEARNED_PREFIXES.get("panel", set())
        learned_mechanical = _LEARNED_PREFIXES.get("mechanical", set())
        learned_plumbing = _LEARNED_PREFIXES.get("plumbing", set())
        if prefix in PANEL_PREFIXES or prefix in learned_panel:
            return "panel", "panel_tag"
        in_mechanical = prefix in MECH_PREFIXES or prefix in learned_mechanical
        in_plumbing = prefix in PLUMB_PREFIXES or prefix in learned_plumbing
        if in_mechanical and in_plumbing:
            return "other", "ambiguous_tag"
        if in_mechanical:
            return "mechanical", "mechanical_tag"
        if in_plumbing:
            return "plumbing", "plumbing_tag"
        return "other", "other_tag"

    if raw.isalpha() and 2 <= len(raw) <= 5:
        return "other", "uppercase_word"
    return "other", "other"


def is_important(det: dict, page_index: int) -> bool:
    """Strict importance test applied after reference matching.

    A detection is important when it is either explicitly expected by the
    legacy reference, or it is a clean, hyphenated equipment tag with a known
    discipline prefix.  Bare legend words, wire labels, room tags and drawing
    callouts are deliberately excluded unless the reference expects them.
    """
    if det.get("matched_reference"):
        return True
    if det.get("category") == "keynote":
        return False
    text = det.get("text", "")
    if has_bad_chars(text):
        return False
    raw = text.strip().upper()
    key = match_key(raw)
    if not key or len(key) > 12:
        return False

    category = det.get("category", "other")

    # Panel schedules/circuits are only meaningful on panel sheets.
    if category == "panel":
        if not (99 <= page_index <= 142):
            return False
        if PANEL_SPECIAL_RE.match(key) or PANEL_CIRCUIT_RE.match(key) or PANEL_RE.match(key):
            return True
        m = TAG_RE.match(key)
        if m and "-" in raw and len(m.group("prefix")) >= 2:
            return True
        return False

    # Equipment tags: require an explicit hyphen and a multi-letter prefix.
    # This removes single-letter wire/room labels such as W1, C7, S2, N4.
    m = TAG_RE.match(key)
    if not m:
        return False
    prefix = m.group("prefix")
    if len(prefix) < 2 or "-" not in raw:
        return False
    if raw.count("-") > 1:
        return False
    if category in {"mechanical", "plumbing"}:
        return True
    return False


def confidence_bucket(source: str, confidence: float) -> str:
    if source == "pdf_text" or confidence >= 0.70:
        return "confirmed"
    if confidence >= 0.40:
        return "uncertain"
    return "low"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def bbox_union(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def bbox_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_center(b: list[float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_distance(a: list[float], b: list[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def point_in_bbox(x: float, y: float, b: list[float], pad: float = 0.0) -> bool:
    return b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad


def clamp_bbox(b: list[float], width: float, height: float) -> list[float]:
    x0 = max(0.0, min(float(b[0]), width))
    y0 = max(0.0, min(float(b[1]), height))
    x1 = max(0.0, min(float(b[2]), width))
    y1 = max(0.0, min(float(b[3]), height))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def fuzzy_match_key(a: str, b: str) -> bool:
    """Conservative OCR-error matching.

    Only strings of at least four characters that share their first two
    characters are allowed to differ.  This prevents unrelated short tags such
    as ``L2``/``P2`` or ``WC2``/``WCO`` from being treated as OCR variants.
    """
    if a == b:
        return True
    if not a or not b:
        return False
    if len(a) < 4 or len(b) < 4:
        return False
    if abs(len(a) - len(b)) > 1:
        return False
    if a[:2] != b[:2]:
        return False
    limit = 1 if max(len(a), len(b)) <= 6 else 2
    return levenshtein(a, b) <= limit


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------


def _word_text(raw: str) -> str:
    return raw.strip().upper().strip(".,;:()[]{}<>\"'")


def extract_pdf_candidates(
    page: fitz.Page,
    page_index: int,
    page_rect: tuple[float, float],
    reference_keys: set[str] | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    words = page.get_text("words")
    parsed = []
    for item in words:
        if len(item) < 8:
            continue
        x0, y0, x1, y1, raw, block, line, word_no = item[:8]
        text = _word_text(str(raw))
        if not text:
            continue
        parsed.append(
            {
                "text": text,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "block": block,
                "line": line,
                "word_no": word_no,
            }
        )

    # Word-level extraction.  Keep every plausible tag/keynote so the legacy
    # reference can still be matched; strict filtering happens after matching.
    for w in parsed:
        text = w["text"]
        key = match_key(text)
        if is_candidate_text(text) or (
            reference_keys and key in reference_keys and not has_bad_chars(text)
        ):
            category, pattern = classify_text(text, page_index, w["bbox"], page_rect)
            candidates.append(
                {
                    "text": text,
                    "bbox": w["bbox"],
                    "confidence": 1.0,
                    "source": "pdf_text",
                    "category": category,
                    "pattern": pattern,
                }
            )

    # Line-level extraction catches tags split across words.
    lines: dict[tuple[Any, Any], list[dict]] = defaultdict(list)
    for w in parsed:
        lines[(w["block"], w["line"])].append(w)
    for _, line_words in lines.items():
        line_words.sort(key=lambda w: w["word_no"])
        if not line_words:
            continue
        texts = [w["text"] for w in line_words]
        joined = " ".join(texts)
        compact = "".join(texts)
        for candidate_text, source_text in ((joined, joined), (compact, compact)):
            # Split on commas / semicolons so a schedule row yields separate tags.
            for piece in re.split(r"[,;]", candidate_text):
                piece = piece.strip()
                if not piece or len(piece) > 40:
                    continue
                # Only use the line-level fallback when the piece is a clean tag.
                key = match_key(piece)
                if not (TAG_RE.match(key) or PANEL_RE.match(key) or PANEL_CIRCUIT_RE.match(key) or PANEL_SPECIAL_RE.match(key)):
                    continue
                if piece in {w["text"] for w in line_words}:
                    continue
                bbox = line_words[0]["bbox"]
                for w in line_words[1:]:
                    bbox = bbox_union(bbox, w["bbox"])
                category, pattern = classify_text(piece, page_index, bbox, page_rect)
                candidates.append(
                    {
                        "text": piece,
                        "bbox": bbox,
                        "confidence": 1.0,
                        "source": "pdf_text",
                        "category": category,
                        "pattern": pattern,
                    }
                )
    return dedup_candidates(candidates)


def extract_ocr_candidates(
    image: Image.Image,
    page_index: int,
    scale: float,
    page_rect: tuple[float, float],
    reference_keys: set[str] | None = None,
) -> list[dict]:
    data = pytesseract.image_to_data(
        image,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    words: list[dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        raw = str(data["text"][i]).strip()
        if not raw:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 20:
            continue
        text = _word_text(raw)
        if not text:
            continue
        left = float(data["left"][i])
        top = float(data["top"][i])
        width = float(data["width"][i])
        height = float(data["height"][i])
        bbox = [left / scale, top / scale, (left + width) / scale, (top + height) / scale]
        words.append(
            {
                "text": text,
                "bbox": bbox,
                "confidence": conf / 100.0,
                "block": data["block_num"][i],
                "par": data["par_num"][i],
                "line": data["line_num"][i],
                "word_no": data["word_num"][i],
            }
        )

    candidates: list[dict] = []

    # Word-level extraction.
    for w in words:
        text = w["text"]
        key = match_key(text)
        if is_candidate_text(text) or (
            reference_keys and key in reference_keys and not has_bad_chars(text)
        ):
            category, pattern = classify_text(text, page_index, w["bbox"], page_rect)
            candidates.append(
                {
                    "text": text,
                    "bbox": w["bbox"],
                    "confidence": w["confidence"],
                    "source": "ocr",
                    "category": category,
                    "pattern": pattern,
                }
            )

    # Line-level extraction for split tags.
    lines: dict[tuple[Any, Any, Any], list[dict]] = defaultdict(list)
    for w in words:
        lines[(w["block"], w["par"], w["line"])].append(w)
    for _, line_words in lines.items():
        line_words.sort(key=lambda w: w["word_no"])
        if not line_words or len(line_words) > 8:
            continue
        texts = [w["text"] for w in line_words]
        joined = " ".join(texts)
        compact = "".join(texts)
        for candidate_text in (joined, compact):
            for piece in re.split(r"[,;]", candidate_text):
                piece = piece.strip()
                if not piece or len(piece) > 40:
                    continue
                key = match_key(piece)
                if not (TAG_RE.match(key) or PANEL_RE.match(key) or PANEL_CIRCUIT_RE.match(key) or PANEL_SPECIAL_RE.match(key)):
                    continue
                if piece in {w["text"] for w in line_words}:
                    continue
                bbox = line_words[0]["bbox"]
                conf = line_words[0]["confidence"]
                for w in line_words[1:]:
                    bbox = bbox_union(bbox, w["bbox"])
                    conf = min(conf, w["confidence"])
                category, pattern = classify_text(piece, page_index, bbox, page_rect)
                candidates.append(
                    {
                        "text": piece,
                        "bbox": bbox,
                        "confidence": conf,
                        "source": "ocr",
                        "category": category,
                        "pattern": pattern,
                    }
                )
    return dedup_candidates(candidates)


def dedup_candidates(candidates: list[dict]) -> list[dict]:
    """Merge candidates with the same match key and strongly overlapping boxes."""
    out: list[dict] = []
    by_key: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        key = match_key(cand["text"])
        if not key:
            continue
        merged = False
        for existing in by_key[key]:
            if bbox_iou(existing["bbox"], cand["bbox"]) >= 0.35 or bbox_distance(existing["bbox"], cand["bbox"]) <= 2.0:
                if cand["confidence"] > existing["confidence"]:
                    existing["confidence"] = cand["confidence"]
                if cand["source"] == "pdf_text" and existing["source"] != "pdf_text":
                    existing["source"] = "pdf_text"
                merged = True
                break
        if not merged:
            copy = dict(cand)
            out.append(copy)
            by_key[key].append(copy)
    return out


def merge_evidence(pdf_candidates: list[dict], ocr_candidates: list[dict]) -> list[dict]:
    """Combine PDF-text and OCR candidates, keeping one entry per text/location."""
    merged = [dict(c) for c in pdf_candidates]
    for ocr in ocr_candidates:
        key = match_key(ocr["text"])
        matched = False
        for existing in merged:
            if match_key(existing["text"]) != key:
                continue
            if bbox_iou(existing["bbox"], ocr["bbox"]) >= 0.25 or bbox_distance(existing["bbox"], ocr["bbox"]) <= 8.0:
                # PDF text wins for the bbox, but keep OCR confidence as evidence.
                existing.setdefault("ocr_confidence", ocr["confidence"])
                existing["ocr_support"] = True
                if ocr["confidence"] > existing.get("confidence", 0.0) and existing["source"] == "ocr":
                    existing["confidence"] = ocr["confidence"]
                matched = True
                break
        if not matched:
            merged.append(dict(ocr))
    return dedup_candidates(merged)


# ---------------------------------------------------------------------------
# Reference matching
# ---------------------------------------------------------------------------


def build_page_old_texts(bucket: dict | None) -> set[str]:
    if not bucket:
        return set()
    texts = set()
    for item in bucket.get("expected", []):
        if item.get("match_key"):
            texts.add(item["match_key"])
    for item in bucket.get("keynotes_expected", []):
        if item.get("match_key"):
            texts.add(item["match_key"])
    return texts


def best_detection_for_expected(expected: dict, detections: list[dict]) -> dict | None:
    key = expected.get("match_key") or match_key(expected["text"])
    if not key:
        return None
    exact = [d for d in detections if match_key(d["text"]) == key]
    if exact:
        ref_bbox = expected.get("bbox")
        if ref_bbox:
            exact.sort(key=lambda d: (-bbox_iou(d["bbox"], ref_bbox), bbox_distance(d["bbox"], ref_bbox)))
        else:
            exact.sort(key=lambda d: (d["source"] != "pdf_text", -d["confidence"], bbox_area(d["bbox"])))
        return exact[0]

    # Fuzzy matching, only when a reference bbox exists and the boxes are close.
    ref_bbox = expected.get("bbox")
    if ref_bbox:
        nearby = [d for d in detections if bbox_distance(d["bbox"], ref_bbox) <= 35.0]
        for det in nearby:
            if fuzzy_match_key(match_key(det["text"]), key):
                return det
    return None


def match_reference(
    detections: list[dict],
    expected_items: list[dict],
    page_rect: tuple[float, float],
    plan_areas: list[list[float]],
) -> tuple[list[dict], list[dict]]:
    """Return (reference_only_items, unmatched_expected)."""
    reference_only: list[dict] = []
    matched_expected_keys: set[str] = set()

    for expected in expected_items:
        key = expected.get("match_key") or match_key(expected["text"])
        det = best_detection_for_expected(expected, detections)
        if det is not None:
            det["matched_reference"] = expected["text"]
            det["already_present_in_old_json"] = True
            det["reference_bbox"] = expected.get("bbox")
            det["reference_block"] = expected.get("reference_block")
            det["reference_file"] = expected.get("reference_file")
            det["reason"] = "Detected on the PDF and matched to an existing reference entry."
            if expected.get("reference_block") in {"used_bboxes", "unused_bboxes"}:
                det["count_as_text"] = True
            if key:
                matched_expected_keys.add(key)
        else:
            fallback_bbox = expected.get("bbox")
            if not fallback_bbox:
                if plan_areas:
                    fallback_bbox = plan_areas[0]
                    for b in plan_areas[1:]:
                        fallback_bbox = bbox_union(fallback_bbox, b)
                else:
                    fallback_bbox = [0.0, 0.0, page_rect[0], page_rect[1]]
            fallback_bbox = clamp_bbox(fallback_bbox, page_rect[0], page_rect[1])
            ref_category = (
                expected.get("discipline")
                if expected.get("discipline") in {"mechanical", "plumbing", "panel"}
                else classify_text(expected["text"], 0, fallback_bbox, page_rect)[0]
            )
            if re.fullmatch(r"\d{1,3}", expected["text"]):
                ref_category = "keynote"
            reference_only.append(
                {
                    "text": expected["text"],
                    "normalized_text": expected.get("normalized") or norm_text(expected["text"]),
                    "match_key": key,
                    "bbox": fallback_bbox,
                    "source": "reference_json" if expected.get("bbox") else "fallback",
                    "category": ref_category,
                    "pattern": "reference_expected",
                    "ocr_confidence": 0.0,
                    "is_fallback_used": True,
                    "reference_bbox": expected.get("bbox"),
                    "reference_block": expected.get("reference_block"),
                    "reference_file": expected.get("reference_file"),
                    "matched_reference": expected["text"],
                    "already_present_in_old_json": True,
                    "visually_detected": False,
                    "reason": "Expected from the legacy reference JSON but not detected by OCR/PDF text on this page.",
                }
            )
    return reference_only, [e for e in expected_items if (e.get("match_key") or match_key(e["text"])) not in matched_expected_keys]


def match_keynotes(
    detections: list[dict],
    expected_keynotes: list[dict],
    page_rect: tuple[float, float],
    plan_areas: list[list[float]],
    page_index: int,
) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    invalid: list[dict] = []
    matched_detection_ids: set[int] = set()

    for expected in expected_keynotes:
        key = expected.get("match_key") or match_key(expected["text"])
        ref_bbox = expected.get("bbox")
        best = None
        best_score = -float("inf")
        for det in detections:
            if id(det) in matched_detection_ids:
                continue
            if match_key(det["text"]) != key:
                continue
            # Expected keynotes are numeric; accept a numeric detection even if
            # its initial category was "other" because of a large box.
            if det["category"] != "keynote" and not NUMERIC_RE.match(key or ""):
                continue
            score = 0.0
            if ref_bbox:
                score = bbox_iou(det["bbox"], ref_bbox) * 10.0 - bbox_distance(det["bbox"], ref_bbox) * 0.01
            if score > best_score:
                best = det
                best_score = score
        if best is not None:
            best["category"] = "keynote"
            best["pattern"] = "keynote_number"
            best["matched_reference"] = expected["text"]
            best["already_present_in_old_json"] = True
            best["reference_bbox"] = ref_bbox
            best["reason"] = "Keynote detected and matched to the reference validation dictionary."
            valid.append(
                {
                    "bbox": best["bbox"],
                    "text": best["text"],
                    "shape": expected.get("shape") or "Circle",
                    "source": best["source"],
                    "ocr_confidence": best["confidence"],
                    "matched_reference": expected["text"],
                }
            )
            matched_detection_ids.add(id(best))
        else:
            bbox = ref_bbox or ([0.0, 0.0, page_rect[0], page_rect[1]])
            bbox = clamp_bbox(bbox, page_rect[0], page_rect[1])
            invalid.append(
                {
                    "bbox": bbox,
                    "text": expected["text"],
                    "shape": expected.get("shape") or "Circle",
                    "source": "reference_json",
                    "ocr_confidence": 0.0,
                    "matched_reference": expected["text"],
                }
            )

    # For pages without reference keynotes we intentionally do not infer new
    # keynotes from arbitrary numeric text.  Drawings contain hundreds of
    # dimensions, circuit numbers and detail callouts; treating every small
    # number near a margin as a keynote produced unusable noise.  Keynote
    # validation is therefore reference-driven and conservative.
    if not expected_keynotes:
        pass

    # Deduplicate by text + rounded bbox.
    def dedup(items: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for item in items:
            key = (item["text"], tuple(round(v, 1) for v in item["bbox"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    return dedup(valid)[:250], dedup(invalid)[:250]


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------


def related_old_files(page_index: int, bucket: dict | None) -> list[str]:
    files = []
    if bucket:
        for source in bucket.get("sources", []):
            if source not in files:
                files.append(source)
    if 99 <= page_index <= 142 and DISCIPLINE_FILES["panel"]:
        files.append(DISCIPLINE_FILES["panel"].name)
    elif 143 <= page_index <= 171 and DISCIPLINE_FILES["mechanical"]:
        files.append(DISCIPLINE_FILES["mechanical"].name)
    elif 173 <= page_index <= 212 and DISCIPLINE_FILES["plumbing"]:
        files.append(DISCIPLINE_FILES["plumbing"].name)
    seen = []
    for f in files:
        if f not in seen:
            seen.append(f)
    return seen


def choose_used_detections(detections: list[dict]) -> list[dict]:
    """One confirmed detection per normalized text, preferring reference matches."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        if det["category"] == "keynote" and not det.get("count_as_text"):
            continue
        if not det.get("important"):
            continue
        if det.get("is_uncertain"):
            continue
        if not det.get("confirmed", False):
            continue
        grouped[match_key(det["text"])].append(det)

    chosen = []
    for key, group in grouped.items():
        group.sort(
            key=lambda d: (
                not bool(d.get("matched_reference")),
                d["source"] != "pdf_text",
                -d.get("confidence", 0.0),
                bbox_area(d["bbox"]),
            )
        )
        chosen.append(group[0])
    chosen.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return chosen


def process_page(page_index: int, dpi: int, annotate: bool) -> dict:
    global _DOC
    doc = _DOC
    page = doc[page_index]
    page_rect = (float(page.rect.width), float(page.rect.height))
    scale = dpi / 72.0

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    bucket = _REFERENCE.get(page_index, {})
    expected_items = bucket.get("expected", [])
    expected_keynotes = bucket.get("keynotes_expected", [])
    plan_areas = bucket.get("plan_areas", [])
    old_texts = build_page_old_texts(bucket)
    old_files = related_old_files(page_index, bucket)

    # Bare reference tags such as FCO/WCO/TD/C are not tag-shaped on their own,
    # so allow them through extraction only when the reference expects them.
    reference_keys = {
        item.get("match_key")
        for item in expected_items + expected_keynotes
        if item.get("match_key")
    }

    pdf_candidates = extract_pdf_candidates(page, page_index, page_rect, reference_keys)
    ocr_candidates = extract_ocr_candidates(image, page_index, scale, page_rect, reference_keys)
    detections = merge_evidence(pdf_candidates, ocr_candidates)

    reference_only, _unmatched = match_reference(detections, expected_items, page_rect, plan_areas)

    # Decide confirmation/uncertainty and missed status for every detection.
    for det in detections:
        key = match_key(det["text"])
        det["already_present_in_old_json"] = key in old_texts or bool(det.get("matched_reference"))
        det["visually_detected"] = True
        det["related_old_json_files"] = old_files
        det["is_uncertain"] = False
        det["confirmed"] = False
        if det.get("matched_reference"):
            det["confirmed"] = True
            det["confidence"] = max(det.get("confidence", 0.0), 0.60)
            det["reason"] = "Detected on the PDF and matched to the legacy reference JSON."
            continue
        bucket_name = confidence_bucket(det["source"], det.get("confidence", 0.0))
        if bucket_name == "confirmed":
            det["confirmed"] = True
            det["reason"] = "Detected on the PDF with sufficient OCR/PDF-text confidence."
        elif bucket_name == "uncertain":
            det["is_uncertain"] = True
            det["reason"] = "OCR evidence is below the confirmation threshold; kept as uncertain."
        else:
            det["is_uncertain"] = True
            det["reason"] = "Low OCR confidence; kept as uncertain rather than a confirmed detection."

    # Keynotes are handled separately from the tag lists.
    valid_keynotes, invalid_keynotes = match_keynotes(
        detections, expected_keynotes, page_rect, plan_areas, page_index
    )

    # Strict importance is decided after reference matching so that any item
    # explicitly expected by the legacy JSON is retained even when its shape is
    # unusual (for example architectural room tags).
    for det in detections:
        det["important"] = is_important(det, page_index)

    # Mark keynote detections that were used by the keynote matcher.
    keynote_keys = {(k["text"], tuple(round(v, 1) for v in k["bbox"])) for k in valid_keynotes}
    for det in detections:
        if det["category"] == "keynote":
            det["used_as_keynote"] = (det["text"], tuple(round(v, 1) for v in det["bbox"])) in keynote_keys

    used_detections = choose_used_detections(detections)
    confident_text_keys = {match_key(d["text"]) for d in used_detections}

    # Confirmed misses: important, confirmed, not present in the old JSON for this page.
    missed = []
    for det in detections:
        if det["category"] == "keynote":
            continue
        if not det.get("important"):
            continue
        if not det.get("confirmed") or det.get("is_uncertain"):
            continue
        if det["already_present_in_old_json"]:
            continue
        missed.append(det)

    # Uncertain detections (OCR only; reference-matched items are not uncertain).
    uncertain = []
    for det in detections:
        if det["category"] == "keynote":
            continue
        if not det.get("important"):
            continue
        if det.get("matched_reference"):
            continue
        if not det.get("is_uncertain"):
            continue
        # A lower-confidence OCR duplicate of text already confirmed on this
        # page is not an unresolved uncertainty; keep the uncertain list clean.
        if match_key(det["text"]) in confident_text_keys:
            continue
        uncertain.append(det)

    # Deduplicate missed/uncertain by text, keeping the best evidence.
    def dedup_dets(items: list[dict]) -> list[dict]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            grouped[match_key(item["text"])].append(item)
        out = []
        for key, group in grouped.items():
            group.sort(key=lambda d: (d["source"] != "pdf_text", -d.get("confidence", 0.0), bbox_area(d["bbox"])))
            out.append(group[0])
        out.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return out

    missed = dedup_dets(missed)
    uncertain = dedup_dets(uncertain)

    # Build the page record (details + validation source of truth).
    det_records = []
    for det in detections:
        if det["category"] == "keynote" and not det.get("count_as_text"):
            continue
        if not det.get("important"):
            continue
        det_records.append(
            {
                "text": det["text"],
                "normalized_text": norm_text(det["text"]),
                "ocr_confidence": round(float(det.get("confidence", 0.0)), 4),
                "bbox": clamp_bbox(det["bbox"], page_rect[0], page_rect[1]),
                "source": det["source"],
                "category": det["category"],
                "matched_reference": det.get("matched_reference"),
                "already_present_in_old_json": bool(det["already_present_in_old_json"]),
                "visually_detected": True,
                "reason": det.get("reason", ""),
                "pattern": det.get("pattern"),
                "is_uncertain": bool(det.get("is_uncertain", False)),
                "confirmed": bool(det.get("confirmed", False)),
                "used_as_keynote": bool(det.get("used_as_keynote", False)),
                "count_as_text": bool(det.get("count_as_text", False)),
                "reference_bbox": det.get("reference_bbox"),
                "reference_block": det.get("reference_block"),
                "reference_file": det.get("reference_file"),
                "related_old_json_files": old_files,
            }
        )

    keynote_records = []
    for k in valid_keynotes:
        keynote_records.append(
            {
                "text": k["text"],
                "normalized_text": norm_text(k["text"]),
                "ocr_confidence": round(float(k.get("ocr_confidence", 0.0)), 4),
                "bbox": k["bbox"],
                "source": k.get("source", "ocr"),
                "category": "keynote",
                "matched_reference": k.get("matched_reference"),
                "already_present_in_old_json": bool(k.get("matched_reference")),
                "visually_detected": True,
                "reason": "Keynote detected on the PDF."
                if not k.get("matched_reference")
                else "Keynote detected and matched to the reference validation dictionary.",
                "pattern": "keynote_number",
                "is_uncertain": False,
                "confirmed": True,
                "used_as_keynote": True,
                "reference_bbox": None,
                "reference_block": None,
                "reference_file": None,
                "related_old_json_files": old_files,
                "keynote_status": "valid_outer_keynote",
            }
        )

    page_record = {
        "page_index": page_index,
        "pdf_page_number": page_index + 1,
        "page_width": page_rect[0],
        "page_height": page_rect[1],
        "discipline": "mechanical" if 143 <= page_index <= 171 else "plumbing" if 173 <= page_index <= 212 else "panel" if 99 <= page_index <= 142 else "architectural/other",
        "detections": det_records,
        "keynotes": keynote_records,
        "reference_only_items": [
            {
                "text": item["text"],
                "normalized_text": item["normalized_text"],
                "bbox": item["bbox"],
                "source": item["source"],
                "category": item["category"],
                "matched_reference": item["matched_reference"],
                "already_present_in_old_json": True,
                "visually_detected": False,
                "ocr_confidence": 0.0,
                "is_fallback_used": True,
                "reason": item["reason"],
                "reference_bbox": item["reference_bbox"],
                "reference_block": item["reference_block"],
                "reference_file": item["reference_file"],
                "related_old_json_files": old_files,
            }
            for item in reference_only
        ],
        "invalid_outer_keynotes": invalid_keynotes,
        "counts": {
            "detections": len(det_records),
            "confirmed_detections": sum(1 for d in det_records if d["confirmed"]),
            "uncertain_detections": len(uncertain),
            "missed": len(missed),
            "reference_only": len(reference_only),
            "valid_keynotes": len(valid_keynotes),
            "invalid_keynotes": len(invalid_keynotes),
        },
        "missed_items": [
            {
                "text": det["text"],
                "normalized_text": norm_text(det["text"]),
                "bbox": clamp_bbox(det["bbox"], page_rect[0], page_rect[1]),
                "category": det["category"],
                "ocr_confidence": round(float(det.get("confidence", 0.0)), 4),
                "detection_source": det["source"],
                "matching_reference_pattern": det.get("pattern"),
                "why_missed": "Text is visibly present in the PDF and absent from the legacy JSON entries for this page.",
                "related_old_json_files": old_files,
            }
            for det in missed
        ],
        "uncertain_items": [
            {
                "text": det["text"],
                "normalized_text": norm_text(det["text"]),
                "bbox": clamp_bbox(det["bbox"], page_rect[0], page_rect[1]),
                "category": det["category"],
                "ocr_confidence": round(float(det.get("confidence", 0.0)), 4),
                "detection_source": det["source"],
                "matching_reference_pattern": det.get("pattern"),
                "why_uncertain": det.get("reason", "OCR confidence below confirmation threshold."),
                "related_old_json_files": old_files,
            }
            for det in uncertain
        ],
    }

    # Annotate the rendered page.
    if annotate:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        try:
            font = ImageFont.truetype(font_path, size=max(12, int(round(dpi * 0.10))))
        except Exception:
            font = ImageFont.load_default()

        def draw_box(bbox: list[float], color: tuple[int, int, int], label: str, prefix: str = "") -> None:
            page_area = max(1.0, page_rect[0] * page_rect[1])
            box_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if box_area > 0.12 * page_area:
                # A whole-plan fallback box would obscure the drawing.  Mark its
                # centre with a small ring instead of drawing the full extent.
                cx = (bbox[0] + bbox[2]) / 2.0 * scale
                cy = (bbox[1] + bbox[3]) / 2.0 * scale
                r = max(6, int(round(dpi * 0.045)))
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
                text = (prefix + label).strip()
                if text:
                    if len(text) > 28:
                        text = text[:27] + "\u2026"
                    draw.text(
                        (cx + r + 3, cy - font.size / 2),
                        text,
                        fill=color,
                        font=font,
                        stroke_width=2,
                        stroke_fill=(255, 255, 255),
                    )
                return
            x0, y0, x1, y1 = [v * scale for v in bbox]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(annotated.width - 1, x1), min(annotated.height - 1, y1)
            if x1 <= x0 or y1 <= y0:
                return
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            text = (prefix + label).strip()
            if not text:
                return
            if len(text) > 28:
                text = text[:27] + "\u2026"
            tx = x0
            ty = y0 - font.size - 3
            if ty < 0:
                ty = y1 + 2
            if ty > annotated.height - font.size:
                ty = max(0, y0 + 1)
            draw.text((tx, ty), text, fill=color, font=font, stroke_width=2, stroke_fill=(255, 255, 255))

        blue = (0, 80, 220)
        red = (210, 30, 30)
        orange = (235, 135, 0)

        # Blue: confidently detected text that is already represented by the
        # legacy JSON. Red is reserved for text detected on the PDF but absent
        # from the legacy JSON; orange is reserved for uncertainty/fallbacks.
        for det in used_detections:
            if det["already_present_in_old_json"]:
                draw_box(det["bbox"], blue, det["text"])

        # Red: confirmed misses (already covered above for used detections, but
        # include misses that were not selected as the representative instance).
        for det in missed:
            draw_box(det["bbox"], red, det["text"])

        # Orange: uncertain OCR detections.
        for det in uncertain:
            draw_box(det["bbox"], orange, det["text"], prefix="? ")

        # Red: reference text missing from the PDF evidence. The fallback box
        # is only a review marker because its location is not directly known.
        for item in reference_only:
            draw_box(item["bbox"], red, item["text"], prefix="MISS ")

        # Keynotes.
        for k in valid_keynotes:
            color = blue if k.get("matched_reference") else red
            draw_box(k["bbox"], color, k["text"])
        for k in invalid_keynotes:
            draw_box(k["bbox"], red, k["text"], prefix="MISS ")

        image_path = _TEMP_DIR / f"page_{page_index:04d}.jpg"
        annotated.save(image_path, "JPEG", quality=82, optimize=True)
    else:
        image_path = None

    page_record["annotated_image"] = str(image_path) if image_path else None
    return page_record


# ---------------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------------

_DOC: fitz.Document | None = None
_REFERENCE: dict[int, dict] = {}
_TEMP_DIR: Path = TEMP_DIR
_LEARNED_PREFIXES: dict[str, set[str]] = {}


def _init_worker(pdf_path: str, reference: dict[int, dict], temp_dir: str) -> None:
    global _DOC, _REFERENCE, _TEMP_DIR
    _DOC = fitz.open(pdf_path)
    _REFERENCE = reference
    _TEMP_DIR = Path(temp_dir)
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)


def load_learned_prefixes(path: Path = PATTERNS_PATH) -> dict[str, set[str]]:
    """Load learned prefix features without making the pipeline depend on them."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        learned = data.get("learned_features", {}).get("tag_prefixes_by_discipline", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return {
        discipline: set(prefixes)
        for discipline, prefixes in learned.items()
        if isinstance(prefixes, dict)
    }


def _worker(args: tuple[int, int, bool]) -> dict:
    page_index, dpi, annotate = args
    try:
        return process_page(page_index, dpi, annotate)
    except Exception as exc:  # keep one bad page from aborting the full run
        page = _DOC[page_index]
        page_rect = (float(page.rect.width), float(page.rect.height))
        image_path = None
        if annotate:
            try:
                scale = dpi / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                image_path = _TEMP_DIR / f"page_{page_index:04d}.jpg"
                image.save(image_path, "JPEG", quality=82, optimize=True)
            except Exception:
                image_path = None
        return {
            "page_index": page_index,
            "pdf_page_number": page_index + 1,
            "page_width": page_rect[0],
            "page_height": page_rect[1],
            "discipline": "unknown",
            "detections": [],
            "keynotes": [],
            "reference_only_items": [],
            "invalid_outer_keynotes": [],
            "counts": {
                "detections": 0,
                "confirmed_detections": 0,
                "uncertain_detections": 0,
                "missed": 0,
                "reference_only": 0,
                "valid_keynotes": 0,
                "invalid_keynotes": 0,
            },
            "missed_items": [],
            "uncertain_items": [],
            "annotated_image": str(image_path) if image_path else None,
            "processing_error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def empty_validation_page() -> dict:
    return {
        "used_bboxes": {"bbox": [[]], "text": [[]], "shape": [[]], "is_fallback_used": [[]]},
        "unused_bboxes": {"bbox": [[]], "text": [[]], "shape": [[]], "is_fallback_used": [[]]},
        "valid_outer_keynotes": {"bbox": [], "text": [], "shape": []},
        "invalid_outer_keynotes": {"bbox": [], "text": [], "shape": []},
    }


def build_validation(page_records: list[dict]) -> dict:
    validation = {}
    for record in page_records:
        page_index = record["page_index"]
        used = []
        for det in record["detections"]:
            if det["category"] == "keynote" and not det.get("count_as_text"):
                continue
            if not det.get("confirmed") or det.get("is_uncertain"):
                continue
            used.append(det)
        # One entry per normalized text for the validation dictionary.
        grouped: dict[str, list[dict]] = defaultdict(list)
        for det in used:
            grouped[match_key(det["text"])].append(det)
        chosen = []
        for key, group in grouped.items():
            group.sort(
                key=lambda d: (
                    not bool(d.get("matched_reference")),
                    d["source"] != "pdf_text",
                    -d.get("ocr_confidence", 0.0),
                    bbox_area(d["bbox"]),
                )
            )
            chosen.append(group[0])
        chosen.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

        used_bboxes = {
            "bbox": [[d["bbox"] for d in chosen]],
            "text": [[d["text"] for d in chosen]],
            "shape": [["Text" for _ in chosen]],
            "is_fallback_used": [[False for _ in chosen]],
        }

        ref_only = record.get("reference_only_items", [])
        missed_items = [
            {
                "text": item["text"],
                "bbox": item["bbox"],
                "is_fallback_used": False,
            }
            for item in record.get("missed_items", [])
        ]
        unused_items = ref_only + missed_items
        unused_bboxes = {
            "bbox": [[d["bbox"] for d in unused_items]],
            "text": [[d["text"] for d in unused_items]],
            "shape": [["Text" for _ in unused_items]],
            "is_fallback_used": [[d.get("is_fallback_used", True) for d in unused_items]],
        }

        valid_keynotes = record.get("keynotes", [])
        invalid_keynotes = record.get("invalid_outer_keynotes", [])
        page_validation = {
            "used_bboxes": used_bboxes,
            "unused_bboxes": unused_bboxes,
            "valid_outer_keynotes": {
                "bbox": [k["bbox"] for k in valid_keynotes],
                "text": [k["text"] for k in valid_keynotes],
                "shape": [k.get("shape", "Circle") for k in valid_keynotes],
            },
            "invalid_outer_keynotes": {
                "bbox": [k["bbox"] for k in invalid_keynotes],
                "text": [k["text"] for k in invalid_keynotes],
                "shape": [k.get("shape", "Circle") for k in invalid_keynotes],
            },
        }
        validation[str(page_index)] = page_validation
    return validation


def build_details(page_records: list[dict]) -> dict:
    pages = []
    for record in page_records:
        page = {
                "page_index": record["page_index"],
                "pdf_page_number": record["pdf_page_number"],
                "page_width": record["page_width"],
                "page_height": record["page_height"],
                "discipline": record["discipline"],
                "detections": record["detections"],
                "keynotes": record["keynotes"],
                "reference_only_items": record["reference_only_items"],
                "counts": record["counts"],
            }
        if record.get("processing_error"):
            page["processing_error"] = record["processing_error"]
        pages.append(page)
    return {
        "document": {
            "pdf_path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "page_count": len(page_records),
            "page_index_base": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Machine-generated OCR/PDF-text comparison. Not human-verified.",
        },
        "pages": pages,
    }


def build_missed(page_records: list[dict]) -> dict:
    missed = []
    uncertain = []
    for record in page_records:
        for item in record.get("missed_items", []):
            missed.append(
                {
                    "page_index": record["page_index"],
                    "pdf_page_number": record["pdf_page_number"],
                    **item,
                }
            )
        for item in record.get("uncertain_items", []):
            uncertain.append(
                {
                    "page_index": record["page_index"],
                    "pdf_page_number": record["pdf_page_number"],
                    **item,
                }
            )
    return {
        "document": {
            "pdf_path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "page_count": len(page_records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Machine-generated OCR/PDF-text comparison. Not human-verified.",
            "definition": "missed = visibly present and confidently detected, but absent from the legacy JSON for that page; uncertain = OCR evidence below the confirmation threshold.",
        },
        "summary": {
            "confirmed_misses": len(missed),
            "uncertain_detections": len(uncertain),
            "pages_with_misses": len({m['page_index'] for m in missed}),
            "pages_with_uncertain": len({u['page_index'] for u in uncertain}),
        },
        "missed": missed,
        "uncertain": uncertain,
    }


def build_summary(page_records: list[dict], validation: dict, missed_doc: dict, elapsed: float) -> dict:
    category_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    total_detections = 0
    confirmed = 0
    uncertain = 0
    reference_matched = 0
    for record in page_records:
        for det in record["detections"]:
            total_detections += 1
            category_counts[det["category"]] += 1
            source_counts[det["source"]] += 1
            if det["confirmed"]:
                confirmed += 1
            if det["is_uncertain"]:
                uncertain += 1
            if det.get("matched_reference"):
                reference_matched += 1
    return {
        "document": {
            "pdf_path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "page_count": len(page_records),
            "page_index_base": 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "disclaimer": "Machine-generated OCR/PDF-text comparison. Not human-verified.",
        },
        "page_counts": {
            "pdf_pages": len(page_records),
            "validation_pages": len(validation),
            "details_pages": len(page_records),
            "pages_with_detections": sum(1 for r in page_records if r["detections"]),
            "pages_without_detections": sum(1 for r in page_records if not r["detections"]),
            "pages_with_misses": missed_doc["summary"]["pages_with_misses"],
            "pages_with_uncertain": missed_doc["summary"]["pages_with_uncertain"],
        },
        "detections": {
            "total_important_detections": total_detections,
            "confirmed_detections": confirmed,
            "uncertain_detections": uncertain,
            "reference_matched": reference_matched,
            "confirmed_misses": missed_doc["summary"]["confirmed_misses"],
            "by_category": dict(sorted(category_counts.items())),
            "by_source": dict(sorted(source_counts.items())),
        },
        "validation": {
            "used_entries": sum(len(v["used_bboxes"]["text"][0]) for v in validation.values()),
            "unused_entries": sum(len(v["unused_bboxes"]["text"][0]) for v in validation.values()),
            "valid_keynotes": sum(len(v["valid_outer_keynotes"]["text"]) for v in validation.values()),
            "invalid_keynotes": sum(len(v["invalid_outer_keynotes"]["text"]) for v in validation.values()),
        },
    }


def assemble_pdf(page_records: list[dict], output_path: Path) -> None:
    out = fitz.open()
    for record in sorted(page_records, key=lambda r: r["page_index"]):
        width = record["page_width"]
        height = record["page_height"]
        page = out.new_page(width=width, height=height)
        image_path = record.get("annotated_image")
        if image_path and Path(image_path).exists():
            page.insert_image(page.rect, filename=image_path, keep_proportion=False)
        else:
            # Fallback: leave a blank page of the original size.
            pass
    out.save(str(output_path), garbage=3, deflate=True)
    out.close()


def validate_outputs(
    validation: dict,
    details: dict,
    missed_doc: dict,
    page_indexes: list[int],
    annotated_path: Path,
) -> dict:
    errors = []
    warnings = []
    expected_pages = {str(i) for i in page_indexes}
    expected_page_set = set(page_indexes)

    # Validation JSON structural checks.
    if set(validation.keys()) != expected_pages:
        errors.append("validation page keys do not match 0..page_count-1")
    required_page_keys = {"used_bboxes", "unused_bboxes", "valid_outer_keynotes", "invalid_outer_keynotes"}
    for page_key, page_value in validation.items():
        if set(page_value.keys()) != required_page_keys:
            errors.append(f"page {page_key}: unexpected page-level keys {sorted(page_value.keys())}")
        for block_name in ("used_bboxes", "unused_bboxes"):
            block = page_value.get(block_name, {})
            if set(block.keys()) != {"bbox", "text", "shape", "is_fallback_used"}:
                errors.append(f"page {page_key}: {block_name} has wrong keys")
                continue
            groups = [block["bbox"], block["text"], block["shape"], block["is_fallback_used"]]
            if len({len(g) for g in groups}) != 1:
                errors.append(f"page {page_key}: {block_name} group counts are not aligned")
            for gi in range(min(len(g) for g in groups)):
                n = len(groups[0][gi])
                if any(len(g[gi]) != n for g in groups):
                    errors.append(f"page {page_key}: {block_name}[{gi}] arrays are not aligned")
                for bbox in groups[0][gi]:
                    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                        errors.append(f"page {page_key}: {block_name} bbox is not four numerics: {bbox!r}")
        for block_name in ("valid_outer_keynotes", "invalid_outer_keynotes"):
            block = page_value.get(block_name, {})
            if set(block.keys()) != {"bbox", "text", "shape"}:
                errors.append(f"page {page_key}: {block_name} has wrong keys")
                continue
            if not (len(block["bbox"]) == len(block["text"]) == len(block["shape"])):
                errors.append(f"page {page_key}: {block_name} arrays are not aligned")
            for bbox in block["bbox"]:
                if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                    errors.append(f"page {page_key}: {block_name} bbox is not four numerics: {bbox!r}")

    # Details page coverage.
    detail_pages = {str(p["page_index"]) for p in details.get("pages", [])}
    if detail_pages != expected_pages:
        errors.append("details page indexes do not match 0..page_count-1")
    for page in details.get("pages", []):
        if page.get("processing_error"):
            errors.append(f"page {page.get('page_index')}: {page['processing_error']}")
        for det in page.get("detections", []):
            bbox = det.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                errors.append(f"details page {page['page_index']}: bad bbox {bbox!r}")
        for k in page.get("keynotes", []):
            bbox = k.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                errors.append(f"details page {page['page_index']}: bad keynote bbox {bbox!r}")

    # Missed page coverage.
    for item in missed_doc.get("missed", []) + missed_doc.get("uncertain", []):
        if item.get("page_index") not in expected_page_set:
            errors.append(f"missed/uncertain item has invalid page index: {item.get('page_index')!r}")

    # Annotated PDF checks.
    if not annotated_path.exists():
        errors.append("annotated PDF was not created")
    else:
        with fitz.open(annotated_path) as doc:
            if len(doc) != len(page_indexes):
                errors.append(f"annotated PDF page count {len(doc)} != {len(page_indexes)}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": {
            "validation_pages": len(validation),
            "details_pages": len(details.get("pages", [])),
            "missed_items": len(missed_doc.get("missed", [])),
            "uncertain_items": len(missed_doc.get("uncertain", [])),
            "annotated_pdf_exists": annotated_path.exists(),
        },
    }


def main() -> None:
    global PDF_PATH, OUTPUT_DIR, TEMP_DIR, _LEARNED_PREFIXES
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--temp-dir", type=Path, default=TEMP_DIR)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pages", type=str, default="", help="Comma-separated zero-based page indexes for a test run")
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing per-page JSON records in temp dir")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-page records and process only the missing pages",
    )
    args = parser.parse_args()

    PDF_PATH = args.pdf.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    TEMP_DIR = args.temp_dir.resolve()
    _LEARNED_PREFIXES = load_learned_prefixes()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with fitz.open(PDF_PATH) as doc:
        page_count = len(doc)
        page_sizes = [(float(p.rect.width), float(p.rect.height)) for p in doc]

    if args.pages:
        page_indexes = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
    else:
        page_indexes = list(range(page_count))

    reference = build_index()

    if args.reuse:
        page_records = []
        for page_index in page_indexes:
            path = TEMP_DIR / f"page_{page_index:04d}.json"
            if not path.exists():
                raise SystemExit(f"Missing reusable page record: {path}")
            page_records.append(json.loads(path.read_text()))
    else:
        page_records = []
        remaining = list(page_indexes)
        if args.resume:
            reused = []
            remaining = []
            for page_index in page_indexes:
                json_path = TEMP_DIR / f"page_{page_index:04d}.json"
                image_path = TEMP_DIR / f"page_{page_index:04d}.jpg"
                usable = json_path.exists() and (args.no_annotate or image_path.exists())
                if usable:
                    try:
                        record = json.loads(json_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        usable = False
                    else:
                        if record.get("page_index") != page_index:
                            usable = False
                if usable:
                    reused.append(record)
                else:
                    remaining.append(page_index)
            page_records.extend(reused)
            print(
                f"resume: reusing {len(reused)} page record(s), processing {len(remaining)} page(s)",
                flush=True,
            )

        # Remove stale page records for the pages we are about to process.
        for page_index in remaining:
            for suffix in (".json", ".jpg"):
                stale = TEMP_DIR / f"page_{page_index:04d}{suffix}"
                if stale.exists():
                    stale.unlink()

        tasks = [(i, args.dpi, not args.no_annotate) for i in remaining]
        if args.workers <= 1:
            _init_worker(str(PDF_PATH), reference, str(TEMP_DIR))
            for task in tasks:
                record = _worker(task)
                (TEMP_DIR / f"page_{record['page_index']:04d}.json").write_text(json.dumps(record, indent=2))
                page_records.append(record)
                print(f"page {record['page_index']:3d}/{page_count - 1} detections={record['counts']['detections']} missed={record['counts']['missed']}", flush=True)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=args.workers,
                initializer=_init_worker,
                initargs=(str(PDF_PATH), reference, str(TEMP_DIR)),
            ) as pool:
                for record in pool.imap_unordered(_worker, tasks, chunksize=1):
                    (TEMP_DIR / f"page_{record['page_index']:04d}.json").write_text(json.dumps(record, indent=2))
                    page_records.append(record)
                    print(f"page {record['page_index']:3d}/{page_count - 1} detections={record['counts']['detections']} missed={record['counts']['missed']}", flush=True)

        page_records.sort(key=lambda r: r["page_index"])

    # For a partial test run, only write the test pages, not the full document.
    full_run = len(page_indexes) == page_count and set(page_indexes) == set(range(page_count))

    validation = build_validation(page_records)
    details = build_details(page_records)
    missed_doc = build_missed(page_records)
    summary = build_summary(page_records, validation, missed_doc, time.time() - start)

    (OUTPUT_DIR / "detected_text_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (OUTPUT_DIR / "detected_text_details.json").write_text(json.dumps(details, indent=2) + "\n")
    (OUTPUT_DIR / "missed_text.json").write_text(json.dumps(missed_doc, indent=2) + "\n")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    annotated_path = OUTPUT_DIR / "annotated_full_document.pdf"
    if full_run and not args.no_annotate:
        assemble_pdf(page_records, annotated_path)
    elif args.pages:
        # Test-run annotated PDF with just the selected pages, clearly named.
        assemble_pdf(page_records, OUTPUT_DIR / "annotated_test_pages.pdf")
        annotated_path = OUTPUT_DIR / "annotated_test_pages.pdf"

    report = validate_outputs(validation, details, missed_doc, [r["page_index"] for r in page_records], annotated_path)
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== VALIDATION ===")
    print(json.dumps(report, indent=2))
    if not full_run:
        print("\nPartial run: output JSON files contain only the selected pages.")


if __name__ == "__main__":
    main()
