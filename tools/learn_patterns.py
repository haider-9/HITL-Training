#!/usr/bin/env python3
"""Build a transparent detection plan from the existing reference labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from build_reference_index import build_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "pattern_learning" / "learned_patterns.json"


def learn_auxiliary_patterns() -> dict[str, Any]:
    """Summarize OCR, detector, and schedule JSONs as supporting evidence."""
    json_dir = PROJECT_ROOT / "source" / "json"
    ocr_files = sorted(json_dir.glob("single_page_ocr_word_dict_*.json"))
    raw_detection_files = sorted(json_dir.glob("single_page_raw_detections_*.json"))
    schedule_files = sorted(json_dir.glob("single_page_schedule_data_*.json"))
    human_feedback_path = PROJECT_ROOT / "outputs" / "one_page_ocr" / "human_learning.json"

    ocr_words: Counter[str] = Counter()
    confidence_total = 0.0
    confidence_count = 0
    ocr_pages = 0
    for path in ocr_files:
        data = json.loads(path.read_text())
        for words in data.values():
            if not isinstance(words, list):
                continue
            ocr_pages += 1
            for word in words:
                if not isinstance(word, dict):
                    continue
                text = str(word.get("text", "")).strip().upper()
                if text:
                    ocr_words[text] += 1
                confidence = word.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidence_total += float(confidence)
                    confidence_count += 1

    detector_classes: Counter[str] = Counter()
    detector_counts: Counter[str] = Counter()
    detector_pages = 0
    for path in raw_detection_files:
        data = json.loads(path.read_text())
        for detections in data.values():
            if not isinstance(detections, list):
                continue
            detector_pages += 1
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                name = detection.get("name")
                if isinstance(name, str) and name.strip():
                    detector_classes[name.strip()] += 1
                confidence = detection.get("confidence")
                if isinstance(confidence, (int, float)):
                    detector_counts["confidence_values"] += 1

    schedule_blocks: Counter[str] = Counter()
    schedule_pages = 0
    for path in schedule_files:
        data = json.loads(path.read_text())
        for page in data.values():
            if not isinstance(page, dict):
                continue
            schedule_pages += 1
            schedule_blocks.update(str(key) for key in page)

    human_feedback = []
    if human_feedback_path.exists():
        data = json.loads(human_feedback_path.read_text())
        if isinstance(data, dict) and isinstance(data.get("feedback"), list):
            human_feedback = [item for item in data["feedback"] if isinstance(item, dict)]
    human_decisions = Counter(str(item.get("decision", "unknown")) for item in human_feedback)
    human_categories = Counter(str(item.get("category", "other")) for item in human_feedback)

    return {
        "files": [
            path.name
            for path in [*ocr_files, *raw_detection_files, *schedule_files]
            + ([human_feedback_path] if human_feedback_path.exists() else [])
        ],
        "ocr": {
            "pages": ocr_pages,
            "word_count": sum(ocr_words.values()),
            "unique_words": len(ocr_words),
            "mean_confidence": round(confidence_total / confidence_count, 6)
            if confidence_count
            else None,
            "frequent_words": dict(ocr_words.most_common(50)),
        },
        "visual_detectors": {
            "pages": detector_pages,
            "class_counts": dict(sorted(detector_classes.items())),
            "confidence_values": detector_counts["confidence_values"],
        },
        "schedule": {
            "pages": schedule_pages,
            "block_counts": dict(sorted(schedule_blocks.items())),
        },
        "human_feedback": {
            "feedback_items": len(human_feedback),
            "decisions": dict(sorted(human_decisions.items())),
            "categories": dict(sorted(human_categories.items())),
        },
    }


def build_patterns() -> dict:
    index = build_index()
    category_counts: Counter[str] = Counter()
    discipline_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    tag_prefix_counts: Counter[str] = Counter()
    prefixes_by_discipline: dict[str, Counter[str]] = defaultdict(Counter)
    geometry_by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    examples: dict[str, set[str]] = defaultdict(set)
    pages_by_category: dict[str, set[int]] = defaultdict(set)
    reference_items = 0

    for page_index, page in sorted(index.items()):
        for item in page.get("expected", []):
            reference_items += 1
            category = item.get("category", item.get("discipline", "reference"))
            discipline = item.get("discipline", "unknown")
            pattern = item.get("pattern", item.get("reference_block", "reference_item"))
            text = item.get("text") or item.get("match_key", "")
            category_counts[category] += 1
            discipline_counts[discipline] += 1
            pattern_counts[pattern] += 1
            match = re.match(r"([A-Z]{1,6})[- ]?\d", str(text).upper())
            if match:
                tag_prefix_counts[match.group(1)] += 1
                prefixes_by_discipline[discipline][match.group(1)] += 1
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                width = max(0.0, float(bbox[2]) - float(bbox[0]))
                height = max(0.0, float(bbox[3]) - float(bbox[1]))
                if width and height:
                    geometry_by_category[category].append(
                        {
                            "width": width,
                            "height": height,
                            "area": width * height,
                            "aspect_ratio": width / height,
                        }
                    )
            pages_by_category[category].add(page_index)
            if text and len(examples[pattern]) < 12:
                examples[pattern].add(text)

    auxiliary = learn_auxiliary_patterns()

    geometry_profiles = {}
    for category, values in sorted(geometry_by_category.items()):
        geometry_profiles[category] = {
            "sample_count": len(values),
            "mean": {
                field: round(mean(item[field] for item in values), 4)
                for field in values[0]
            },
            "median": {
                field: round(median(item[field] for item in values), 4)
                for field in values[0]
            },
            "min": {
                field: round(min(item[field] for item in values), 4)
                for field in values[0]
            },
            "max": {
                field: round(max(item[field] for item in values), 4)
                for field in values[0]
            },
        }

    return {
        "schema_version": 2,
        "source": "build_reference_index.build_index",
        "purpose": "Transparent detection plan learned from all supplied JSON evidence and human review labels.",
        "input_files": sorted(
            {
                source
                for page in index.values()
                for source in page.get("sources", [])
            }
            | set(auxiliary["files"])
        ),
        "reference_pages": len(index),
        "reference_items": reference_items,
        "what_to_detect": [
            "mechanical, plumbing, and panel tags",
            "door and panel references",
            "keynote numbers when their geometry matches the learned gate",
            "reference-only items that OCR should recover or flag as missed",
            "visual detector classes such as tags, callouts, titles, and plan areas",
            "schedule blocks that identify doors, keynotes, and other reference regions",
        ],
        "detection_rules": {
            "reference_match": "match normalized OCR text to a reference item on the same PDF page",
            "strict_detection": "retain matched references plus clean known-prefix equipment tags",
            "geometry": "retain OCR bounding boxes and compare them with reference boxes",
            "uncertain": "report plausible OCR text that cannot be confidently classified",
        },
        "counts": {
            "by_category": dict(sorted(category_counts.items())),
            "by_discipline": dict(sorted(discipline_counts.items())),
            "by_pattern": dict(sorted(pattern_counts.items())),
            "by_tag_prefix": dict(sorted(tag_prefix_counts.items())),
        },
        "learned_features": {
            "tag_prefixes_by_discipline": {
                discipline: dict(sorted(prefixes.items()))
                for discipline, prefixes in sorted(prefixes_by_discipline.items())
            },
            "reference_geometry_by_category": geometry_profiles,
        },
        "auxiliary_learning": auxiliary,
        "pages_by_category": {
            category: sorted(pages)
            for category, pages in sorted(pages_by_category.items())
        },
        "examples_by_pattern": {
            pattern: sorted(values)
            for pattern, values in sorted(examples.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    patterns = build_patterns()
    output.write_text(json.dumps(patterns, indent=2) + "\n")
    print(json.dumps({"output": str(output), "reference_pages": patterns["reference_pages"], "reference_items": patterns["reference_items"]}, indent=2))


if __name__ == "__main__":
    main()
