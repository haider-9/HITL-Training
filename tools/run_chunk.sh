#!/usr/bin/env bash
# Compatibility entry point: process one explicit zero-based page only.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 1 ]]; then
    echo "usage: tools/run_chunk.sh PAGE_INDEX" >&2
    echo "This runner no longer starts multi-page OCR jobs." >&2
    exit 2
fi

exec .venv/bin/python tools/annotate_reference_pages.py --page "$1"
