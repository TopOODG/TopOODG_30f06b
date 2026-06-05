#!/usr/bin/env python3
"""Strip Jupyter notebook outputs for Git clean/textconv filters.

Usage:
    python3 scripts/strip_ipynb_outputs.py            # read notebook JSON from stdin, write cleaned JSON to stdout
    python3 scripts/strip_ipynb_outputs.py file.ipynb # clean one or more files and print the cleaned notebook(s) to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def strip_notebook(nb: dict[str, Any]) -> dict[str, Any]:
    """Remove outputs and execution counts from code cells."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("outputs"):
            cell["outputs"] = []

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None

        # Common notebook UI metadata that can create noisy diffs.
        meta = cell.get("metadata")
        if isinstance(meta, dict):
            for key in ("collapsed", "scrolled", "ExecuteTime", "trusted"):
                meta.pop(key, None)

    nb_meta = nb.get("metadata")
    if isinstance(nb_meta, dict):
        nb_meta.pop("widgets", None)

    return nb


def cleaned_text(text: str) -> str:
    nb = json.loads(text)
    strip_notebook(nb)
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def emit_cleaned_text(text: str) -> None:
    sys.stdout.write(cleaned_text(text))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.paths:
        for raw_path in args.paths:
            path = Path(raw_path)
            emit_cleaned_text(path.read_text(encoding="utf-8"))
    else:
        emit_cleaned_text(sys.stdin.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
