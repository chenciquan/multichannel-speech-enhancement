#!/usr/bin/env python3
"""Freeze inputs and build a small explicit-geometry, six-channel discovery set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcse.simulation import build_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="Selected mono16k source JSONL; no output-based selection")
    parser.add_argument("--noises", type=Path, required=True, help="Exactly two registered mono16k noise recordings")
    parser.add_argument("--geometry-json", type=Path, required=True, help="Explicit coordinates, source directions and array-domain label")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist; no overwrite or automatic cleanup")
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    try:
        summary = build_diagnostics(args.sources, args.noises, args.geometry_json,
                                    args.output_dir, seed=args.seed)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error),
                          "notice": "Any partial output is left intact for inspection; it is not a completed dataset."}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
