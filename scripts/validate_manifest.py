#!/usr/bin/env python3
"""Validate a diagnostic JSONL manifest; never generate samples or results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcse.manifest import validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL manifest; audio paths are relative to this file.")
    parser.add_argument("--check-audio", action="store_true", help="Read WAV files and check PCM, channels, rate and length.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable validation report.")
    args = parser.parse_args()
    report = validate_manifest(args.manifest, check_audio=args.check_audio)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"{'PASS' if report.valid else 'FAIL'}: {report.manifest}")
        print(f"Rows: {report.row_count}; schema-valid records: {len(report.records)}; audio checked: {report.audio_checked}")
        for level, issues in (("ERROR", report.errors), ("WARNING", report.warnings)):
            for issue in issues:
                location = f"line {issue.line}" if issue.line is not None else "manifest"
                print(f"{level} [{issue.code}] {location}: {issue.message}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
