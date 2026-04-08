from __future__ import annotations

import argparse
from pathlib import Path

from .converter import convert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Nunez-Elizalde 2022 fUSI recordings to BIDS.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source Subjects directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output BIDS root directory.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subset of subjects (e.g. CR017 CR019).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NIfTI outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan conversion and write a manifest without generating NIfTI files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert(
        src=args.src,
        out=args.out,
        subjects=args.subjects,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    if summary.planned_runs == 0:
        print("No runs found with current filters.")
        return

    if summary.dry_run:
        print(f"Dry-run complete: {summary.planned_runs} runs planned.")
        print(f"Manifest: {summary.manifest_path}")
        return

    print(
        f"Conversion complete: {summary.converted_runs} converted, "
        f"{summary.skipped_runs} skipped. Manifest: {summary.manifest_path}"
    )
