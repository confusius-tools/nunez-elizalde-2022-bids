from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from rich.console import Console

from .converter import convert

CONSOLE = Console()


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


def build_upload_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the fUSI-BIDS dataset to OSF and regenerate the\n"
            "dataset index used by confusius.datasets.\n\n"
            "Requires an OSF personal access token:\n"
            "  1. Log in at https://osf.io\n"
            "  2. Go to Settings > Personal Access Tokens\n"
            "  3. Create a token with the 'osf.full_write' scope\n"
            "  4. Pass it via --token or set the OSF_TOKEN environment variable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bids-dir",
        type=Path,
        help="Local BIDS root directory to upload. Skipped with --index-only.",
    )
    parser.add_argument(
        "--project",
        default="43skw",
        help="OSF project ID (default: 43skw).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="OSF personal access token. Falls back to OSF_TOKEN env var.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Re-upload files that differ from OSF (skip identical files).",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip file upload; only regenerate and upload dataset_index.json.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=(
            "Ignore incremental index updates and rebuild dataset_index.json "
            "by scanning OSF storage."
        ),
    )
    return parser


def upload_main() -> None:
    from .upload import generate_index_with_retry, upload_dataset, upload_index

    args = build_upload_parser().parse_args()

    token = args.token or os.environ.get("OSF_TOKEN")
    if not token:
        CONSOLE.print(
            "[bold red]Error:[/] OSF token required. "
            "Pass --token or set the OSF_TOKEN environment variable.\n"
            "Get a token at https://osf.io > Settings > Personal Access Tokens."
        )
        raise SystemExit(1)

    CONSOLE.rule("[bold blue]OSF Upload Workflow")
    CONSOLE.print(f"[bold]Project[/]: [cyan]{args.project}[/]")
    if args.index_only:
        CONSOLE.print("[bold]Mode[/]: [yellow]Index only[/]")
    else:
        CONSOLE.print(f"[bold]BIDS directory[/]: [cyan]{args.bids_dir}[/]")
        CONSOLE.print(
            "[bold]Mode[/]: "
            f"[cyan]{'Update existing files' if args.update else 'Skip existing files'}[/]"
        )

    index: dict[str, dict[str, Any]] | None = None

    if not args.index_only:
        if args.bids_dir is None:
            CONSOLE.print(
                "[bold red]Error:[/] --bids-dir is required unless --index-only is set."
            )
            raise SystemExit(1)
        CONSOLE.rule("[bold blue]Step 1/3: Upload Dataset Files")
        index = upload_dataset(args.bids_dir, token, args.project, update=args.update)

    CONSOLE.rule("[bold blue]Step 2/3: Build Dataset Index")
    if args.rebuild_index or args.index_only or index is None:
        CONSOLE.print("[cyan]Generating index from OSF storage...[/]")
        index = generate_index_with_retry(token, args.project)
    else:
        CONSOLE.print("[cyan]Using incrementally updated index from upload run...[/]")
    CONSOLE.print(f"[green]Index contains {len(index)} files.[/]")

    CONSOLE.rule("[bold blue]Step 3/3: Upload Dataset Index")
    CONSOLE.print("[cyan]Uploading dataset_index.json to OSF...[/]")
    upload_index(index, token, args.project)
    CONSOLE.rule("[bold green]Upload Finished")


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
