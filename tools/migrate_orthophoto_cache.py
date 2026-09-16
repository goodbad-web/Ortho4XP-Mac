#!/usr/bin/env python3
"""Convert and safely clean the Orthophotos JPEG cache."""

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

import O4_Imagery_Cache as CACHE  # noqa: E402


def _add_scope_arguments(parser):
    parser.add_argument(
        "--tile",
        action="append",
        required=True,
        help="Tile selector such as +34+132; repeat for multiple tiles.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider code filter; repeat to include multiple providers.",
    )
    parser.add_argument(
        "--zl",
        action="append",
        type=int,
        default=[],
        help="Zoom level filter; repeat to include multiple levels.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of conversion workers (default: 2).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: Ortho4XP_cache_reports/cache-migration-<timestamp>.json).",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Manage the Orthophotos JPEG/WebP cache without touching DDS outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert JPEG cache files to WebP.")
    _add_scope_arguments(convert)
    convert.add_argument(
        "--quality",
        type=int,
        required=True,
        help="WebP quality from 80 to 100.",
    )
    convert.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a valid WebP already exists.",
    )
    convert.add_argument(
        "--dry-run",
        action="store_true",
        help="Encode and verify without replacing any WebP file.",
    )

    cleanup = subparsers.add_parser(
        "cleanup", help="Remove JPEG only after WebP revalidation."
    )
    _add_scope_arguments(cleanup)
    cleanup.add_argument(
        "--apply",
        action="store_true",
        help="Actually remove eligible JPEG files.",
    )
    cleanup.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly confirm the destructive cleanup operation.",
    )
    cleanup.add_argument(
        "--dry-run",
        action="store_true",
        help="Keep the default non-destructive cleanup preview.",
    )
    return parser


def _progress(done, total, result):
    print(
        f"[{done}/{total}] {result['status']}: {result['jpeg']}"
        + (f" ({result['reason']})" if result.get("reason") else "")
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        print("ERROR: --workers must be at least 1.", file=sys.stderr)
        return 2
    if args.command == "cleanup" and args.apply and not args.yes:
        print(
            "ERROR: cleanup --apply requires explicit confirmation with --yes.",
            file=sys.stderr,
        )
        return 2
    if args.command == "cleanup" and args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run cannot be combined.", file=sys.stderr)
        return 2
    try:
        report = CACHE.migrate_cache(
            mode=args.command,
            tiles=args.tile,
            providers=args.provider,
            zoomlevels=args.zl,
            quality=getattr(args, "quality", None),
            workers=args.workers,
            force=getattr(args, "force", False),
            dry_run=args.dry_run,
            apply=getattr(args, "apply", False),
            confirmed=getattr(args, "yes", False),
            progress=_progress,
            report_path=args.report,
        )
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"Completed {report['mode']}: {report['files_considered']} file(s)."
    )
    print(f"JSON report: {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
