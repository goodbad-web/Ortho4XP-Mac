#!/usr/bin/env python3
"""Manage local GSI DEM archives and build Ortho4XP-compatible rasters."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import O4_GSI_DEM_Utils as GSI  # noqa: E402


def _paths_from_args(args):
    defaults = GSI.default_paths(Path(__file__).resolve().parent)
    input_arg = getattr(args, "input_dir", None)
    output_arg = getattr(args, "output_dir", None)
    input_dir = Path(input_arg).expanduser() if input_arg else defaults.input_dir
    output_dir = Path(output_arg).expanduser() if output_arg else defaults.output_dir
    return input_dir, output_dir


def _progress(stage, completed, total, message):
    if total:
        print(f"[{stage}] {completed}/{total}: {message}", flush=True)
    else:
        print(f"[{stage}] {message}", flush=True)


def _add_common_paths(parser):
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Managed GSI input directory (default: Elevation_data/GSI/input)",
    )


def _add_build_arguments(parser):
    _add_common_paths(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Generated DEM directory (default: Elevation_data/GSI/output)",
    )
    parser.add_argument(
        "--mesh-code",
        action="append",
        dest="mesh_codes",
        help="Third-level Japanese mesh code; repeat for batch processing",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="One geographic rectangle in decimal degrees",
    )
    parser.add_argument(
        "--resolution",
        choices=("auto", "1m", "5m", "10m"),
        default="auto",
        help="Output resolution; auto keeps the highest available source",
    )
    parser.add_argument(
        "--hgt-tile",
        action="append",
        dest="hgt_tiles",
        help="Explicit one-degree HGT tile such as N34E132; repeatable",
    )
    parser.add_argument(
        "--make-vrt",
        action="store_true",
        help="Create gsi_dem.vrt from successful GeoTIFF outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs atomically",
    )
    parser.add_argument(
        "--source-crs",
        help="Explicit CRS override for input XML with an unknown CRS",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Import, catalog, and convert local GSI DEM ZIP archives."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import", help="Copy and classify GSI ZIPs into the managed input area"
    )
    import_parser.add_argument(
        "--from",
        dest="source_dir",
        type=Path,
        required=True,
        help="Source directory containing downloaded ZIP archives",
    )
    _add_common_paths(import_parser)

    scan_parser = subparsers.add_parser(
        "scan", help="Rebuild the managed input catalog"
    )
    _add_common_paths(scan_parser)

    build_parser = subparsers.add_parser(
        "build", help="Build GeoTIFF and optional HGT outputs"
    )
    _add_build_arguments(build_parser)
    return parser


def _validate_build_args(args):
    if not args.mesh_codes and not args.bbox and not args.hgt_tiles:
        raise GSI.GSIError("build requires --mesh-code, --bbox, or --hgt-tile")
    if args.bbox and args.mesh_codes:
        # Both are useful in a batch, so this is intentionally allowed.
        pass


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dir, output_dir = _paths_from_args(args)
    cancel_event = threading.Event()

    def request_cancel(signum, frame):
        del signum, frame
        print("Cancellation requested; waiting for the current archive...", flush=True)
        cancel_event.set()

    signal.signal(signal.SIGINT, request_cancel)
    signal.signal(signal.SIGTERM, request_cancel)

    try:
        if args.command == "import":
            result = GSI.import_gsi_archives(
                args.source_dir,
                input_dir,
                progress=_progress,
                cancel_event=cancel_event,
            )
            print(
                f"Imported={result.imported} duplicates={result.skipped_duplicates} "
                f"quarantined={result.quarantined}"
            )
            if result.product_counts:
                print(f"Products: {result.product_counts}")
            if result.date_counts:
                print(f"Dates: {result.date_counts}")
            print(f"Catalog: {result.catalog_path}")
            for message in result.messages:
                print(f"WARNING: {message}")
            if result.cancelled:
                return 130
            return 0 if not result.failed else 1

        if args.command == "scan":
            result = GSI.scan_gsi_input(
                input_dir,
                write_catalog=True,
                progress=_progress,
                cancel_event=cancel_event,
            )
            print(
                f"Ready={result.ready} duplicates={result.duplicates} "
                f"quarantined={result.quarantined} invalid={result.invalid} "
                f"modified={result.modified} missing={result.missing}"
            )
            print(f"ZIPs={result.zip_count} products={result.product_counts} dates={result.date_counts}")
            print(f"Selection={result.selection_counts}")
            print(f"Catalog: {result.catalog_path}")
            return 0 if not (result.invalid or result.modified or result.missing) else 1

        _validate_build_args(args)
        options = GSI.GSIOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            mesh_codes=tuple(args.mesh_codes or ()),
            bbox=tuple(args.bbox) if args.bbox else None,
            resolution=args.resolution,
            make_vrt=args.make_vrt,
            hgt_tiles=tuple(args.hgt_tiles or ()),
            overwrite=args.overwrite,
            source_crs=args.source_crs,
        )
        result = GSI.build_gsi_dem(
            options,
            progress=_progress,
            cancel_event=cancel_event,
        )
        for output in result.outputs:
            print(f"Written: {output}")
        if result.vrt:
            print(f"VRT: {result.vrt}")
        if result.manifest:
            print(f"Manifest: {result.manifest}")
        if result.recommended_custom_dem:
            print(f"custom_dem: {result.recommended_custom_dem}")
        for failure in result.failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        if result.cancelled:
            return 130
        if result.failures or not result.outputs:
            return 1
        return 0
    except GSI.GSICancelled:
        return 130
    except GSI.GSIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
