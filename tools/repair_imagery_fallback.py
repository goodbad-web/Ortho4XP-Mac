#!/usr/bin/env python3
"""Repair imagery files produced by the obsolete fixed-ZL16 fallback.

The command is deliberately log-scoped.  It never scans or rewrites the
whole imagery cache, and it does not touch terrain, DSF, or scenery metadata.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

from PIL import Image  # noqa: E402

import O4_Config_Utils as CFG  # noqa: E402
import O4_File_Names as FNAMES  # noqa: E402
import O4_Imagery_Cache as CACHE  # noqa: E402
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_UI_Utils as UI  # noqa: E402


_REBUILT_LINE = re.compile(
    r"\[Quality Check\]\s+Rebuilt\s+"
    r"(?P<target>\S+\.(?:jpg|webp))\s+from\s+"
    r"(?P<parent>\S+\.(?:jpg|webp))\."
)
_IMAGE_NAME = re.compile(
    r"(?P<y>-?\d+)_(?P<x>-?\d+)_(?P<provider>.+?)"
    r"(?P<zl>1[5-9]|2[0-9])\."
    r"(?P<ext>jpg|webp)$"
)
_TILE_DIR = re.compile(r"zOrtho4XP_(?P<lat>[+-]\d{1,3})(?P<lon>[+-]\d{1,3})$")


def _parse_image_name(name):
    match = _IMAGE_NAME.fullmatch(name)
    if not match:
        return None
    return {
        "name": name,
        "til_y_top": int(match.group("y")),
        "til_x_left": int(match.group("x")),
        "provider_code": match.group("provider"),
        "zoomlevel": int(match.group("zl")),
        "extension": match.group("ext"),
    }


def _tile_coordinates(log_path):
    for part in reversed(Path(log_path).resolve().parts):
        match = _TILE_DIR.fullmatch(part)
        if match:
            return int(match.group("lat")), int(match.group("lon"))
    return None


def extract_targets(log_paths):
    """Extract and deduplicate only explicit old fallback log entries."""
    targets = {}
    for log_path_value in log_paths:
        log_path = Path(log_path_value).resolve()
        coordinates = _tile_coordinates(log_path)
        if coordinates is None:
            raise ValueError(f"could not determine tile from log path: {log_path}")
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"could not read log {log_path}: {error}") from error
        for line_number, line in enumerate(lines, 1):
            match = _REBUILT_LINE.search(line)
            if not match:
                continue
            target = _parse_image_name(match.group("target"))
            parent = _parse_image_name(match.group("parent"))
            if target is None or parent is None:
                continue
            if target["provider_code"] != parent["provider_code"]:
                continue
            if target["zoomlevel"] <= 16 or parent["zoomlevel"] >= target["zoomlevel"]:
                continue
            lat, lon = coordinates
            target.update(
                {
                    "parent_name_in_log": parent["name"],
                    "log_path": str(log_path),
                    "line_number": line_number,
                    "tile_lat": lat,
                    "tile_lon": lon,
                    "tile_build_dir": str(log_path.parent),
                }
            )
            key = (str(log_path.parent), target["name"])
            targets[key] = target
    return sorted(
        targets.values(),
        key=lambda item: (
            item["tile_build_dir"],
            item["til_y_top"],
            item["til_x_left"],
            item["provider_code"],
            item["zoomlevel"],
        ),
    )


def _load_tile(target):
    tile = CFG.Tile(target["tile_lat"], target["tile_lon"], "")
    if not tile.read_from_config():
        raise ValueError(
            f"could not load tile configuration for "
            f"{FNAMES.short_latlon(target['tile_lat'], target['tile_lon'])}"
        )
    return tile


def _initialize_imagery():
    IMG.initialize_extents_dict()
    IMG.initialize_color_filters_dict()
    IMG.initialize_providers_dict()
    IMG.initialize_combined_providers_dict()


def _image_is_4096(path):
    if not path or not os.path.isfile(path):
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == (4096, 4096)
    except Exception:
        return False


def _target_context(target, tile):
    provider_code = target["provider_code"]
    if provider_code not in IMG.providers_dict:
        raise ValueError(f"unknown provider {provider_code}")
    provider = IMG.providers_dict[provider_code]
    file_dir = os.path.abspath(
        FNAMES.jpeg_file_dir_from_attributes(
            tile.lat, tile.lon, target["zoomlevel"], provider
        )
    )
    target_path = os.path.join(file_dir, target["name"])
    dds_path = os.path.abspath(os.path.join(
        tile.build_dir,
        "textures",
        FNAMES.dds_file_name_from_attributes(
            target["til_x_left"],
            target["til_y_top"],
            target["zoomlevel"],
            provider_code,
        ),
    ))
    return file_dir, target_path, dds_path


def _parent_candidates(target, file_dir):
    candidates = []
    for parent_zl in range(target["zoomlevel"] - 1, 15, -1):
        crop_box = IMG._parent_crop_box(
            target["til_x_left"],
            target["til_y_top"],
            target["zoomlevel"],
            parent_zl,
        )
        if crop_box is None:
            continue
        parent_x_left, parent_y_top = crop_box[:2]
        parent_dir = IMG._parent_file_dir(
            file_dir, target["provider_code"], parent_zl
        )
        cache_path = IMG._valid_parent_cache_path(
            parent_dir,
            parent_x_left,
            parent_y_top,
            parent_zl,
            target["provider_code"],
        )
        candidates.append(
            {
                "zoomlevel": parent_zl,
                "til_x_left": parent_x_left,
                "til_y_top": parent_y_top,
                "cache_path": cache_path,
                "available": bool(cache_path),
            }
        )
    return candidates


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_entry(path, backup_root, index):
    path = Path(path)
    existed = path.is_file()
    entry = {
        "path": str(path),
        "exists": existed,
        "sha256": _sha256(path) if existed else None,
        "backup_path": None,
    }
    if existed:
        backup_path = backup_root / "files" / f"{index:04d}-{path.name}"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        entry["backup_path"] = str(backup_path)
    return entry


def _restore_entry(entry):
    path = Path(entry["path"])
    backup_path = entry.get("backup_path")
    if entry.get("exists"):
        if not backup_path or not Path(backup_path).is_file():
            raise OSError(f"backup is missing for {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, path)
    elif path.exists():
        path.unlink()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name("." + path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _configure_direct_conversion(tile):
    previous = {
        name: getattr(UI, name, None)
        for name in (
            "red_flag",
            "defer_gpu_batch",
            "defer_fp8_batch",
            "preserve_batch_inputs",
            "dds_converter",
            "dds_format",
            "use_gpu_acceleration",
            "use_gpu_for_color_filters",
        )
    }
    UI.red_flag = False
    UI.defer_gpu_batch = False
    UI.defer_fp8_batch = False
    # Preserve masks and intermediate inputs until the target transaction has
    # been validated; this is still convert_texture's normal conversion path.
    UI.preserve_batch_inputs = True
    UI.dds_converter = getattr(tile, "dds_converter", UI.dds_converter)
    UI.dds_format = getattr(tile, "dds_format", UI.dds_format)
    UI.use_gpu_acceleration = getattr(
        tile, "use_gpu_acceleration", UI.use_gpu_acceleration
    )
    UI.use_gpu_for_color_filters = getattr(
        tile, "use_gpu_for_color_filters", UI.use_gpu_for_color_filters
    )
    return previous


def _restore_direct_conversion(previous):
    for name, value in previous.items():
        setattr(UI, name, value)


def _repair_one(target, tile, backup_entries, parent_download_cache=None):
    file_dir, target_path, dds_path = _target_context(target, tile)
    rebuilt, image, parent_zl, parent_path = IMG._rebuild_from_parent(
        file_dir,
        target["til_x_left"],
        target["til_y_top"],
        target["zoomlevel"],
        target["provider_code"],
        parent_download_cache=parent_download_cache,
    )
    if not rebuilt:
        return {
            **target,
            "status": "failed",
            "reason": "all_parent_fallbacks_failed",
            "target_path": target_path,
            "dds_path": dds_path,
        }

    target_backup = next(
        entry for entry in backup_entries if entry["path"] == target_path
    )
    dds_backup = next(entry for entry in backup_entries if entry["path"] == dds_path)
    try:
        os.makedirs(file_dir, exist_ok=True)
        IMG.save_imagery_cache_image(image, target_path, jpeg_quality=90)
        if not _image_is_4096(target_path):
            raise ValueError("rebuilt JPEG failed 4096x4096 validation")
        if not IMG.convert_texture(
            tile,
            target["til_x_left"],
            target["til_y_top"],
            target["zoomlevel"],
            target["provider_code"],
            type="dds",
        ):
            raise ValueError("DDS conversion failed")
        valid, error = IMG.validate_dds_file(
            dds_path,
            expected_dimensions=(4096, 4096),
            require_mipmaps=True,
        )
        if not valid:
            raise ValueError(error)
        return {
            **target,
            "status": "repaired",
            "parent_zl": parent_zl,
            "parent_path": parent_path,
            "target_path": target_path,
            "dds_path": dds_path,
        }
    except Exception as error:
        rollback_errors = []
        for entry in (target_backup, dds_backup):
            try:
                _restore_entry(entry)
            except Exception as rollback_error:
                rollback_errors.append(str(rollback_error))
        reason = str(error)
        if rollback_errors:
            reason += "; rollback failed: " + ", ".join(rollback_errors)
        return {
            **target,
            "status": "rolled_back",
            "reason": reason,
            "parent_zl": parent_zl,
            "parent_path": parent_path,
            "target_path": target_path,
            "dds_path": dds_path,
        }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Repair old fixed-ZL16 imagery fallback results from explicit logs."
    )
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        type=Path,
        help="Build log to inspect; repeat for each tile log.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace validated JPEG and DDS targets.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly confirm --apply.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the explicit log scope without changing cache files (default).",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Backup directory for --apply (default: a persistent temporary directory).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: a temporary report path).",
    )
    return parser


def _default_report_path():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.gettempdir()) / f"ortho4xp-imagery-repair-{stamp}.json"


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.apply and not args.yes:
        print("ERROR: --apply requires explicit confirmation with --yes.", file=sys.stderr)
        return 2
    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run cannot be combined.", file=sys.stderr)
        return 2

    try:
        targets = extract_targets(args.log)
        _initialize_imagery()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    mode = "apply" if args.apply else "dry-run"
    report = {
        "version": 1,
        "mode": mode,
        "logs": [str(Path(path).resolve()) for path in args.log],
        "targets_extracted": len(targets),
        "targets": [],
        "backup_dir": None,
    }

    contexts = []
    for target in targets:
        try:
            tile = _load_tile(target)
            file_dir, target_path, dds_path = _target_context(target, tile)
            candidates = _parent_candidates(target, file_dir)
            context = {
                "target": target,
                "tile": tile,
                "file_dir": file_dir,
                "target_path": target_path,
                "dds_path": dds_path,
                "candidates": candidates,
            }
            contexts.append(context)
            if not args.apply:
                report["targets"].append(
                    {
                        **target,
                        "status": "would_repair",
                        "target_path": target_path,
                        "dds_path": dds_path,
                        "parent_candidates": candidates,
                    }
                )
        except Exception as error:
            report["targets"].append(
                {**target, "status": "failed", "reason": str(error)}
            )

    if args.apply:
        backup_dir = args.backup_dir or Path(
            tempfile.mkdtemp(prefix="ortho4xp-imagery-repair-")
        )
        backup_dir.mkdir(parents=True, exist_ok=True)
        report["backup_dir"] = str(backup_dir.resolve())
        paths = []
        for context in contexts:
            paths.extend((context["target_path"], context["dds_path"]))
        backup_entries = []
        try:
            for index, path in enumerate(dict.fromkeys(paths)):
                backup_entries.append(_backup_entry(path, backup_dir, index))
            manifest = {
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entries": backup_entries,
            }
            manifest_path = backup_dir / "manifest.json"
            _write_json(manifest_path, manifest)
            report["manifest_path"] = str(manifest_path.resolve())
        except Exception as error:
            print(f"ERROR: could not create backup: {error}", file=sys.stderr)
            return 2

        previous_retry_state = {
            name: getattr(IMG, name)
            for name in (
                "check_tms_response",
                "max_connect_retries",
                "max_baddata_retries",
                "http_timeout",
            )
        }
        IMG.check_tms_response = False
        IMG.max_connect_retries = 1
        IMG.max_baddata_retries = 1
        IMG.http_timeout = min(float(IMG.http_timeout), 3.0)
        parent_download_cache = {}
        try:
            for context in contexts:
                target = context["target"]
                entries = [
                    entry
                    for entry in backup_entries
                    if entry["path"]
                    in (context["target_path"], context["dds_path"])
                ]
                previous_ui_state = _configure_direct_conversion(context["tile"])
                try:
                    result = _repair_one(
                        target,
                        context["tile"],
                        entries,
                        parent_download_cache=parent_download_cache,
                    )
                finally:
                    _restore_direct_conversion(previous_ui_state)
                report["targets"].append(result)
        finally:
            for name, value in previous_retry_state.items():
                setattr(IMG, name, value)

    report_path = args.report or _default_report_path()
    try:
        _write_json(report_path, report)
    except OSError as error:
        print(f"ERROR: could not write report: {error}", file=sys.stderr)
        return 2

    counts = {}
    for result in report["targets"]:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(f"Extracted {len(targets)} target(s); mode={mode}; results={counts}.")
    print(f"JSON report: {report_path.resolve()}")
    if report.get("manifest_path"):
        print(f"Backup manifest: {report['manifest_path']}")
    return 0 if not any(status in counts for status in ("failed", "rolled_back")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
