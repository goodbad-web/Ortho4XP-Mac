"""Ortho4XP integration boundary for the embedded headless MUXP engine."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
import shutil

import O4_File_Names as FNAMES
import O4_UI_Utils as UI
from muxp_engine import MuxpApplyError, apply_muxp_to_dsf, write_manifest


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enabled(tile):
    return bool(getattr(tile, "muxp_enabled", False))


def apply_to_staged_dsf(tile, dsf_path):
    """Apply enabled tile MUXP updates to the DSF transaction candidate."""
    if not enabled(tile):
        return None
    import O4_Config_Utils as CFG

    project_root = os.path.abspath(FNAMES.Ortho4XP_dir)
    muxp_folder = getattr(CFG, "muxp_folder", "MUXP")
    UI.vprint(
        1,
        "MUXP: searching {} for tile {}".format(
            os.path.join(project_root, muxp_folder), FNAMES.short_latlon(tile.lat, tile.lon)
        ),
    )
    try:
        result = apply_muxp_to_dsf(
            dsf_path=os.path.abspath(dsf_path),
            project_root=project_root,
            muxp_folder=muxp_folder,
            tile=FNAMES.short_latlon(tile.lat, tile.lon),
            allow_side_effects=bool(getattr(tile, "muxp_allow_side_effects", False)),
            cancel_checker=UI.is_cancel_requested,
            xplane_root=(
                os.path.dirname(os.path.abspath(CFG.custom_scenery_dir))
                if getattr(CFG, "custom_scenery_dir", "")
                else project_root
            ),
        )
    except MuxpApplyError as error:
        UI.vprint(
            0,
            UI.ui_text(
                "ERROR: MUXP processing failed: {}".format(error),
                "エラー: MUXP処理に失敗しました: {}".format(error),
            ),
        )
        raise
    tile._muxp_result = result
    if result.manifest.get("result") == "no_match":
        UI.vprint(
            1,
            UI.ui_text(
                "MUXP: no matching files; continuing normal tile build.",
                "MUXP: 対象ファイルがないため、通常のタイルビルドを継続します。",
            ),
        )
    else:
        UI.vprint(
            1,
            "MUXP: {} file result={} changed={}".format(
                len(result.manifest.get("files", [])),
                result.manifest.get("result"),
                result.changed,
            ),
        )
    return result


def backup_existing_dsf(tile, dsf_path):
    """Keep an immutable generation before MUXP replaces the active DSF."""
    result = getattr(tile, "_muxp_result", None)
    if result is None or not result.changed or not os.path.isfile(dsf_path):
        return None
    backup_dir = os.path.join(tile.build_dir, "muxp_backups")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = os.path.join(
        backup_dir,
        "{}.{}.{}.dsf".format(
            os.path.basename(dsf_path), stamp, _sha256(dsf_path)[:16]
        ),
    )
    shutil.copy2(dsf_path, destination)
    return destination


def publish_manifest(tile):
    result = getattr(tile, "_muxp_result", None)
    if result is None:
        return None
    manifest = dict(result.manifest)
    manifest["published_dsf"] = os.path.join(
        "Earth nav data", FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf"
    )
    return write_manifest(tile.build_dir, manifest)
