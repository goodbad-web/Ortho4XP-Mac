"""Transactional, strict, headless MUXP application for Ortho4XP."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import resource
import shutil
import time
from typing import Callable, Iterable

from . import MUXP_FILE_DEFS
from . import _legacy
from .muxp_file import readMuxpFile, validate_muxp


LOG_NAME = "Ortho4XP.MUXP"
log = logging.getLogger(LOG_NAME)

SIDE_EFFECT_COMMANDS = frozenset(
    {"unflatten_default_apt", "extract_mesh_to_file", "insert_mesh_from_file"}
)
_TILE_RE = re.compile(r"^([+-]\d+)([+-]\d+)$")


class MuxpApplyError(RuntimeError):
    """Raised when a strict MUXP update cannot be safely applied."""


@dataclass(frozen=True)
class MuxpFile:
    path: str
    relative_path: str
    sha256: str
    update: dict


@dataclass(frozen=True)
class MuxpApplyResult:
    manifest: dict
    changed: bool


class _TimingQueue:
    """Small queue-compatible sink used to measure command boundaries."""

    def __init__(self, cancelled: Callable[[], bool] | None):
        self.cancelled = cancelled or (lambda: False)
        self.timings = []
        self._active = None

    def put(self, item):
        if self.cancelled():
            raise MuxpApplyError("MUXP processing cancelled at command boundary")
        if isinstance(item, tuple) and item and item[0] == "status":
            message = str(item[1])
            if message.startswith("Processing "):
                now = time.perf_counter()
                if self._active is not None:
                    self._active["elapsed_ms"] = (now - self._active.pop("_started")) * 1000.0
                    self.timings.append(self._active)
                label = message[len("Processing "):].split("\n", 1)[0]
                self._active = {
                    "command": label,
                    "elapsed_ms": 0.0,
                    "_started": now,
                }

    def finish(self):
        if self._active is not None:
            self._active["elapsed_ms"] = (
                time.perf_counter() - self._active.pop("_started")
            ) * 1000.0
            self.timings.append(self._active)
            self._active = None


class _HeadlessMuxp(_legacy.muxpGUI):
    """Reuse the upstream command implementation without constructing Tk."""

    def __init__(self, dsf, muxp_path, xpfolder, allow_side_effects, cancelled):
        self.runfile = muxp_path
        self.xpfolder = xpfolder
        self.muxpfolder = os.path.join(xpfolder, "Custom Scenery", "zzzz_MUXP_default_mesh_updates")
        self.kmlExport = 0
        self.current_action = "read"
        self.dsf = dsf
        self.dsf_sceneryPack = "Custom Scenery/zOrtho4XP"
        self.conflictStrategy = "IGNORE"
        self.activatePack = 0
        self.global_scenery_pack = "Global Scenery/X-Plane 12 Global Scenery"
        self.autoOrtho4XP = 0
        self.performance_backend = "python"
        self.performance_profiling = 1
        self.performance_workers = 1
        self.allow_side_effects = bool(allow_side_effects)
        self.msg_queue = _TimingQueue(cancelled)

    def request_ui(self, action, *args):
        if action == "display_note":
            log.warning("%s", args[0] if args else "MUXP note")
            return None
        if action == "warn_window" and self.allow_side_effects:
            return "OK"
        raise MuxpApplyError(
            "MUXP requested interactive or side-effect action '{}'; "
            "enable muxp_allow_side_effects to permit it".format(action)
        )


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tile(value: str) -> str:
    match = _TILE_RE.fullmatch(str(value).strip())
    if not match:
        raise MuxpApplyError("invalid MUXP tile '{}'; expected +lat+lon".format(value))
    latitude, longitude = int(match.group(1)), int(match.group(2))
    return "{:+03d}{:+04d}".format(latitude, longitude)


def _version(value) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MuxpApplyError("invalid MUXP version '{}'".format(value)) from exc


def _source_is_ortho4xp(source_dsf: str) -> bool:
    tokens = str(source_dsf or "").split()
    if not tokens:
        return True
    packs = [token[5:] for token in tokens if token.startswith("pack=")]
    if not packs:
        return False
    # DEFAULT is allowed only as a legacy fallback after an explicit
    # Ortho4XP preference.  A DEFAULT-only source is intentionally rejected.
    return any("Ortho4XP" in value for value in packs) and all(
        "Ortho4XP" in value or value == "DEFAULT" for value in packs
    )


def _strict_read(path: str, tile: str) -> MuxpFile:
    update, error = readMuxpFile(path, LOG_NAME)
    if update is None:
        raise MuxpApplyError("could not read MUXP file {}: {}".format(path, error))
    try:
        version = _version(update.get("muxp_version"))
    except MuxpApplyError:
        raise
    if version > _version(MUXP_FILE_DEFS.SUPPORTED_MUXP_FILE_VERSION):
        raise MuxpApplyError(
            "MUXP file version {} is newer than supported {}".format(
                version, MUXP_FILE_DEFS.SUPPORTED_MUXP_FILE_VERSION
            )
        )
    if _canonical_tile(update.get("tile", "")) != _canonical_tile(tile):
        raise MuxpApplyError("MUXP tile does not match requested tile: {}".format(path))
    if not str(update.get("id", "")).strip():
        raise MuxpApplyError("MUXP id is empty: {}".format(path))
    if len(str(update.get("area", "")).split()) != 4:
        raise MuxpApplyError("MUXP area must contain four values: {}".format(path))
    code, message = validate_muxp(update, LOG_NAME)
    if code != 0:
        raise MuxpApplyError("strict MUXP validation failed for {}: {}".format(path, message))
    if not _source_is_ortho4xp(update.get("source_dsf", "")):
        raise MuxpApplyError(
            "MUXP source_dsf is not compatible with Ortho4XP: {}".format(
                update.get("source_dsf", "")
            )
        )
    commands = update.get("commands", [])
    for command in commands:
        name = command.get("command")
        if name not in MUXP_FILE_DEFS.MUXP_COMMANDS:
            raise MuxpApplyError("unsupported MUXP command '{}'".format(name))
    return MuxpFile(
        path=os.path.abspath(path),
        relative_path="",
        sha256=_sha256(path),
        update=update,
    )


def discover_muxp_files(project_root: str, folder: str, tile: str) -> list[MuxpFile]:
    """Find and strictly validate all matching MUXP files."""
    root = Path(folder)
    if not root.is_absolute():
        root = Path(project_root) / root
    root = root.resolve()
    if not root.is_dir():
        return []
    candidates = sorted(
        (path for path in root.rglob("*.muxp") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    parsed = []
    for path in candidates:
        update, error = readMuxpFile(str(path), LOG_NAME)
        if update is None:
            raise MuxpApplyError("could not read MUXP file {}: {}".format(path, error))
        if "tile" not in update:
            continue
        if _canonical_tile(update["tile"]) != _canonical_tile(tile):
            continue
        entry = _strict_read(str(path), tile)
        parsed.append(
            MuxpFile(
                path=entry.path,
                relative_path=path.relative_to(root).as_posix(),
                sha256=entry.sha256,
                update=entry.update,
            )
        )

    by_id = {}
    for entry in parsed:
        update_id = entry.update["id"]
        by_id.setdefault(update_id, []).append(entry)
    selected = []
    for update_id, entries in by_id.items():
        latest_version = max(_version(entry.update["version"]) for entry in entries)
        latest = [entry for entry in entries if _version(entry.update["version"]) == latest_version]
        if len({entry.sha256 for entry in latest}) > 1:
            raise MuxpApplyError(
                "same MUXP id/version has different contents: {} {}".format(
                    update_id, latest_version
                )
            )
        selected.append(sorted(latest, key=lambda entry: entry.relative_path)[0])
    return sorted(selected, key=lambda entry: entry.relative_path.casefold())


def _existing_updates(properties: dict) -> dict[str, tuple[Decimal, str]]:
    result = {}
    for key, value in properties.items():
        if not key.startswith("muxp/update/"):
            continue
        try:
            update_id, version, area = value.split("/", 2)
            result[update_id] = (_version(version), area)
        except (ValueError, TypeError):
            raise MuxpApplyError("invalid existing MUXP property {}={}".format(key, value))
    return result


def _add_update_property(properties: dict, update: dict):
    numbers = []
    for key in properties:
        if key.startswith("muxp/update/"):
            try:
                numbers.append(int(key.rsplit("/", 1)[1]))
            except ValueError:
                raise MuxpApplyError("invalid MUXP update property key {}".format(key))
    next_number = max(numbers, default=0) + 1
    area = " ".join(str(value) for value in update["area"])
    properties["muxp/update/{}".format(next_number)] = "{}/{}/{}".format(
        update["id"], update["version"], area
    )


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.sys.platform == "darwin":
        value /= 1024.0 * 1024.0
    else:
        value /= 1024.0
    return round(value, 3)


def apply_muxp_to_dsf(
    dsf_path: str,
    project_root: str,
    muxp_folder: str,
    tile: str,
    allow_side_effects: bool = False,
    cancel_checker: Callable[[], bool] | None = None,
    xplane_root: str | None = None,
) -> MuxpApplyResult:
    """Apply matching MUXP files to an already staged Ortho4XP DSF."""
    started = time.perf_counter()
    matching = discover_muxp_files(project_root, muxp_folder, tile)
    base_manifest = {
        "schema_version": 1,
        "tile": _canonical_tile(tile),
        "enabled": True,
        "muxp_folder": str(muxp_folder),
        "files": [],
        "result": "no_match" if not matching else "pending",
        "source_dsf": "validated against generated Ortho4XP DSF",
        "command_timings": [],
        "side_effects_allowed": bool(allow_side_effects),
        "read_ms": 0.0,
        "process_ms": 0.0,
        "write_ms": 0.0,
        "elapsed_ms": 0.0,
        "peak_rss_mb": _peak_rss_mb(),
    }
    if not matching:
        if os.path.isfile(dsf_path):
            digest = _sha256(dsf_path)
            base_manifest["input_dsf_sha256"] = digest
            base_manifest["output_dsf_sha256"] = digest
        base_manifest["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return MuxpApplyResult(base_manifest, False)

    command_names = {
        command.get("command")
        for entry in matching
        for command in entry.update.get("commands", [])
    }
    if command_names & SIDE_EFFECT_COMMANDS and not allow_side_effects:
        raise MuxpApplyError(
            "MUXP contains side-effect commands {}; explicit permission is required".format(
                sorted(command_names & SIDE_EFFECT_COMMANDS)
            )
        )

    log_name = LOG_NAME
    _legacy.LogName = log_name
    _legacy.log = log
    dsf = _legacy.XPLNEDSF(log_name, None)
    read_started = time.perf_counter()
    input_dsf_sha256 = _sha256(dsf_path)
    if dsf.read(dsf_path) != 0:
        raise MuxpApplyError("could not read staged DSF {}".format(dsf_path))
    read_ms = (time.perf_counter() - read_started) * 1000.0
    existing = _existing_updates(dsf.Properties)
    applied = []
    skipped = []
    all_timings = []
    for entry in matching:
        update = entry.update
        update_id = update["id"]
        version = _version(update["version"])
        previous = existing.get(update_id)
        if previous is not None and previous[0] == version:
            skipped.append(entry)
            continue
        if previous is not None and previous[0] > version:
            skipped.append(entry)
            continue
        if previous is not None and previous[0] < version:
            raise MuxpApplyError(
                "staged DSF contains older MUXP update {}; rebuild from base before applying {}".format(
                    update_id, update["version"]
                )
            )
        queue = _TimingQueue(cancel_checker)
        runner = _HeadlessMuxp(
            dsf,
            entry.path,
            xplane_root or project_root,
            allow_side_effects,
            cancel_checker,
        )
        runner.msg_queue = queue
        process_started = time.perf_counter()
        error = runner.processMuxp(entry.path, update)
        queue.finish()
        if error:
            raise MuxpApplyError(
                "MUXP command processing failed for {} with code {}".format(
                    entry.relative_path, error
                )
            )
        all_timings.extend(queue.timings)
        if "muxp/HashDSFbaseFile" not in dsf.Properties:
            dsf.Properties["muxp/HashDSFbaseFile"] = str(dsf.FileHash)
        _add_update_property(dsf.Properties, update)
        applied.append(entry)
        existing[update_id] = (version, " ".join(str(value) for value in update["area"]))
        base_manifest["files"].append(
            {
                "path": entry.relative_path,
                "sha256": entry.sha256,
                "id": update_id,
                "version": str(update["version"]),
                "area": list(update["area"]),
                "process_ms": round((time.perf_counter() - process_started) * 1000.0, 3),
                "status": "applied",
            }
        )
    for entry in skipped:
        base_manifest["files"].append(
            {
                "path": entry.relative_path,
                "sha256": entry.sha256,
                "id": entry.update["id"],
                "version": str(entry.update["version"]),
                "area": list(entry.update["area"]),
                "process_ms": 0.0,
                "status": "skipped_idempotent",
            }
        )

    base_manifest["read_ms"] = round(read_ms, 3)
    base_manifest["process_ms"] = round((time.perf_counter() - started) * 1000.0 - read_ms, 3)
    base_manifest["command_timings"] = all_timings
    if applied:
        write_started = time.perf_counter()
        if dsf.write(dsf_path) != 0:
            raise MuxpApplyError("could not write staged DSF {}".format(dsf_path))
        base_manifest["write_ms"] = round((time.perf_counter() - write_started) * 1000.0, 3)
        base_manifest["result"] = "applied"
    else:
        base_manifest["result"] = "skipped_idempotent"
    base_manifest["input_dsf_sha256"] = input_dsf_sha256
    base_manifest["output_dsf_sha256"] = _sha256(dsf_path)
    base_manifest["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    base_manifest["peak_rss_mb"] = _peak_rss_mb()
    return MuxpApplyResult(base_manifest, bool(applied))


def write_manifest(tile_build_dir: str, manifest: dict) -> str:
    """Atomically publish the tile-local MUXP manifest."""
    destination = os.path.join(tile_build_dir, "Ortho4XP_muxp.json")
    temporary = destination + ".tmp"
    os.makedirs(tile_build_dir, exist_ok=True)
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination
