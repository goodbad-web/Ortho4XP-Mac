import os
import sys
import subprocess
import time
import shutil
import json
import queue
import threading
import tempfile
import re
import traceback
import multiprocessing
import uuid
from contextlib import contextmanager, nullcontext
from itertools import count
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_OSM_Utils as OSM
import O4_Imagery_Utils as IMG
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_DSF_Utils as DSF
import O4_Overlay_Utils as OVL
import O4_DSF_Budget as DSF_BUDGET
import O4_Performance_Utils as PERF
import O4_Shared_Memory as SHMEM
from O4_Parallel_Utils import (
    parallel_launch,
    parallel_join,
    multiprocessing_pool,
    ReusableMultiprocessingPool,
)
from O4_Tile_Scheduler import ConversionResult, ConversionTask, TileConversionScheduler
from O4_ASHelper_Server import ASHelperJSONLServer
from PIL import Image

max_convert_slots = 8
max_download_slots = 8
skip_downloads = False
skip_converts = False
enable_streaming_conversion = False
conversion_queue_size = 0
gpu_batch_size = 0
gpu_dds_workers = 8
gpu_batch_wait_ms = 50
enable_shared_memory_handoff = False
shared_memory_budget_gb = 0
enable_parallel_overlay = False
max_parallel_tiles = 1


_BUILD_TRANSACTION_MARKER = ".Ortho4XP_build_recovery.json"
_BUILD_TRANSACTION_JOURNAL = _BUILD_TRANSACTION_MARKER + ".journal"
_LEGACY_MASK_PATTERN = re.compile(r"^-?\d+_-?\d+\.png$")
_DISTANCE_MASK_PATTERN = re.compile(r"^-?\d+_-?\d+_dist\.png$")
_MASK_TEXTURE_PATTERN = re.compile(r"^-?\d+_-?\d+_.+_ZL\d+\.png$")
_DDS_TEXTURE_PATTERN = re.compile(r"^-?\d+_-?\d+_.+\.dds$")
_AUTO_REDUCE_SETTING_NAMES = (
    "max_levelled_segs",
    "water_simplification",
    "cover_zl",
    "curvature_tol",
    "limit_tris",
)
_ASHELPER_CAPABILITY_CACHE = {}

# TensorOps keeps several FP16 activation/output buffers resident until the
# command buffer completes.  The estimate below intentionally models the
# fixed FP8SR graph rather than relying on the source JPEG size.  It is used
# only for scheduling; ASHelper remains the authority for validating the
# actual model pack and tile geometry.
TENSOROPS_MEMORY_FRACTION = 0.70
TENSOROPS_MAX_WORKERS = 4
TENSOROPS_FIXED_RESERVE_BYTES = 256 * 1024 * 1024
TENSOROPS_ESTIMATE_SAFETY_FACTOR = 1.35
TENSOROPS_MAX_TILE_DIMENSION = 2050


def _tensorops_estimated_working_set_bytes(spec):
    """Estimate one TensorOps child peak for a direct-DDS item.

    The current FP8SR contract has K values 32, 320, 320 and three 32-channel
    layer outputs.  A 2048px tile therefore dominates the estimate even when
    the source JPEG is small.  The safety factor covers Metal allocator and
    readback overhead observed in the real helper process.
    """

    try:
        width, height = (int(value) for value in spec.get("input_size", (0, 0)))
    except (AttributeError, TypeError, ValueError):
        width, height = 0, 0
    if width <= 0 or height <= 0:
        return TENSOROPS_FIXED_RESERVE_BYTES

    tile_width = min(width, TENSOROPS_MAX_TILE_DIMENSION)
    tile_height = min(height, TENSOROPS_MAX_TILE_DIMENSION)
    tile_pixels = tile_width * tile_height
    # FP16 activations: (32 + 320 + 320) values per pixel.  The three layer
    # outputs add 3 * 32 values per pixel.  Input RGBA and the final RGBA
    # output are retained for the duration of a tiled request.  Include the
    # temporary compressed DDS payload as well; it is written before the
    # parent atomically publishes the final path.
    graph_bytes = tile_pixels * 2 * (32 + 320 + 320 + (3 * 32))
    full_frame_bytes = width * height * (4 + 16)
    dds_bytes = _tensorops_estimated_dds_bytes(spec)
    estimated = (
        graph_bytes + full_frame_bytes + dds_bytes
    ) * TENSOROPS_ESTIMATE_SAFETY_FACTOR
    return int(estimated + TENSOROPS_FIXED_RESERVE_BYTES)


def _tensorops_estimated_dds_bytes(spec):
    """Estimate one temporary DDS payload including its complete mip chain."""

    try:
        width, height = (int(value) for value in spec.get("input_size", (0, 0)))
        output_width = width * 2
        output_height = height * 2
        target_format = str(spec.get("target_format", "BC3")).upper()
    except (AttributeError, TypeError, ValueError):
        return 0
    if output_width <= 0 or output_height <= 0:
        return 0
    block_bytes = 8 if target_format == "BC1" else 16
    payload_bytes = 0
    level_width = output_width
    level_height = output_height
    while True:
        payload_bytes += (
            max(1, (level_width + 3) // 4)
            * max(1, (level_height + 3) // 4)
            * block_bytes
        )
        if level_width == 1 and level_height == 1:
            break
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)
    # BC7 uses the optional DX10 extension header. ASHelper currently accepts
    # BC1/BC3, but keeping the estimate format-aware makes the planner safe
    # for a future direct-DDS converter as well.
    header_bytes = 128 + (20 if target_format == "BC7" else 0)
    return header_bytes + payload_bytes


def _run_tensorops_child(command):
    """Run one TensorOps child and return its output plus child RSS."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output = process.stdout.read() if process.stdout is not None else ""
        if hasattr(os, "wait4"):
            while True:
                try:
                    _, wait_status, usage = os.wait4(process.pid, 0)
                    break
                except InterruptedError:
                    continue
            if hasattr(os, "waitstatus_to_exitcode"):
                returncode = os.waitstatus_to_exitcode(wait_status)
            elif os.WIFEXITED(wait_status):
                returncode = os.WEXITSTATUS(wait_status)
            elif os.WIFSIGNALED(wait_status):
                returncode = -os.WTERMSIG(wait_status)
            else:
                returncode = 1
            process.returncode = returncode
            rss = float(usage.ru_maxrss)
            if sys.platform == "darwin":
                rss /= 1024.0 * 1024.0
            else:
                rss /= 1024.0
            rss_scope = "child_wait4"
        else:
            returncode = process.wait()
            rss = 0.0
            rss_scope = "unavailable"
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return (
        subprocess.CompletedProcess(command, returncode, output),
        {"peak_rss_mb": rss, "rss_scope": rss_scope},
    )


def _tensorops_memory_plan(specs, max_workers=None, chunk_size=8):
    """Return a conservative worker plan for TensorOps direct-DDS work."""

    chunk_size = max(1, int(chunk_size))
    chunk_count = (len(specs) + chunk_size - 1) // chunk_size if specs else 0
    if not specs:
        return {
            "can_run": False,
            "workers": 0,
            "chunk_count": 0,
            "memory_budget_mb": 0,
            "parent_rss_mb": 0,
            "estimated_worker_mb": 0,
            "estimated_dds_mb": 0,
            "estimated_total_mb": 0,
            "reason": "empty",
        }

    requested = max_convert_slots if max_workers is None else max_workers
    try:
        requested = max(1, int(requested))
    except (TypeError, ValueError):
        requested = 1
    requested = min(TENSOROPS_MAX_WORKERS, requested, chunk_count)
    def input_max_dimension(spec):
        try:
            return max(int(value) for value in spec.get("input_size", (0, 0)))
        except (AttributeError, TypeError, ValueError):
            return 0

    large_input = any(input_max_dimension(spec) >= 2048 for spec in specs)
    if large_input:
        # A 2048px input already occupies the large tiled working set.  Keep
        # it in one child even on high-memory hosts; MetalFX remains the
        # explicit retry path when the serialized TensorOps child fails.
        requested = min(requested, 1)
    estimated_worker_bytes = max(
        _tensorops_estimated_working_set_bytes(spec) for spec in specs
    )
    estimated_dds_bytes = max(
        _tensorops_estimated_dds_bytes(spec) for spec in specs
    )
    physical_bytes = int(PERF.physical_memory_bytes() or 0)
    parent_rss_bytes = int(PERF.peak_rss_bytes() or 0)
    if physical_bytes <= 0:
        # A host without a readable physical-memory value cannot be safely
        # auto-sized.  Keep the historical single-child behavior instead of
        # guessing a larger parallelism.
        workers = 1
        reason = "physical_memory_unavailable"
        budget_bytes = 0
    else:
        budget_bytes = int(physical_bytes * TENSOROPS_MEMORY_FRACTION)
        available_bytes = budget_bytes - parent_rss_bytes - TENSOROPS_FIXED_RESERVE_BYTES
        workers = min(
            requested,
            max(0, available_bytes // max(1, estimated_worker_bytes)),
        )
        if workers <= 0:
            reason = "memory_budget_exceeded"
        elif large_input:
            reason = "large_input_serialized"
        else:
            reason = "ok"

    return {
        "can_run": workers > 0,
        "workers": int(workers),
        "chunk_count": chunk_count,
        "memory_budget_mb": int(budget_bytes / (1024 * 1024)),
        "parent_rss_mb": int(parent_rss_bytes / (1024 * 1024)),
        "estimated_worker_mb": int(estimated_worker_bytes / (1024 * 1024)),
        "estimated_dds_mb": int(estimated_dds_bytes / (1024 * 1024)),
        "estimated_total_mb": int(
            (
                parent_rss_bytes
                + TENSOROPS_FIXED_RESERVE_BYTES
                + estimated_worker_bytes * workers
            )
            / (1024 * 1024)
        ),
        "reason": reason,
    }


def _is_generated_dds_name(name):
    """Recognize DDS names produced from an orthogrid texture tile."""
    for suffix in (".gpu.tmp.dds", ".tmp.dds"):
        if name.endswith(suffix):
            name = name[: -len(suffix)] + ".dds"
            break
    return bool(_DDS_TEXTURE_PATTERN.match(name))


def _is_generated_mask_name(name):
    """Recognize legacy, distance, and per-texture mask output names."""
    return bool(
        _LEGACY_MASK_PATTERN.match(name)
        or _DISTANCE_MASK_PATTERN.match(name)
        or _MASK_TEXTURE_PATTERN.match(name)
    )


def _is_generated_terrain_name(name):
    """Recognize terrain files derived from generated orthogrid textures."""
    if not name.endswith(".ter"):
        return False
    stem = name[:-4]
    for suffix in ("_water_overlay", "_sea_overlay", "_water", "_sea", "_overlay", ""):
        if suffix and not stem.endswith(suffix):
            continue
        texture_stem = stem[: -len(suffix)] if suffix else stem
        if _is_generated_dds_name(texture_stem + ".dds"):
            return True
    return False


class _BuildTransaction:
    """Keep a tile build recoverable while publishing one coherent result.

    Staging is created next to each canonical output volume.  In particular,
    masks and overlays must not be staged below the build directory: doing so
    makes an otherwise atomic ``os.replace`` fail with ``EXDEV`` on common
    installations.  DDS files are still hardlinked/copied using the existing
    policy, so large tile payloads are not duplicated unnecessarily.
    """

    def __init__(self, tile, preserve_inputs=False, include_overlay=False):
        self.build_dir = os.path.abspath(tile.build_dir)
        self.mask_dir = os.path.abspath(FNAMES.mask_dir(tile.lat, tile.lon))
        self.grouped = bool(getattr(tile, "grouped", False))
        self.preserve_inputs = bool(preserve_inputs)
        self.include_overlay = bool(include_overlay)
        self.parent_dir = os.path.dirname(self.build_dir) or os.curdir
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.parent_dir, exist_ok=True)
        self.root = tempfile.mkdtemp(
            prefix=".o4xp-build-transaction-", dir=self.parent_dir
        )
        self.roots = {"tile": self.root, "shared": self.root}
        mask_parent = os.path.dirname(self.mask_dir) or os.curdir
        os.makedirs(mask_parent, exist_ok=True)
        self.mask_root = tempfile.mkdtemp(
            prefix=".o4xp-mask-transaction-", dir=mask_parent
        )
        self.roots["mask"] = self.mask_root
        if self.include_overlay:
            overlay_parent = os.path.abspath(FNAMES.Overlay_dir)
            os.makedirs(overlay_parent, exist_ok=True)
            self.overlay_root = tempfile.mkdtemp(
                prefix=".o4xp-overlay-transaction-", dir=overlay_parent
            )
            self.roots["overlay"] = self.overlay_root
        else:
            self.overlay_root = None
        self.overlay_active = False
        self.marker_path = os.path.join(
            self.build_dir, _BUILD_TRANSACTION_MARKER
        )
        self.journal_path = os.path.join(
            self.build_dir, _BUILD_TRANSACTION_JOURNAL
        )
        self.tile_lat = int(tile.lat)
        self.tile_lon = int(tile.lon)
        self.config_paths = {
            "config": os.path.join(
                self.build_dir,
                "Ortho4XP_{}.cfg".format(
                    FNAMES.short_latlon(self.tile_lat, self.tile_lon)
                ),
            ),
            "backup": os.path.join(
                self.build_dir,
                "Ortho4XP_{}.cfg.bak".format(
                    FNAMES.short_latlon(self.tile_lat, self.tile_lon)
                ),
            ),
        }
        self.initial_config = {}
        try:
            self._snapshot_initial_config()
            if self.preserve_inputs:
                # A standalone Step 3 still needs the existing mesh, masks,
                # and reusable DDS files visible at their canonical paths.
                self._write_marker("snapshotting", None, None)
                self._snapshot_initial_outputs()
                self._write_marker("active", None, None)
            else:
                self._write_marker("snapshotting", None, None)
                # Keep the pre-build snapshot immutable for the whole
                # transaction.  Moving it out of the canonical tree made a
                # later rollback depend on the exact order of intermediate
                # restores and could consume the only copy of the old tile.
                self._snapshot_initial_outputs()
                self._write_marker("clearing", None, None)
                self._remove_current_outputs()
                self._write_marker("active", None, None)
        except Exception:
            # Roll back *all* staging allocations, including the mask/overlay
            # volumes.  The old implementation only cleaned preserve_inputs
            # failures and could leave a marker that hid an EXDEV failure.
            recovery_failed = False
            if not self.preserve_inputs:
                try:
                    self.restore_snapshot_files("initial")
                except Exception:
                    # The original exception remains the useful diagnostic;
                    # recovery on the next launch can use any surviving marker.
                    recovery_failed = True
            if not recovery_failed:
                for root in set(self.roots.values()):
                    shutil.rmtree(root, ignore_errors=True)
                for path in (
                    self.marker_path,
                    self.marker_path + ".tmp",
                    self.journal_path,
                ):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            raise

    def _snapshot_initial_config(self):
        config_root = os.path.join(self.root, "initial", "config")
        for name, source_path in self.config_paths.items():
            present = os.path.isfile(source_path)
            self.initial_config[name] = present
            if present:
                os.makedirs(config_root, exist_ok=True)
                shutil.copy2(
                    source_path,
                    os.path.join(config_root, os.path.basename(source_path)),
                )

    def _staging_root(self, kind):
        return self.roots["mask"] if kind == "mask" else self.roots[kind]

    def _snapshot_kind_root(self, snapshot_name, kind):
        return os.path.join(self._staging_root(kind), snapshot_name, kind)

    def _snapshot_initial_outputs(self):
        previous_overlay_active = self.overlay_active
        self.overlay_active = self.include_overlay
        try:
            self._link_current_to("initial")
        finally:
            self.overlay_active = previous_overlay_active

    def _write_marker(
        self,
        state,
        best_snapshot,
        best_settings=None,
        target_snapshot=None,
    ):
        marker = {
            "version": 2,
            "state": state,
            # Keep the v1 field for older recovery tooling and tests.
            "transaction_root": self.root,
            "transaction_roots": dict(self.roots),
            "build_dir": self.build_dir,
            "mask_dir": self.mask_dir,
            "overlay_dir": os.path.abspath(FNAMES.Overlay_dir),
            "overlay_path": self.overlay_path if self.include_overlay else None,
            "include_overlay": self.include_overlay,
            "initial_config": dict(self.initial_config),
            "grouped": self.grouped,
            "preserve_inputs": self.preserve_inputs,
            "lat": self.tile_lat,
            "lon": self.tile_lon,
            "best_snapshot": best_snapshot,
            "best_settings": best_settings,
            "best_config": getattr(self, "best_config", None),
            "target_snapshot": target_snapshot,
        }
        marker_tmp = self.marker_path + ".tmp"
        with open(marker_tmp, "w", encoding="utf-8") as stream:
            json.dump(marker, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(marker_tmp, self.marker_path)
        try:
            with open(self.journal_path, "a", encoding="utf-8") as stream:
                json.dump(marker, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            # The manifest is authoritative.  A journal write can fail on a
            # read-only sidecar filesystem without making a successfully
            # published tile unusable.
            pass

    @property
    def overlay_path(self):
        return os.path.join(
            os.path.abspath(FNAMES.Overlay_dir),
            "Earth nav data",
            FNAMES.round_latlon(self.tile_lat, self.tile_lon),
            FNAMES.short_latlon(self.tile_lat, self.tile_lon) + ".dsf",
        )

    def overlay_candidate_path(self):
        if not self.include_overlay or self.overlay_root is None:
            raise RuntimeError("overlay staging is not enabled")
        path = os.path.join(
            self.overlay_root,
            "overlay-candidate",
            os.path.basename(self.overlay_path),
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def activate_overlay(self, candidate_path):
        if not os.path.isfile(candidate_path):
            return False
        os.makedirs(os.path.dirname(self.overlay_path), exist_ok=True)
        os.replace(candidate_path, self.overlay_path)
        self.overlay_active = True
        return True

    def set_best_snapshot(self, snapshot_name, settings=None, config=None):
        self.best_config = config
        self._write_marker(
            "active",
            snapshot_name,
            dict(settings) if settings is not None else None,
        )

    def _tile_output_paths(self):
        paths = []
        if not os.path.isdir(self.build_dir):
            return paths

        data_prefix = "Data" + FNAMES.short_latlon(
            self.tile_lat, self.tile_lon
        )
        for name in os.listdir(self.build_dir):
            path = os.path.join(self.build_dir, name)
            if os.path.isfile(path) and name.startswith(data_prefix):
                paths.append(("tile", path))

        dsf_base = os.path.join(
            self.build_dir,
            "Earth nav data",
            FNAMES.long_latlon(self.tile_lat, self.tile_lon) + ".dsf",
        )
        for path in (dsf_base, dsf_base + ".bak", dsf_base + ".tmp"):
            if os.path.isfile(path):
                paths.append(("tile", path))

        terrain_dir = os.path.join(self.build_dir, "terrain")
        if os.path.isdir(terrain_dir):
            for dir_path, _, names in os.walk(terrain_dir):
                for name in names:
                    if _is_generated_terrain_name(name):
                        path = os.path.join(dir_path, name)
                        if os.path.isfile(path):
                            paths.append(("shared" if self.grouped else "tile", path))

        textures_dir = os.path.join(self.build_dir, "textures")
        if os.path.isdir(textures_dir):
            for name in os.listdir(textures_dir):
                path = os.path.join(textures_dir, name)
                if not os.path.isfile(path):
                    continue
                is_texture = _is_generated_dds_name(name)
                is_mask = _is_generated_mask_name(name)
                is_water_transition = name == "water_transition.png"
                if is_texture or is_mask or is_water_transition:
                    paths.append(("shared" if self.grouped else "tile", path))
        return paths

    def _mask_output_paths(self):
        paths = []
        if not os.path.isdir(self.mask_dir):
            return paths
        for dir_path, _, names in os.walk(self.mask_dir):
            for name in names:
                if not _is_generated_mask_name(name):
                    continue
                path = os.path.join(dir_path, name)
                if os.path.isfile(path):
                    paths.append(("mask", path))
        return paths

    def _current_output_paths(self):
        paths = self._tile_output_paths() + self._mask_output_paths()
        if (
            self.include_overlay
            and self.overlay_active
            and os.path.isfile(self.overlay_path)
        ):
            paths.append(("overlay", self.overlay_path))
        return paths

    def _link_current_to(self, snapshot_name):
        """Snapshot current files while leaving standalone inputs visible."""
        for kind, source_path in self._current_output_paths():
            source_root = {
                "mask": self.mask_dir,
                "overlay": os.path.dirname(self.overlay_path),
            }.get(kind, self.build_dir)
            relative_path = os.path.relpath(source_path, source_root)
            destination_path = os.path.join(
                self._snapshot_kind_root(snapshot_name, kind), relative_path
            )
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            if source_path.lower().endswith(".dds"):
                os.link(source_path, destination_path)
            else:
                shutil.copy2(source_path, destination_path)

    def _move_current_to(self, snapshot_name):
        for kind, source_path in self._current_output_paths():
            source_root = {
                "mask": self.mask_dir,
                "overlay": os.path.dirname(self.overlay_path),
            }.get(kind, self.build_dir)
            relative_path = os.path.relpath(source_path, source_root)
            destination_path = os.path.join(
                self._snapshot_kind_root(snapshot_name, kind), relative_path
            )
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            os.replace(source_path, destination_path)

    def _remove_current_outputs(self):
        for _, source_path in self._current_output_paths():
            try:
                os.remove(source_path)
            except OSError:
                pass

    def capture_candidate(self, attempt):
        snapshot_name = "candidate-{}".format(attempt)
        self._move_current_to(snapshot_name)
        return snapshot_name

    def clone_snapshot(self, source_snapshot, target_snapshot):
        """Clone a candidate without consuming it."""
        found = False
        for kind in ("tile", "shared", "mask", "overlay"):
            source_kind_root = self._snapshot_kind_root(source_snapshot, kind)
            if not os.path.isdir(source_kind_root):
                continue
            found = True
            target_kind_root = self._snapshot_kind_root(target_snapshot, kind)
            for dir_path, _, names in os.walk(source_kind_root):
                for name in names:
                    source_path = os.path.join(dir_path, name)
                    relative_path = os.path.relpath(source_path, source_kind_root)
                    destination_path = os.path.join(target_kind_root, relative_path)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    if source_path.lower().endswith(".dds"):
                        os.link(source_path, destination_path)
                    else:
                        shutil.copy2(source_path, destination_path)
        if not found:
            raise FileNotFoundError(source_snapshot)
        return target_snapshot

    def discard_current(self, label):
        self._move_current_to("discarded-{}".format(label))

    def prepare_attempt(self):
        """Restore shared grouped assets before the next clean attempt."""
        if not self.grouped:
            return
        source_root = self._snapshot_kind_root("initial", "shared")
        if not os.path.isdir(source_root):
            return
        for dir_path, _, names in os.walk(source_root):
            for name in names:
                source_path = os.path.join(dir_path, name)
                relative_path = os.path.relpath(source_path, source_root)
                destination_path = os.path.join(self.build_dir, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                os.replace(source_path, destination_path)

    def restore_snapshot(self, snapshot_name):
        self.discard_current("before-restore")
        self.restore_snapshot_files(snapshot_name)
        if snapshot_name == "initial":
            self.restore_initial_config()

    def restore_initial_config(self):
        config_root = os.path.join(self.root, "initial", "config")
        for name, destination_path in self.config_paths.items():
            if self.initial_config.get(name, False):
                source_path = os.path.join(
                    config_root, os.path.basename(destination_path)
                )
                if os.path.isfile(source_path):
                    self._publish_snapshot_file(source_path, destination_path)
            else:
                try:
                    os.remove(destination_path)
                except OSError:
                    pass

    @staticmethod
    def _publish_snapshot_file(source_path, destination_path):
        """Publish a snapshot file without consuming the initial snapshot."""
        destination_dir = os.path.dirname(destination_path) or os.curdir
        os.makedirs(destination_dir, exist_ok=True)
        temporary_path = os.path.join(
            destination_dir,
            ".o4xp-restore-{}-{}".format(
                os.getpid(), uuid.uuid4().hex
            ),
        )
        try:
            if source_path.lower().endswith(".dds"):
                try:
                    os.link(source_path, temporary_path)
                except OSError:
                    shutil.copy2(source_path, temporary_path)
            else:
                shutil.copy2(source_path, temporary_path)
            os.replace(temporary_path, destination_path)
        finally:
            try:
                os.remove(temporary_path)
            except OSError:
                pass

    def restore_snapshot_files(self, snapshot_name):
        """Move the remaining files of a snapshot into canonical paths."""
        for kind, destination_root in (
            ("tile", self.build_dir),
            ("shared", self.build_dir),
            ("mask", self.mask_dir),
            ("overlay", os.path.dirname(self.overlay_path)),
        ):
            if kind == "overlay" and not self.include_overlay:
                continue
            source_root = self._snapshot_kind_root(snapshot_name, kind)
            if not os.path.isdir(source_root):
                continue
            for dir_path, _, names in os.walk(source_root):
                for name in names:
                    source_path = os.path.join(dir_path, name)
                    relative_path = os.path.relpath(source_path, source_root)
                    destination_path = os.path.join(destination_root, relative_path)
                    if snapshot_name == "initial":
                        self._publish_snapshot_file(source_path, destination_path)
                    else:
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        os.replace(source_path, destination_path)

    def snapshot_has_files(self, snapshot_name):
        for kind in ("tile", "shared", "mask", "overlay"):
            if kind == "overlay" and not self.include_overlay:
                continue
            snapshot_root = self._snapshot_kind_root(snapshot_name, kind)
            for _, _, names in os.walk(snapshot_root):
                if names:
                    return True
        return False

    def export_snapshot(self, snapshot_name, status):
        """Copy a candidate to persistent, non-published degraded staging."""
        if not self.snapshot_has_files(snapshot_name):
            raise FileNotFoundError(snapshot_name)
        staging_root = tempfile.mkdtemp(
            prefix=".o4xp-degraded-{}-".format(
                FNAMES.short_latlon(self.tile_lat, self.tile_lon)
            ),
            dir=self.parent_dir,
        )
        try:
            for kind in ("tile", "shared", "mask"):
                source_kind_root = self._snapshot_kind_root(snapshot_name, kind)
                if not os.path.isdir(source_kind_root):
                    continue
                destination_kind_root = os.path.join(staging_root, kind)
                for dir_path, _, names in os.walk(source_kind_root):
                    for name in names:
                        source_path = os.path.join(dir_path, name)
                        relative_path = os.path.relpath(source_path, source_kind_root)
                        destination_path = os.path.join(
                            destination_kind_root, relative_path
                        )
                        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                        if source_path.lower().endswith(".dds"):
                            try:
                                os.link(source_path, destination_path)
                            except OSError:
                                shutil.copy2(source_path, destination_path)
                        else:
                            shutil.copy2(source_path, destination_path)
            status_path = os.path.join(staging_root, "status.json")
            with open(status_path, "w", encoding="utf-8") as stream:
                json.dump(status, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            return staging_root
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

    def cleanup(self):
        for root in set(self.roots.values()):
            if os.path.isdir(root):
                shutil.rmtree(root)
        for path in (
            self.marker_path,
            self.marker_path + ".tmp",
            getattr(self, "journal_path", self.marker_path + ".journal"),
        ):
            try:
                os.remove(path)
            except OSError:
                pass


def _transaction_marker_path(tile):
    return os.path.join(
        os.path.abspath(tile.build_dir), _BUILD_TRANSACTION_MARKER
    )


def _transaction_journal_path(tile):
    return os.path.join(
        os.path.abspath(tile.build_dir), _BUILD_TRANSACTION_JOURNAL
    )


def _recover_build_transaction(tile):
    """Restore a left-over full-pipeline transaction after a hard stop."""
    marker_path = _transaction_marker_path(tile)
    journal_path = _transaction_journal_path(tile)
    if not os.path.isfile(marker_path):
        if not os.path.isfile(journal_path):
            return True
        marker = None
        try:
            with open(journal_path, "r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        candidate = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(candidate, dict):
                        marker = candidate
        except OSError:
            return True
        if marker is None:
            try:
                os.remove(journal_path)
            except OSError:
                pass
            return True
    else:
        marker = None
    try:
        if marker is None:
            with open(marker_path, "r", encoding="utf-8") as stream:
                marker = json.load(stream)
        version = int(marker.get("version", 1))
        root = os.path.abspath(marker["transaction_root"])
        build_dir = os.path.abspath(tile.build_dir)
        mask_dir = os.path.abspath(FNAMES.mask_dir(tile.lat, tile.lon))
        parent_dir = os.path.dirname(build_dir) or os.curdir
        if (
            os.path.dirname(root) != os.path.abspath(parent_dir)
            or not os.path.basename(root).startswith(".o4xp-build-transaction-")
            or marker.get("build_dir") != build_dir
            or marker.get("mask_dir") != mask_dir
            or int(marker.get("lat")) != int(tile.lat)
            or int(marker.get("lon")) != int(tile.lon)
        ):
            raise ValueError("recovery marker does not match the current tile")
        transaction = _BuildTransaction.__new__(_BuildTransaction)
        transaction.build_dir = build_dir
        transaction.mask_dir = mask_dir
        transaction.grouped = bool(marker.get("grouped", False))
        transaction.preserve_inputs = bool(marker.get("preserve_inputs", False))
        transaction.parent_dir = os.path.abspath(parent_dir)
        transaction.root = root
        transaction.roots = {
            "tile": root,
            "shared": root,
            "mask": root,
        }
        transaction.mask_root = root
        transaction.overlay_root = None
        transaction.include_overlay = False
        if version >= 2:
            transaction.include_overlay = bool(marker.get("include_overlay", False))
            raw_roots = marker.get("transaction_roots", {})
            if not isinstance(raw_roots, dict):
                raise ValueError("recovery marker has invalid transaction roots")
            transaction.roots = {
                "tile": os.path.abspath(raw_roots.get("tile", root)),
                "shared": os.path.abspath(raw_roots.get("shared", root)),
                "mask": os.path.abspath(raw_roots.get("mask", root)),
            }
            transaction.root = transaction.roots["tile"]
            transaction.mask_root = transaction.roots["mask"]
            if transaction.include_overlay:
                overlay_root = raw_roots.get("overlay")
                if not overlay_root:
                    raise ValueError("recovery marker has no overlay root")
                transaction.overlay_root = os.path.abspath(overlay_root)
                transaction.roots["overlay"] = transaction.overlay_root
        transaction.marker_path = marker_path
        transaction.journal_path = journal_path
        transaction.tile_lat = int(tile.lat)
        transaction.tile_lon = int(tile.lon)
        transaction.config_paths = {
            "config": os.path.join(
                build_dir,
                "Ortho4XP_{}.cfg".format(FNAMES.short_latlon(tile.lat, tile.lon)),
            ),
            "backup": os.path.join(
                build_dir,
                "Ortho4XP_{}.cfg.bak".format(FNAMES.short_latlon(tile.lat, tile.lon)),
            ),
        }
        transaction.initial_config = dict(marker.get("initial_config", {}))
        transaction.best_config = marker.get("best_config")
        state = marker.get("state", "active")
        transaction.overlay_active = bool(
            transaction.include_overlay
            and state in ("committing", "discarding", "restoring")
        )
        best_snapshot = marker.get("best_snapshot")
        best_settings = marker.get("best_settings")
        best_config = marker.get("best_config")
        target_snapshot = marker.get("target_snapshot")

        # A v2 marker may have roots on separate volumes.  Validate every
        # root before using it so a stale/crafted marker cannot make recovery
        # move files outside the configured output locations.
        roots_to_validate = {
            "tile": (transaction.roots["tile"], parent_dir, ".o4xp-build-transaction-"),
            "shared": (transaction.roots["shared"], parent_dir, ".o4xp-build-transaction-"),
            "mask": (
                transaction.roots["mask"],
                os.path.dirname(mask_dir) or os.curdir,
                ".o4xp-mask-transaction-",
            ),
        }
        if transaction.include_overlay:
            roots_to_validate["overlay"] = (
                transaction.roots["overlay"],
                os.path.abspath(FNAMES.Overlay_dir),
                ".o4xp-overlay-transaction-",
            )
        for kind, (candidate_root, expected_parent, prefix) in roots_to_validate.items():
            expected_parent = os.path.abspath(expected_parent)
            if os.path.dirname(candidate_root) != expected_parent:
                # v1 puts all kinds under the build parent.  It has already
                # passed the historical validation above and remains valid.
                if version == 1 and kind in ("shared", "mask"):
                    continue
                raise ValueError("recovery {} root is outside its volume".format(kind))
            if not os.path.basename(candidate_root).startswith(prefix):
                if version == 1 and kind in ("shared", "mask"):
                    continue
                raise ValueError("recovery {} root has an invalid name".format(kind))

        if state == "snapshotting":
            # Standalone snapshot creation only links/copies into staging; the
            # canonical outputs have not been changed yet. Discard a partial
            # snapshot rather than attempting to restore incomplete inputs.
            transaction.cleanup()
            return True

        if state == "clearing":
            # The v2 snapshot is complete, but canonical generated files may
            # have been partially removed before the process stopped.
            transaction.discard_current("before-restore")
            transaction._write_marker("restoring", None, None, "initial")
            transaction.restore_snapshot_files("initial")
            transaction.restore_initial_config()
            transaction.cleanup()
            return True

        if state in ("complete", "restored"):
            transaction.cleanup()
            return True

        if version >= 2 and state in ("overlay-pending", "committing"):
            # Overlay generation and activation are part of the same public
            # transaction.  An interrupted commit is deliberately resolved
            # to the old state; the next invocation can then build again.
            target_snapshot = "initial"
            transaction._write_marker(
                "discarding", None, None, target_snapshot
            )
            transaction.discard_current("before-restore")
            transaction._write_marker(
                "restoring", None, None, target_snapshot
            )
            transaction.restore_snapshot_files(target_snapshot)
            transaction.restore_initial_config()
            transaction.cleanup()
            return True

        atomic_overlay_transaction = bool(
            version >= 2 and transaction.include_overlay
        )
        if state == "discarding":
            target_snapshot = (
                "initial"
                if atomic_overlay_transaction
                else target_snapshot or best_snapshot or "initial"
            )
            transaction.discard_current("before-restore")
            transaction._write_marker(
                "restoring",
                best_snapshot,
                best_settings,
                target_snapshot,
            )
            state = "restoring"

        if state == "restoring":
            target_snapshot = (
                "initial"
                if atomic_overlay_transaction
                else target_snapshot or best_snapshot or "initial"
            )
            transaction.restore_snapshot_files(target_snapshot)
            if target_snapshot == "initial":
                if version >= 2:
                    transaction.restore_initial_config()
                best_settings = None
                best_config = None
            else:
                state = "config-pending"

        elif state == "config-pending":
            if atomic_overlay_transaction:
                transaction._write_marker("discarding", None, None, "initial")
                transaction.discard_current("before-restore")
                transaction._write_marker("restoring", None, None, "initial")
                transaction.restore_snapshot_files("initial")
                transaction.restore_initial_config()
                transaction.cleanup()
                return True
            if best_snapshot and transaction.snapshot_has_files(best_snapshot):
                transaction.restore_snapshot_files(best_snapshot)
        else:
            snapshot_name = "initial" if atomic_overlay_transaction else best_snapshot or "initial"
            transaction._write_marker(
                "discarding",
                best_snapshot,
                best_settings,
                snapshot_name,
            )
            transaction.discard_current("before-restore")
            transaction._write_marker(
                "restoring",
                best_snapshot,
                best_settings,
                snapshot_name,
            )
            transaction.restore_snapshot_files(snapshot_name)
            if snapshot_name == "initial":
                if version >= 2:
                    transaction.restore_initial_config()
                best_settings = None
                best_config = None

        try:
            from O4_Config_Utils import list_tile_vars
            config_names = set(list_tile_vars)
        except (ImportError, AttributeError):
            config_names = set(_AUTO_REDUCE_SETTING_NAMES)
        if best_config:
            for name, value in best_config.items():
                if name in config_names:
                    setattr(tile, name, value)
        if best_settings:
            for name, value in best_settings.items():
                if name in _AUTO_REDUCE_SETTING_NAMES:
                    setattr(tile, name, value)
        if best_config or best_settings:
            if not tile.write_to_config():
                raise OSError("could not persist recovered tile configuration")
        transaction.cleanup()
        UI.vprint(
            0,
            UI.ui_text(
                "Recovered an interrupted tile build transaction.",
                "中断したタイルビルドの復元処理を完了しました。",
            ),
        )
        return True
    except Exception as error:
        UI.logprint("ERROR: Could not recover tile build transaction:", repr(error))
        UI.vprint(
            0,
            UI.ui_text(
                "ERROR: Could not recover the previous tile build safely: {}".format(
                    error
                ),
                "エラー: 前回のタイルビルドを安全に復元できません: {}".format(
                    error
                ),
            ),
        )
        return False


def _ashelper_capabilities(as_helper):
    """Probe all ASHelper capabilities once per executable revision."""
    try:
        stat = os.stat(as_helper)
        cache_key = (os.path.abspath(as_helper), stat.st_mtime_ns, stat.st_size)
    except OSError:
        cache_key = (os.path.abspath(as_helper), None, None)
    cached = _ASHELPER_CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    try:
        result = subprocess.run(
            [as_helper, "--capabilities"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        capabilities = {
            "metal_available": False,
            "metalfx_spatial_available": False,
            "tensorops_available": False,
            "fp8_tensorops_available": False,
            "shared_memory_version": 0,
            "shared_memory_input_format": None,
            "shared_memory_output_format": None,
            "shared_memory_max_buffer_bytes": 0,
            "probe_error": type(error).__name__,
        }
    else:
        values = {}
        for line in (result.stdout or "").splitlines():
            if "=" not in line:
                continue
            name, value = line.strip().split("=", 1)
            normalized = value.strip()
            lowered = normalized.lower()
            if lowered in ("true", "false"):
                values[name] = lowered == "true"
            else:
                try:
                    values[name] = int(normalized)
                except ValueError:
                    values[name] = normalized
        capabilities = {
            "metal_available": bool(result.returncode == 0 and values.get("metal_available")),
            "metalfx_spatial_available": bool(
                result.returncode == 0 and values.get("metalfx_spatial_available")
            ),
            "tensorops_available": bool(
                result.returncode == 0 and values.get("tensorops_available")
            ),
            "fp8_tensorops_available": bool(
                result.returncode == 0 and values.get("fp8_tensorops_available")
            ),
            "shared_memory_version": int(values.get("shared_memory_version", 0) or 0),
            "shared_memory_input_format": values.get("shared_memory_input_format"),
            "shared_memory_output_format": values.get("shared_memory_output_format"),
            "shared_memory_max_buffer_bytes": int(
                values.get("shared_memory_max_buffer_bytes", 0) or 0
            ),
        }
        if result.returncode != 0:
            capabilities["probe_error"] = f"exit_{result.returncode}"
    _ASHELPER_CAPABILITY_CACHE[cache_key] = dict(capabilities)
    if not capabilities["metal_available"]:
        UI.vprint(1, "WARNING: ASHelper Metal is unavailable; using CPU conversion.")
    return dict(capabilities)


def _ashelper_metal_available(as_helper):
    """Compatibility wrapper for callers that only need Metal."""
    return bool(_ashelper_capabilities(as_helper).get("metal_available"))


def _ashelper_metalfx_available(as_helper):
    """Probe MetalFX separately from the generic Metal DDS capability."""
    return bool(_ashelper_capabilities(as_helper).get("metalfx_spatial_available"))


def _ashelper_tensorops_available(as_helper):
    """Probe the macOS 27 TensorOps capability before batch deferral."""
    return bool(_ashelper_capabilities(as_helper).get("tensorops_available"))


def _opencl_capabilities():
    """Report OpenCL build/runtime flags without forcing a device dispatch."""
    try:
        import cv2

        return {
            "opencl_compiled": bool(cv2.ocl.haveOpenCL()),
            "opencl_enabled": bool(cv2.ocl.useOpenCL()),
        }
    except Exception as error:
        return {
            "opencl_compiled": False,
            "opencl_enabled": False,
            "opencl_probe_error": type(error).__name__,
        }


def _cpu_fallback_convert_args(convert_list, prepared_input_paths):
    """Build CPU conversion arguments while retaining prepared image work."""
    fallback_convert_list = []
    for item_index, item in enumerate(convert_list):
        prepared_file = (
            prepared_input_paths[item_index]
            if item_index < len(prepared_input_paths)
            else None
        )
        # A direct JPEG still needs the normal CPU-side color/mask
        # preprocessing. Only reuse files that already contain that work.
        if prepared_file:
            item_tile, item_x, item_y, item_z, item_provider = item
            # TensorOps batch output contains only the upscale.  The normal
            # GPU batch applies masks and color filters afterwards, so this
            # intermediate must not be reused by the CPU fallback when that
            # later batch fails.
            if IMG.normalize_upscale_backend(
                getattr(item_tile, "upscale_backend", "none")
            ) in ("tensorops", "metalfx_spatial"):
                prepared_file = None
            if prepared_file and item_provider in IMG.providers_dict:
                direct_cache = IMG.find_imagery_cache_path(
                    item_x,
                    item_y,
                    item_z,
                    item_provider,
                    FNAMES.jpeg_file_dir_from_attributes(
                        item_tile.lat,
                        item_tile.lon,
                        item_z,
                        IMG.providers_dict[item_provider],
                    ),
                )
                if direct_cache and os.path.abspath(prepared_file) == os.path.abspath(direct_cache):
                    prepared_file = None
        if not prepared_file or not os.path.isfile(prepared_file):
            prepared_file = None
        if prepared_file:
            fallback_convert_list.append((*item, "dds", prepared_file))
        else:
            fallback_convert_list.append(item)
    return fallback_convert_list


def _run_cpu_fallback(
    fallback_convert_list, config_data, max_slots, progress
):
    """Run all fallback conversions in the configured CPU pool."""
    cpu_config_data = dict(config_data)
    cpu_config_data.update(
        {
            "use_gpu_acceleration": False,
            "upscale_backend": "ci_lanczos",
            "defer_gpu_batch": False,
            "defer_fp8_batch": False,
            "preserve_batch_inputs": True,
        }
    )
    return multiprocessing_pool(
        IMG.convert_texture,
        fallback_convert_list,
        max_slots,
        progress=progress,
        init_func=IMG.init_worker,
        init_args=cpu_config_data,
    )


def _streaming_queue_size(tile):
    requested = int(
        getattr(tile, "conversion_queue_size", conversion_queue_size) or 0
    )
    if requested <= 0:
        requested = max(
            32,
            2 * max(1, int(getattr(tile, "max_convert_slots", max_convert_slots))),
        )
    return min(128, max(1, requested))


def _streaming_gpu_batch_size(tile):
    requested = int(getattr(tile, "gpu_batch_size", gpu_batch_size) or 0)
    # Keep the existing safe batch size as the initial automatic value.  The
    # setting is deliberately shared by the scheduler and future ASHelper
    # server implementation, while each backend may lower it before dispatch.
    automatic = 32
    if getattr(tile, "enable_shared_memory_handoff", enable_shared_memory_handoff):
        # Shared-memory input/output is intentionally conservative until the
        # real-tile acceptance run establishes a larger safe batch size.
        automatic = 8
    return max(1, requested if requested > 0 else automatic)


def _streaming_batch_wait_ms(tile):
    return max(
        0,
        int(getattr(tile, "gpu_batch_wait_ms", gpu_batch_wait_ms) or 0),
    )


def _start_ashelper_jsonl_server(tile, metrics=None):
    """Start one tile-local ASHelper process for raster/GPU work when useful."""
    use_gpu = bool(
        getattr(tile, "use_gpu_acceleration", getattr(UI, "use_gpu_acceleration", True))
    )
    dds_converter = getattr(
        tile, "dds_converter", getattr(UI, "dds_converter", "nvcompress")
    )
    wants_server = bool(
        use_gpu
        and (
            dds_converter == "TextureConverter"
            or getattr(tile, "use_gpu_for_masks", False)
            or getattr(tile, "use_gpu_for_dem_smoothing", False)
        )
    )
    if not wants_server:
        return None
    as_helper = os.path.join(UI.Ortho4XP_dir, "Utils", "mac", "ASHelper")
    if not os.path.isfile(as_helper) or not os.access(as_helper, os.X_OK):
        return None
    capabilities = _ashelper_capabilities(as_helper)
    capabilities.update(_opencl_capabilities())
    if metrics is not None:
        metrics.set_capabilities(capabilities)
    if not capabilities.get("metal_available", False):
        return None
    try:
        server = ASHelperJSONLServer(
            as_helper,
            logger=lambda message: UI.vprint(1, message),
        )
        server.start()
        tile._ashelper_jsonl_server = server
        return server
    except Exception as error:
        UI.vprint(
            1,
            "WARNING: ASHelper JSONL server could not start; retaining CPU/OpenCL "
            "fallbacks: {}".format(error),
        )
        return None


def _conversion_worker_config(tile, effective_gpu):
    """Build the spawn-worker globals used by a streaming CPU dispatcher."""
    return {
        "use_magick": IMG.use_magick,
        "use_texture_converter": getattr(IMG, "use_texture_converter", False),
        "dds_convert_cmd": IMG.dds_convert_cmd,
        "gdal_transl_cmd": IMG.gdal_transl_cmd,
        "gdalwarp_cmd": IMG.gdalwarp_cmd,
        "as_helper_cmd": getattr(IMG, "as_helper_cmd", None),
        "providers_dict": IMG.providers_dict,
        "local_combined_providers_dict": IMG.local_combined_providers_dict,
        "color_filters_dict": IMG.color_filters_dict,
        "extents_dict": IMG.extents_dict,
        "Ortho4XP_dir": UI.Ortho4XP_dir,
        "verbosity": UI.verbosity,
        "cleaning_level": UI.cleaning_level,
        "upscale_backend": IMG.normalize_upscale_backend(
            getattr(tile, "upscale_backend", "none")
        ),
        "upscale_scope": IMG.normalize_upscale_scope(
            getattr(tile, "upscale_scope", "all")
        ),
        "fp8_model_path": getattr(
            tile, "fp8_model_path", getattr(IMG, "fp8_model_path", "")
        ),
        "imagery_cache_format": getattr(IMG, "imagery_cache_format", "jpg"),
        "imagery_cache_quality": getattr(IMG, "imagery_cache_quality", ""),
        "dds_converter": getattr(
            tile, "dds_converter", getattr(UI, "dds_converter", "nvcompress")
        ),
        "dds_format": getattr(
            tile, "dds_format", getattr(UI, "dds_format", "BC3")
        ),
        "use_gpu_acceleration": bool(effective_gpu),
        "use_gpu_for_color_filters": getattr(
            tile, "use_gpu_for_color_filters", False
        ),
        "defer_gpu_batch": False,
        "defer_fp8_batch": False,
        "preserve_batch_inputs": False,
        "is_worker": True,
    }


def _streaming_gpu_source(item):
    """Return read-only source details for streaming GPU eligibility."""
    tile, til_x_left, til_y_top, zoomlevel, provider_code = item
    if provider_code not in IMG.providers_dict:
        return None
    if provider_code in IMG.local_combined_providers_dict:
        return None
    if IMG.should_upscale_texture(
        tile, til_x_left, til_y_top, zoomlevel, provider_code
    ):
        return None
    if not IMG.can_defer_gpu_batch_for_texture(
        tile, til_x_left, til_y_top, zoomlevel, provider_code
    ):
        return None

    out_file_name = FNAMES.dds_file_name_from_attributes(
        til_x_left, til_y_top, zoomlevel, provider_code
    )
    file_dir = FNAMES.jpeg_file_dir_from_attributes(
        tile.lat, tile.lon, zoomlevel, IMG.providers_dict[provider_code]
    )
    input_path = IMG.find_imagery_cache_path(
        til_x_left, til_y_top, zoomlevel, provider_code, file_dir
    )
    if not input_path or not IMG._jpeg_file_is_ready(input_path):
        return None
    if input_path.lower().endswith(".webp"):
        return None

    with Image.open(input_path) as source_image:
        input_size = source_image.size

    return {
        "tile": tile,
        "til_x_left": til_x_left,
        "til_y_top": til_y_top,
        "zoomlevel": zoomlevel,
        "provider_code": provider_code,
        "out_file_name": FNAMES.dds_file_name_from_attributes(
            til_x_left, til_y_top, zoomlevel, provider_code
        ),
        "input_path": input_path,
        "input_size": input_size,
    }


def _build_streaming_gpu_spec(task_id, item, dds_format):
    """Prepare one direct-cache conversion for the resident ASHelper route.

    The first streaming GPU lane intentionally handles only opaque provider
    cache inputs without an upscale.  Combined providers, WebP inputs, and
    upscale backends stay on the existing CPU/legacy paths until their
    intermediate-image contracts are moved into the JSONL protocol.
    """
    source = _streaming_gpu_source(item)
    if source is None:
        return None
    tile = source["tile"]
    til_x_left = source["til_x_left"]
    til_y_top = source["til_y_top"]
    zoomlevel = source["zoomlevel"]
    provider_code = source["provider_code"]
    out_file_name = source["out_file_name"]
    input_path = source["input_path"]
    input_size = source["input_size"]

    png_file_name = out_file_name.replace("dds", "png")
    mask_path = "none"
    generated_mask_path = None
    if tile.imprint_masks_to_dds:
        mask_path, generated_mask_path = _resolve_gpu_batch_mask(
            tile,
            til_x_left,
            til_y_top,
            zoomlevel,
            provider_code,
            png_file_name,
        )

    color_code = IMG.providers_dict[provider_code].get("color_filters", "none")
    r, g, b = 1.0, 1.0, 1.0
    contrast, brightness, saturation = 1.0, 0.0, 1.0
    for color_filter in IMG.color_filters_dict.get(color_code, []):
        filter_name = color_filter[0]
        if filter_name == "brightness-contrast":
            brightness_value, contrast_value = color_filter[1:3]
            brightness = brightness_value / 255.0
            contrast = 1.0 + (contrast_value / 128.0)
        elif filter_name == "saturation":
            saturation = 1.0 + (color_filter[1] / 100.0)

    has_alpha = False
    if mask_path != "none":
        with Image.open(mask_path) as mask_image:
            has_alpha = mask_image.convert("L").getextrema()[0] < 255
    target_format = IMG.resolve_dds_format(dds_format, has_alpha)
    if target_format not in ("BC1", "BC3"):
        if generated_mask_path:
            try:
                os.remove(generated_mask_path)
            except OSError:
                pass
        return None

    final_path = os.path.join(tile.build_dir, "textures", out_file_name)
    temporary_path = final_path + ".gpu.tmp.dds"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    try:
        os.remove(temporary_path)
    except OSError:
        pass
    return {
        "task_id": task_id,
        "item": item,
        "request": {
            "id": task_id,
            "input": input_path,
            "mask": mask_path,
            "output": temporary_path,
            "format": target_format,
            "r": r,
            "g": g,
            "b": b,
            "contrast": contrast,
            "brightness": brightness,
            "saturation": saturation,
        },
        "input_size": input_size,
        "temporary_path": temporary_path,
        "final_path": final_path,
        "target_format": target_format,
        "cleanup_paths": [generated_mask_path] if generated_mask_path else [],
    }


def _shared_memory_spec(spec, budget, generation, max_buffer_bytes=0):
    """Attach an RGB8 input and DDS output shared-memory lease to a spec.

    The path-based request remains present as a compatibility fallback.  The
    ASHelper shared transport reads only the versioned descriptors below.
    """
    request = spec["request"]
    if request.get("mask") not in (None, "none", ""):
        return None
    if any(
        abs(float(request.get(name, default)) - default) > 1e-9
        for name, default in (
            ("r", 1.0),
            ("g", 1.0),
            ("b", 1.0),
            ("contrast", 1.0),
            ("brightness", 0.0),
            ("saturation", 1.0),
        )
    ):
        return None
    input_path = request.get("input")
    if not input_path:
        return None
    try:
        with Image.open(input_path) as source_image:
            rgb_image = source_image.convert("RGB")
            width, height = rgb_image.size
            raw_rgb = rgb_image.tobytes()
        output_capacity = SHMEM.dds_capacity_bytes(
            width,
            height,
            spec["target_format"],
        )
        max_buffer = int(max_buffer_bytes or 0)
        if max_buffer > 0 and output_capacity > max_buffer:
            return None
        total_bytes = len(raw_rgb) + output_capacity
        if not budget.acquire(total_bytes, timeout=5.0):
            return None
        input_region = SHMEM.SharedMemoryRegion(len(raw_rgb), label="rgb_input")
        output_region = SHMEM.SharedMemoryRegion(output_capacity, label="dds_output")
        input_region.write(raw_rgb)
        request["input_shared_memory"] = input_region.descriptor(
            width=width,
            height=height,
            stride=width * 3,
            pixel_format=SHMEM.DEFAULT_INPUT_PIXEL_FORMAT,
            used_bytes=len(raw_rgb),
            read_only=True,
        )
        request["output_shared_memory"] = output_region.descriptor(
            width=width,
            height=height,
            stride=0,
            pixel_format=SHMEM.DEFAULT_OUTPUT_PIXEL_FORMAT,
            used_bytes=0,
            read_only=False,
        )
        request["generation"] = str(generation)
        shared_spec = dict(spec)
        shared_spec["shared_memory"] = True
        shared_spec["shared_memory_bytes"] = total_bytes
        shared_spec["input_region"] = input_region
        shared_spec["output_region"] = output_region
        shared_spec["input_dimensions"] = (width, height)
        shared_spec["shared_generation"] = str(generation)
        shared_spec["request"] = request
        return shared_spec
    except Exception:
        try:
            budget.release(total_bytes)
        except (UnboundLocalError, NameError):
            pass
        for region_name in ("input_region", "output_region"):
            region = locals().get(region_name)
            if region is not None:
                region.close()
        return None


class _StreamingConversionRunner:
    """Bridge downloads to bounded CPU and resident GPU conversion routes."""

    def __init__(self, tile):
        self.tile = tile
        self.metrics = getattr(tile, "_performance_metrics", None)
        self.started = time.perf_counter()
        self._ids = count(1)
        self._submitted_keys = set()
        dds_converter = getattr(
            tile, "dds_converter", getattr(UI, "dds_converter", "nvcompress")
        )
        self.dds_converter = dds_converter
        self.dds_format = getattr(
            tile, "dds_format", getattr(UI, "dds_format", "BC3")
        )
        try:
            self.gpu_parallelism = max(
                1,
                min(
                    12,
                    int(getattr(tile, "gpu_dds_workers", gpu_dds_workers)),
                ),
            )
        except (TypeError, ValueError):
            self.gpu_parallelism = 1
        use_gpu = bool(
            getattr(tile, "use_gpu_acceleration", getattr(UI, "use_gpu_acceleration", True))
        )
        self.as_helper = os.path.join(UI.Ortho4XP_dir, "Utils", "mac", "ASHelper")
        gpu_requested = bool(
            use_gpu
            and dds_converter == "TextureConverter"
            and "dar" in sys.platform
        )
        self.capabilities = (
            _ashelper_capabilities(self.as_helper)
            if gpu_requested and os.path.isfile(self.as_helper)
            else {}
        )
        self.shared_memory_enabled = bool(
            getattr(
                tile,
                "enable_shared_memory_handoff",
                enable_shared_memory_handoff,
            )
            and self.capabilities.get("shared_memory_version", 0) >= 1
            and self.capabilities.get("shared_memory_input_format") == "RGB8"
            and self.capabilities.get("shared_memory_output_format") == "DDS"
        )
        self.shared_memory_budget = None
        if self.shared_memory_enabled:
            self.shared_memory_budget = SHMEM.SharedMemoryBudget(
                SHMEM.shared_memory_budget_bytes(
                    getattr(
                        tile,
                        "shared_memory_budget_gb",
                        shared_memory_budget_gb,
                    )
                )
            )
        self.shared_memory_session_id = uuid.uuid4().hex
        self._shared_specs_inflight = []
        self.effective_gpu = bool(
            use_gpu
            and (not gpu_requested or self.capabilities.get("metal_available", False))
        )
        self.gpu_server = None
        self._owns_gpu_server = False
        shared_server = getattr(tile, "_ashelper_jsonl_server", None)
        if shared_server is not None and not shared_server.gpu_disabled:
            self.gpu_server = shared_server
        if (
            self.gpu_server is None
            and gpu_requested
            and self.effective_gpu
            and os.path.isfile(self.as_helper)
            and os.access(self.as_helper, os.X_OK)
        ):
            try:
                self.gpu_server = ASHelperJSONLServer(
                    self.as_helper,
                    logger=lambda message: UI.vprint(1, message),
                )
                self.gpu_server.start()
                self._owns_gpu_server = True
            except Exception as error:
                UI.vprint(
                    1,
                    "WARNING: ASHelper JSONL server is unavailable; "
                    "streaming GPU conversion will use CPU fallback: {}".format(error),
                )
                self.gpu_server = None
        if self.metrics is not None:
            self.metrics.set_capabilities(
                dict(
                    self.capabilities,
                    ashelper_jsonl_server=self.gpu_server is not None,
                    shared_memory_handoff=self.shared_memory_enabled,
                )
            )
            self.metrics.set_value(
                "shared_memory_budget_bytes",
                self.shared_memory_budget.capacity_bytes
                if self.shared_memory_budget is not None
                else 0,
            )
        # The resident server owns the GPU lane. CPU fallback workers must not
        # independently initialize Metal for the same tile.
        worker_gpu = self.effective_gpu and self.gpu_server is None
        self.config_data = _conversion_worker_config(tile, worker_gpu)
        self.pool = ReusableMultiprocessingPool(
            max(1, int(getattr(tile, "max_convert_slots", max_convert_slots))),
            init_func=IMG.init_worker,
            init_args=self.config_data,
        )
        self.progress = {"done": 0, "bar": 3, "message": "Converting DDS textures"}
        self.scheduler = TileConversionScheduler(
            dispatch_cpu=self._dispatch_cpu,
            dispatch_gpu=self._dispatch_gpu if self.gpu_server is not None else None,
            gpu_eligible=self._gpu_eligible,
            queue_size=_streaming_queue_size(tile),
            cpu_batch_size=max(
                1,
                int(getattr(tile, "max_convert_slots", max_convert_slots)),
            ),
            gpu_batch_size=_streaming_gpu_batch_size(tile),
            batch_wait_ms=_streaming_batch_wait_ms(tile),
            metrics=self.metrics,
            logger=lambda message: UI.vprint(1, message),
        )
        self.scheduler.start()

    def submit(self, payload):
        payload = tuple(payload)
        if len(payload) >= 5:
            duplicate_key = tuple(payload[1:5])
            if duplicate_key in self._submitted_keys:
                if self.metrics is not None:
                    self.metrics.increment("conversion_duplicates_suppressed")
                UI.vprint(2, "Skipping duplicate texture conversion:", duplicate_key)
                return True
            self._submitted_keys.add(duplicate_key)
        task_id = "texture-{:08d}".format(next(self._ids))
        accepted = self.scheduler.submit(
            ConversionTask(task_id, payload),
            timeout=None,
        )
        if not accepted and len(payload) >= 5:
            self._submitted_keys.discard(tuple(payload[1:5]))
        return accepted

    def _dispatch_cpu(self, tasks):
        payloads = [task.payload for task in tasks]
        if self.metrics is not None:
            self.metrics.increment("conversion_batches_cpu")
        return bool(
            self.pool.run(
                IMG.convert_texture,
                payloads,
                progress=None,
            )
        )

    def _gpu_eligible(self, task):
        if (
            self.gpu_server is None
            or self.gpu_server.gpu_disabled
            or self.dds_converter != "TextureConverter"
        ):
            return False
        item = task.payload
        try:
            # Eligibility is deliberately read-only.  In particular, do not
            # call _build_streaming_gpu_spec here: that path may materialize
            # a mask, which would otherwise be generated and removed again
            # before the actual GPU dispatch.
            return _streaming_gpu_source(item) is not None
        except Exception:
            return False

    def _dispatch_gpu(self, tasks):
        try:
            return self._dispatch_gpu_impl(tasks)
        finally:
            for spec in self._shared_specs_inflight:
                self._release_shared_spec(spec)
            self._shared_specs_inflight = []

    def _dispatch_gpu_impl(self, tasks):
        if self.gpu_server is None or self.gpu_server.gpu_disabled:
            raise RuntimeError("ASHelper GPU server is disabled")
        path_specs = []
        for task in tasks:
            try:
                spec = _build_streaming_gpu_spec(
                    task.task_id, task.payload, self.dds_format
                )
            except Exception as error:
                spec = None
                UI.vprint(1, "WARNING: Could not prepare streaming GPU task:", error)
            if spec is None:
                return {
                    task.task_id: False
                    for task in tasks
                }
            path_specs.append(spec)

        specs = path_specs
        transport = "path"
        shared_specs = []
        generation = uuid.uuid4().hex
        if self.shared_memory_enabled and self.shared_memory_budget is not None:
            for spec in path_specs:
                shared_spec = _shared_memory_spec(
                    spec,
                    self.shared_memory_budget,
                    generation,
                    self.capabilities.get("shared_memory_max_buffer_bytes", 0),
                )
                if shared_spec is None:
                    if self.metrics is not None:
                        self.metrics.increment("shared_memory_fallback")
                    for prepared in shared_specs:
                        self._release_shared_spec(prepared)
                    shared_specs = []
                    break
                shared_specs.append(shared_spec)
            if shared_specs and len(shared_specs) == len(path_specs):
                specs = shared_specs
                transport = "shared_memory"
                if self.metrics is not None:
                    self.metrics.increment(
                        "shared_alloc_bytes",
                        sum(spec.get("shared_memory_bytes", 0) for spec in specs),
                    )
            elif shared_specs:
                for prepared in shared_specs:
                    self._release_shared_spec(prepared)
                shared_specs = []
        self._shared_specs_inflight = (
            list(specs) if transport == "shared_memory" else []
        )

        if self.metrics is not None:
            self.metrics.increment("conversion_batches_gpu")
            if transport == "shared_memory":
                self.metrics.increment("conversion_batches_shared_memory")
        try:
            response = self.gpu_server.convert_batch(
                [spec["request"] for spec in specs],
                gpu=True,
                parallelism=self.gpu_parallelism,
                transport=transport,
                server_session_id=self.shared_memory_session_id,
                generation=generation,
            )
        finally:
            if transport == "path":
                for spec in shared_specs:
                    self._release_shared_spec(spec)
        response_by_id = {
            result.get("id"): result
            for result in response.get("results", [])
            if isinstance(result, dict)
        }
        if transport == "shared_memory":
            shared_protocol_errors = {
                "unsupported_transport",
                "stale_generation",
                "invalid_shared_task",
                "shared_memory_descriptor_invalid",
                "shared_memory_geometry_invalid",
                "shared_input_invalid",
                "shared_output_invalid",
            }
            protocol_failure = any(
                result.get("error") in shared_protocol_errors
                for result in response_by_id.values()
                if isinstance(result, dict)
            )
            if protocol_failure:
                # The ASHelper process answered, so preserve the resident GPU
                # server and downgrade only the transport for this tile.
                self.shared_memory_enabled = False
                if self.metrics is not None:
                    self.metrics.increment("shared_memory_fallback")
                for spec in path_specs:
                    path_request = dict(spec["request"])
                    path_request.pop("input_shared_memory", None)
                    path_request.pop("output_shared_memory", None)
                    path_request.pop("generation", None)
                    spec["request"] = path_request
                specs = path_specs
                transport = "path"
                response = self.gpu_server.convert_batch(
                    [spec["request"] for spec in specs],
                    gpu=True,
                    parallelism=self.gpu_parallelism,
                    transport="path",
                    server_session_id=self.shared_memory_session_id,
                )
                response_by_id = {
                    result.get("id"): result
                    for result in response.get("results", [])
                    if isinstance(result, dict)
                }
        if transport == "shared_memory":
            retry_specs = []
            for spec in specs:
                result = response_by_id.get(spec["task_id"], {})
                required_bytes = int(result.get("required_bytes", 0) or 0)
                if result.get("error") == "output_buffer_too_small" and required_bytes > 0:
                    if self._resize_shared_output(spec, required_bytes):
                        retry_specs.append(spec)
            if retry_specs:
                retry_response = self.gpu_server.convert_batch(
                    [spec["request"] for spec in retry_specs],
                    gpu=True,
                    parallelism=self.gpu_parallelism,
                    transport="shared_memory",
                    server_session_id=self.shared_memory_session_id,
                    generation=generation,
                )
                for result in retry_response.get("results", []):
                    if isinstance(result, dict) and result.get("id"):
                        response_by_id[result["id"]] = result
        normalized = {}
        for spec in specs:
            task_id = spec["task_id"]
            result = response_by_id.get(task_id, {})
            ok = bool(result.get("ok"))
            error = result.get("error")
            if ok:
                if transport == "shared_memory":
                    used_bytes = int(result.get("used_bytes", 0) or 0)
                    payload = spec["output_region"].read(used_bytes)
                    valid, validation_error = IMG.validate_dds_bytes(
                        payload,
                        expected_format=spec["target_format"],
                        expected_dimensions=spec["input_size"],
                        require_mipmaps=True,
                    )
                    if valid:
                        try:
                            with open(spec["temporary_path"], "wb") as output_stream:
                                output_stream.write(payload)
                                output_stream.flush()
                                os.fsync(output_stream.fileno())
                            os.replace(spec["temporary_path"], spec["final_path"])
                        except OSError as publish_error:
                            ok = False
                            error = "atomic_publish:{}".format(
                                type(publish_error).__name__
                            )
                    else:
                        ok = False
                        error = validation_error or "invalid_dds"
                else:
                    valid, validation_error = IMG.validate_dds_file(
                        spec["temporary_path"],
                        expected_format=spec["target_format"],
                        expected_dimensions=spec["input_size"],
                        require_mipmaps=True,
                    )
                    if valid:
                        try:
                            os.replace(spec["temporary_path"], spec["final_path"])
                        except OSError as publish_error:
                            ok = False
                            error = "atomic_publish:{}".format(
                                type(publish_error).__name__
                            )
                    else:
                        ok = False
                        error = validation_error or "invalid_dds"
            if not ok:
                try:
                    os.remove(spec["temporary_path"])
                except OSError:
                    pass
            for cleanup_path in spec.get("cleanup_paths", ()):
                try:
                    os.remove(cleanup_path)
                except OSError:
                    pass
            normalized[task_id] = ConversionResult(
                task_id,
                ok,
                "gpu",
                error,
            )
        return normalized

    def _release_shared_spec(self, spec):
        input_region = spec.get("input_region")
        output_region = spec.get("output_region")
        for region in (input_region, output_region):
            if region is not None:
                region.close()
        if self.shared_memory_budget is not None:
            self.shared_memory_budget.release(spec.get("shared_memory_bytes", 0))

    def _resize_shared_output(self, spec, required_bytes):
        """Retry one shared-memory item once when ASHelper reports its size."""
        if self.shared_memory_budget is None:
            return False
        required_bytes = int(required_bytes)
        old_output = spec.get("output_region")
        if old_output is None or required_bytes <= old_output.size_bytes:
            return False
        input_region = spec.get("input_region")
        if input_region is None:
            return False
        old_total = int(spec.get("shared_memory_bytes", 0) or 0)
        old_output.close()
        self.shared_memory_budget.release(old_total)
        new_total = input_region.size_bytes + required_bytes
        if not self.shared_memory_budget.acquire(new_total, timeout=5.0):
            # The old region is deliberately not recreated here.  The caller
            # will route this item through the normal CPU fallback.
            spec["output_region"] = None
            spec["shared_memory_bytes"] = 0
            return False
        try:
            new_output = SHMEM.SharedMemoryRegion(required_bytes, label="dds_output_retry")
        except Exception:
            self.shared_memory_budget.release(new_total)
            spec["output_region"] = None
            spec["shared_memory_bytes"] = 0
            return False
        spec["output_region"] = new_output
        spec["shared_memory_bytes"] = new_total
        spec["request"]["output_shared_memory"] = new_output.descriptor(
            width=spec["input_size"][0],
            height=spec["input_size"][1],
            pixel_format=SHMEM.DEFAULT_OUTPUT_PIXEL_FORMAT,
            used_bytes=0,
            read_only=False,
        )
        return True

    def finish(self):
        if UI.is_cancel_requested():
            self.scheduler.cancel()
        else:
            self.scheduler.close()
        try:
            results = self.scheduler.wait()
            expected = self.scheduler.submitted_count
            success = (
                self.scheduler.error is None
                and len(results) == expected
                and (expected == 0 or all(result.ok for result in results))
            )
            return success, results
        finally:
            if self.gpu_server is not None and self._owns_gpu_server:
                self.gpu_server.close()
            if UI.is_cancel_requested() or self.scheduler.error is not None:
                self.pool.terminate()
            else:
                self.pool.close()
            if self.metrics is not None:
                self.metrics.record_stage(
                    "texture conversion",
                    (time.perf_counter() - self.started) * 1000.0,
                )


def _activate_dsf(dsf_tmp_path, dsf_path):
    """Atomically activate a completed DSF while preserving rollback safety."""
    backup_path = dsf_path + ".bak"
    had_existing = os.path.exists(dsf_path)
    if not os.path.isfile(dsf_tmp_path):
        raise FileNotFoundError(dsf_tmp_path)
    if had_existing:
        os.replace(dsf_path, backup_path)
    try:
        os.replace(dsf_tmp_path, dsf_path)
    except Exception:
        if had_existing and not os.path.exists(dsf_path) and os.path.exists(backup_path):
            os.replace(backup_path, dsf_path)
        raise


def _sync_terrain_load_centers(tile, validated_dimensions=None):
    """Synchronize generated terrain metadata with the final DDS headers.

    ``validated_dimensions`` is populated only by the current build after a
    DDS has passed the complete validator. Any other DDS still takes the
    historical header-reading path, so an untrusted existing file cannot
    bypass validation.
    """
    terrain_dir = os.path.join(tile.build_dir, "terrain")
    if not os.path.isdir(terrain_dir):
        raise FileNotFoundError(terrain_dir)

    validated_dimensions = validated_dimensions or {}
    staged = []
    synchronized_count = 0
    unchanged_count = 0
    rewritten_count = 0
    dimension_cache_hits = 0
    try:
        for dir_path, _, names in os.walk(terrain_dir):
            for name in names:
                if not _is_generated_terrain_name(name):
                    continue
                terrain_path = os.path.join(dir_path, name)
                with open(terrain_path, "r", encoding="utf-8") as stream:
                    lines = stream.readlines()

                base_texture = None
                load_center_index = None
                for index, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith("BASE_TEX_NOWRAP "):
                        base_texture = stripped.split(None, 1)[1]
                    elif stripped.startswith("LOAD_CENTER "):
                        load_center_index = index

                if not base_texture:
                    raise ValueError(
                        f"Generated terrain has no BASE_TEX_NOWRAP: {terrain_path}"
                    )
                if load_center_index is None:
                    raise ValueError(
                        f"Generated terrain has no LOAD_CENTER: {terrain_path}"
                    )

                dds_path = os.path.normpath(
                    os.path.join(os.path.dirname(terrain_path), base_texture)
                )
                cached_dimensions = validated_dimensions.get(os.path.abspath(dds_path))
                if cached_dimensions is not None:
                    width, height = cached_dimensions
                    dimension_cache_hits += 1
                else:
                    width, height = IMG.read_dds_dimensions(dds_path)
                tokens = lines[load_center_index].split()
                if len(tokens) != 5:
                    raise ValueError(
                        f"Invalid LOAD_CENTER in generated terrain: {terrain_path}"
                    )
                synchronized_count += 1
                if tokens[-1] == str(width):
                    unchanged_count += 1
                    continue
                tokens[-1] = str(width)
                lines[load_center_index] = " ".join(tokens) + "\n"

                temporary_path = terrain_path + ".tmp"
                with open(temporary_path, "w", encoding="utf-8") as stream:
                    stream.writelines(lines)
                    stream.flush()
                    os.fsync(stream.fileno())
                staged.append((temporary_path, terrain_path))
                rewritten_count += 1

        for temporary_path, terrain_path in staged:
            os.replace(temporary_path, terrain_path)
    except Exception:
        for temporary_path, _ in staged:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        raise

    tile._terrain_load_center_stats = {
        "files": synchronized_count,
        "unchanged": unchanged_count,
        "rewritten": rewritten_count,
        "dimension_cache_hits": dimension_cache_hits,
    }
    return synchronized_count


def _resolve_gpu_batch_mask(tile, til_x_left, til_y_top, zoomlevel, provider_code, png_file_name):
    """Return an exact or materialized mask path for an ASHelper batch task."""
    possible_mask_path = os.path.join(
        tile.build_dir,
        "textures",
        FNAMES.mask_file(til_x_left, til_y_top, zoomlevel, provider_code),
    )
    if os.path.exists(possible_mask_path):
        return possible_mask_path, None

    fallback_mask = MASK.needs_mask(
        tile, til_x_left, til_y_top, zoomlevel, provider_code
    )
    if not fallback_mask:
        return "none", None

    fallback_mask_path = os.path.join(
        UI.Ortho4XP_dir,
        "tmp",
        os.path.splitext(png_file_name)[0] + "_mask.png",
    )
    fallback_mask.convert("L").resize((4096, 4096), Image.BICUBIC).save(
        fallback_mask_path
    )
    return fallback_mask_path, fallback_mask_path


def _build_metalfx_direct_dds_spec(item, dds_format):
    """Build one opaque-provider request for ASHelper direct MetalFX DDS work."""
    tile, til_x_left, til_y_top, zoomlevel, provider_code = item
    out_file_name = FNAMES.dds_file_name_from_attributes(
        til_x_left, til_y_top, zoomlevel, provider_code
    )
    file_dir = FNAMES.jpeg_file_dir_from_attributes(
        tile.lat, tile.lon, zoomlevel, IMG.providers_dict[provider_code]
    )
    input_path = IMG.find_imagery_cache_path(
        til_x_left, til_y_top, zoomlevel, provider_code, file_dir
    )
    if not input_path or not IMG._jpeg_file_is_ready(input_path):
        raise FileNotFoundError(f"input source not found for {out_file_name}")

    with Image.open(input_path) as source_image:
        source_width, source_height = source_image.size

    png_file_name = out_file_name.replace("dds", "png")
    mask_path = "none"
    generated_mask_path = None
    if tile.imprint_masks_to_dds:
        mask_path, generated_mask_path = _resolve_gpu_batch_mask(
            tile,
            til_x_left,
            til_y_top,
            zoomlevel,
            provider_code,
            png_file_name,
        )

    color_code = IMG.providers_dict[provider_code].get("color_filters", "none")
    r, g, b = 1.0, 1.0, 1.0
    contrast, brightness, saturation = 1.0, 0.0, 1.0
    for color_filter in IMG.color_filters_dict.get(color_code, []):
        filter_name = color_filter[0]
        if filter_name == "brightness-contrast":
            brightness_value, contrast_value = color_filter[1:3]
            brightness = brightness_value / 255.0
            contrast = 1.0 + (contrast_value / 128.0)
        elif filter_name == "saturation":
            saturation = 1.0 + (color_filter[1] / 100.0)

    has_alpha = False
    if mask_path != "none":
        with Image.open(mask_path) as mask_image:
            has_alpha = mask_image.convert("L").getextrema()[0] < 255
    target_format = IMG.resolve_dds_format(dds_format, has_alpha)
    if target_format not in ("BC1", "BC3"):
        raise ValueError(f"unsupported direct DDS format: {target_format}")

    final_path = os.path.join(tile.build_dir, "textures", out_file_name)
    temporary_path = final_path + ".gpu.tmp.dds"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    try:
        os.remove(temporary_path)
    except OSError:
        pass

    cleanup_paths = []
    if generated_mask_path:
        cleanup_paths.append(generated_mask_path)
    exact_mask_path = os.path.join(
        tile.build_dir,
        "textures",
        FNAMES.mask_file(til_x_left, til_y_top, zoomlevel, provider_code),
    )
    if tile.imprint_masks_to_dds and os.path.isfile(exact_mask_path):
        cleanup_paths.append(exact_mask_path)

    return {
        "item": item,
        "request": {
            "input": input_path,
            "mask": mask_path,
            "output": temporary_path,
            "format": target_format,
            "color": {
                "r": r,
                "g": g,
                "b": b,
                "contrast": contrast,
                "brightness": brightness,
                "saturation": saturation,
            },
        },
        "input_size": (source_width, source_height),
        "temporary_path": temporary_path,
        "final_path": final_path,
        "target_format": target_format,
        "cleanup_paths": cleanup_paths,
    }


def _run_metalfx_direct_dds_batch(as_helper, specs, worker_limit=2, chunk_size=8):
    """Run bounded ASHelper direct-DDS chunks and atomically publish valid DDS files."""
    if not specs:
        return {
            "failed_items": [],
            "batch_tasks": 0,
            "batch_success": 0,
            "batch_fallback": 0,
            "batch_failed": 0,
            "batch_workers": 0,
            "batch_chunks": 0,
            "metalfx_ms": 0.0,
            "readback_ms": 0.0,
            "dds_ms": 0.0,
            "temporary_bytes": 0,
            "duration_ms": 0.0,
            "signal": None,
            "fallback_reasons": {},
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunk_size = max(1, int(chunk_size))
    chunks = [
        specs[index : index + chunk_size]
        for index in range(0, len(specs), chunk_size)
    ]
    batch_workers = min(4, max(1, int(worker_limit)), len(chunks))
    batch_started = time.perf_counter()
    stats = {
        "failed_items": [],
        "batch_tasks": len(specs),
        "batch_success": 0,
        "batch_fallback": 0,
        "batch_failed": 0,
        "batch_workers": batch_workers,
        "batch_chunks": len(chunks),
        "metalfx_ms": 0.0,
        "readback_ms": 0.0,
        "dds_ms": 0.0,
        "temporary_bytes": 0,
        "duration_ms": 0.0,
        "signal": None,
        "fallback_reasons": {},
    }

    def run_direct_chunk(chunk):
        os.makedirs(os.path.join(UI.Ortho4XP_dir, "tmp"), exist_ok=True)
        request_fd, request_path = tempfile.mkstemp(
            prefix=".metalfx-spatial-dds-",
            suffix=".json",
            dir=os.path.join(UI.Ortho4XP_dir, "tmp"),
        )
        request = {
            "version": 1,
            "items": [spec["request"] for spec in chunk],
        }
        try:
            with os.fdopen(request_fd, "w", encoding="utf-8") as stream:
                json.dump(request, stream, separators=(",", ":"))
            result = subprocess.run(
                [as_helper, "--metalfx-spatial-dds-batch", request_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return chunk, result
        finally:
            try:
                os.remove(request_path)
            except OSError:
                pass

    def mark_failed(spec, reason):
        stats["batch_failed"] += 1
        stats["failed_items"].append(spec["item"])
        stats["fallback_reasons"][reason] = (
            stats["fallback_reasons"].get(reason, 0) + 1
        )
        try:
            os.remove(spec["temporary_path"])
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        futures = {
            executor.submit(run_direct_chunk, chunk): chunk for chunk in chunks
        }
        completed = 0
        for future in as_completed(futures):
            try:
                chunk, batch_result = future.result()
            except Exception as error:
                # A failed child launch must not prevent other chunks from
                # publishing their already validated DDS files.
                chunk = futures[future]
                for spec in chunk:
                    mark_failed(spec, f"ashelper_exception:{type(error).__name__}")
                completed += len(chunk)
                UI.vprint(1, f"   MetalFX Spatial DDS batch: {completed}/{len(specs)}")
                continue

            output_level = 0 if batch_result.returncode != 0 else 2
            output_lines = (batch_result.stdout or "").splitlines()
            item_lines = [
                line for line in output_lines if line.startswith("metalfx_dds_item=")
            ]
            for line in output_lines:
                UI.vprint(output_level, "      " + line)

            for index, spec in enumerate(chunk):
                fields = {}
                if index < len(item_lines):
                    fields = dict(
                        field.split("=", 1)
                        for field in item_lines[index].split()
                        if "=" in field
                    )
                for field_name, stat_name in (
                    ("metalfx_ms", "metalfx_ms"),
                    ("readback_ms", "readback_ms"),
                    ("dds_ms", "dds_ms"),
                ):
                    try:
                        stats[stat_name] += float(fields.get(field_name, 0.0))
                    except (TypeError, ValueError):
                        pass

                temp_path = spec["temporary_path"]
                valid, dds_error = IMG.validate_dds_file(
                    temp_path,
                    expected_format=spec["target_format"],
                    expected_dimensions=(
                        spec["input_size"][0] * 2,
                        spec["input_size"][1] * 2,
                    ),
                    require_mipmaps=True,
                )
                if not valid:
                    reason = (
                        fields.get("fallback_reason")
                        or (f"ashelper_exit_{batch_result.returncode}"
                            if batch_result.returncode else None)
                        or dds_error
                        or "direct_dds_invalid"
                    )
                    mark_failed(spec, reason)
                    continue

                try:
                    output_bytes = os.path.getsize(temp_path)
                    os.replace(temp_path, spec["final_path"])
                except (OSError, ValueError) as error:
                    mark_failed(spec, f"atomic_publish:{type(error).__name__}")
                    continue

                stats["batch_success"] += 1
                stats["temporary_bytes"] += output_bytes
                if fields.get("effective_backend") == "ci_lanczos":
                    stats["batch_fallback"] += 1
                    if fields.get("fallback_reason"):
                        reason = fields["fallback_reason"]
                        stats["fallback_reasons"][reason] = (
                            stats["fallback_reasons"].get(reason, 0) + 1
                        )

            completed += len(chunk)
            UI.vprint(1, f"   MetalFX Spatial DDS batch: {completed}/{len(specs)}")

    stats["duration_ms"] = (time.perf_counter() - batch_started) * 1000.0
    fallback_reasons = stats["fallback_reasons"]
    if stats["batch_success"] == 0:
        effective_backend = "failed"
    elif stats["batch_failed"] > 0:
        effective_backend = "mixed"
    elif stats["batch_fallback"] == stats["batch_success"]:
        effective_backend = "ci_lanczos"
    elif stats["batch_fallback"] == 0:
        effective_backend = "metalfx_spatial"
    else:
        effective_backend = "mixed"
    UI.vprint(
        1,
        "   MetalFX direct DDS summary: "
        f"backend=metalfx_spatial effective_backend={effective_backend} "
        f"dispatch=direct_dds png_intermediate=false batch_tasks={stats['batch_tasks']} "
        f"batch_success={stats['batch_success']} batch_fallback={stats['batch_fallback']} "
        f"batch_failed={stats['batch_failed']} batch_workers={stats['batch_workers']} "
        f"batch_chunks={stats['batch_chunks']} metalfx_ms={stats['metalfx_ms']:.2f} "
        f"readback_ms={stats['readback_ms']:.2f} dds_ms={stats['dds_ms']:.2f} "
        f"temporary_bytes={stats['temporary_bytes']} duration_ms={stats['duration_ms']:.2f}"
        + (
            " fallback_reasons="
            + ",".join(
                f"{reason}:{count}" for reason, count in sorted(fallback_reasons.items())
            )
            if fallback_reasons
            else ""
        ),
    )
    return stats


def _build_tensorops_direct_dds_spec(item, dds_format):
    """Build one opaque-provider request for TensorOps direct DDS work."""
    spec = _build_metalfx_direct_dds_spec(item, dds_format)
    spec["tensorops"] = True
    return spec


def _run_tensorops_direct_dds_batch(
    as_helper,
    pack_path,
    specs,
    chunk_size=8,
    worker_limit=1,
):
    """Run bounded TensorOps children and publish valid DDS files atomically."""
    if not specs:
        return {
            "failed_items": [],
            "failed_specs": [],
            "batch_tasks": 0,
            "batch_success": 0,
            "batch_fallback": 0,
            "batch_failed": 0,
            "batch_workers": 0,
            "batch_chunks": 0,
            "chunk_size": max(1, int(chunk_size)),
            "effective_backend": "none",
            "tensorops_dispatch_observed": False,
            "peak_rss_mb": 0,
            "rss_after_item_mb": 0,
            "rss_scope": "child_wait4" if hasattr(os, "wait4") else "unavailable",
            "temporary_bytes": 0,
            "duration_ms": 0.0,
            "fallback_reasons": {},
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunk_size = max(1, int(chunk_size))
    chunks = [
        specs[index : index + chunk_size]
        for index in range(0, len(specs), chunk_size)
    ]
    try:
        batch_workers = min(4, max(1, int(worker_limit)), len(chunks))
    except (TypeError, ValueError):
        batch_workers = 1
    started = time.perf_counter()
    stats = {
        "failed_items": [],
        "failed_specs": [],
        "batch_tasks": len(specs),
        "batch_success": 0,
        "batch_fallback": 0,
        "batch_failed": 0,
        "batch_workers": batch_workers,
        "batch_chunks": len(chunks),
        "chunk_size": chunk_size,
        "effective_backend": "failed",
        "tensorops_dispatch_observed": False,
        "neural_accelerator_confirmed": False,
        "peak_rss_mb": 0,
        "rss_after_item_mb": 0,
        "rss_scope": "child_wait4" if hasattr(os, "wait4") else "unavailable",
        "temporary_bytes": 0,
        "duration_ms": 0.0,
        "signal": None,
        "fallback_reasons": {},
    }

    def record_reason(reason):
        stats["fallback_reasons"][reason] = (
            stats["fallback_reasons"].get(reason, 0) + 1
        )

    def mark_failed(spec, reason):
        if spec["item"] not in stats["failed_items"]:
            stats["batch_failed"] += 1
            stats["failed_items"].append(spec["item"])
            stats["failed_specs"].append(spec)
            record_reason(reason)
        try:
            os.remove(spec["temporary_path"])
        except OSError:
            pass

    def run_chunk(chunk):
        os.makedirs(os.path.join(UI.Ortho4XP_dir, "tmp"), exist_ok=True)
        request_fd, request_path = tempfile.mkstemp(
            prefix=".tensorops-dds-",
            suffix=".json",
            dir=os.path.join(UI.Ortho4XP_dir, "tmp"),
        )
        request = {
            "version": 1,
            "pack": pack_path,
            "fallback_to_ci": False,
            "items": [spec["request"] for spec in chunk],
        }
        try:
            with os.fdopen(request_fd, "w", encoding="utf-8") as stream:
                request_fd = None
                json.dump(request, stream, separators=(",", ":"))
            result, timing = _run_tensorops_child(
                [as_helper, "--tensorops-dds-batch", request_path]
            )
            return chunk, result, timing, None
        except Exception as error:
            return chunk, None, None, error
        finally:
            if request_fd is not None:
                try:
                    os.close(request_fd)
                except OSError:
                    pass
            try:
                os.remove(request_path)
            except OSError:
                pass

    def process_chunk(chunk, result, timing, error):
        if error is not None:
            reason = f"ashelper_exception:{type(error).__name__}"
            for spec in chunk:
                mark_failed(spec, reason)
            return reason

        if timing is not None:
            stats["peak_rss_mb"] = max(
                stats["peak_rss_mb"], float(timing.get("peak_rss_mb", 0.0))
            )
            stats["rss_scope"] = str(timing.get("rss_scope", "unknown"))

        output = result.stdout or ""
        output_lines = output.splitlines()
        item_lines = [
            line for line in output_lines
            if line.startswith("tensorops_dds_item=")
        ]
        output_level = 0 if result.returncode != 0 else 2
        for line in output_lines:
            UI.vprint(output_level, "      " + line)

        signal_reason = (
            f"process_signal_{abs(result.returncode)}"
            if result.returncode < 0
            else None
        )
        if signal_reason:
            stats["signal"] = abs(result.returncode)

        for index, spec in enumerate(chunk):
            fields = {}
            if index < len(item_lines):
                fields = dict(
                    field.split("=", 1)
                    for field in item_lines[index].split()
                    if "=" in field
                )
            try:
                item_rss_mb = int(
                    float(
                        fields.get(
                            "rss_after_item_mb", fields.get("rss_mb", 0)
                        )
                        or 0
                    )
                )
            except (TypeError, ValueError):
                item_rss_mb = 0
            # ``ru_maxrss`` from wait4 is the child-process peak.  The value
            # emitted by Swift is only a point-in-time snapshot after an item;
            # never merge that snapshot into the peak field.
            stats["rss_after_item_mb"] = max(
                stats["rss_after_item_mb"], item_rss_mb
            )
            if fields.get("tensorops_dispatch_observed", "").lower() == "true":
                stats["tensorops_dispatch_observed"] = True
            if fields.get("neural_accelerator_confirmed", "").lower() == "true":
                stats["neural_accelerator_confirmed"] = True

            temp_path = spec["temporary_path"]
            valid, dds_error = IMG.validate_dds_file(
                temp_path,
                expected_format=spec["target_format"],
                expected_dimensions=(
                    spec["input_size"][0] * 2,
                    spec["input_size"][1] * 2,
                ),
                require_mipmaps=True,
            )
            if not valid:
                reason = (
                    fields.get("fallback_reason")
                    or signal_reason
                    or (f"ashelper_exit_{result.returncode}" if result.returncode else None)
                    or dds_error
                    or "direct_dds_invalid"
                )
                mark_failed(spec, reason)
                continue

            try:
                output_bytes = os.path.getsize(temp_path)
                os.replace(temp_path, spec["final_path"])
            except (OSError, ValueError) as publish_error:
                mark_failed(spec, f"atomic_publish:{type(publish_error).__name__}")
                continue

            stats["batch_success"] += 1
            stats["temporary_bytes"] += output_bytes
            if fields.get("effective_backend") == "ci_lanczos":
                stats["batch_fallback"] += 1
                if fields.get("fallback_reason"):
                    record_reason(fields["fallback_reason"])

        if result.returncode < 0:
            for spec in chunk[len(item_lines):]:
                mark_failed(spec, signal_reason or f"process_signal_{abs(result.returncode)}")
        return signal_reason

    os.makedirs(os.path.join(UI.Ortho4XP_dir, "tmp"), exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=batch_workers) as executor:
        futures = {executor.submit(run_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                completed_chunk, result, timing, error = future.result()
            except Exception as future_error:
                completed_chunk, result, timing, error = (
                    chunk,
                    None,
                    None,
                    future_error,
                )
            reason = process_chunk(completed_chunk, result, timing, error)
            completed += len(completed_chunk)
            suffix = f" failed ({reason})" if reason and result is None else ""
            UI.vprint(
                1,
                f"   TensorOps direct DDS: {completed}/{len(specs)}{suffix}",
            )

    stats["duration_ms"] = (time.perf_counter() - started) * 1000.0
    if stats["batch_success"] == 0:
        stats["effective_backend"] = "failed"
    elif stats["batch_failed"] > 0:
        stats["effective_backend"] = "mixed"
    elif stats["batch_fallback"] == stats["batch_success"]:
        stats["effective_backend"] = "ci_lanczos"
    elif stats["batch_fallback"] == 0 and stats["batch_failed"] == 0:
        stats["effective_backend"] = "tensorops"
    else:
        stats["effective_backend"] = "mixed"
    UI.vprint(
        1,
        "   TensorOps direct DDS summary: "
        f"backend=tensorops effective_backend={stats['effective_backend']} dispatch=direct_dds "
        f"png_intermediate=false batch_tasks={stats['batch_tasks']} "
        f"batch_success={stats['batch_success']} batch_fallback={stats['batch_fallback']} "
        f"batch_failed={stats['batch_failed']} batch_workers={stats['batch_workers']} "
        f"batch_chunks={stats['batch_chunks']} chunk_size={stats['chunk_size']} "
        f"tensorops_dispatch_observed={str(stats['tensorops_dispatch_observed']).lower()} "
        f"neural_accelerator_confirmed={str(stats['neural_accelerator_confirmed']).lower()} "
        f"rss_scope={stats['rss_scope']} "
        f"peak_rss_mb={stats['peak_rss_mb']} rss_after_item_mb={stats['rss_after_item_mb']} "
        f"temporary_bytes={stats['temporary_bytes']} signal={stats['signal'] or 0} "
        f"duration_ms={stats['duration_ms']:.2f}"
        + (
            " fallback_reasons="
            + ",".join(
                f"{reason}:{count}"
                for reason, count in sorted(stats["fallback_reasons"].items())
            )
            if stats["fallback_reasons"]
            else ""
        ),
    )
    return stats

################################################################################
def download_textures(
    tile,
    download_queue,
    convert_queue=None,
    conversion_submit=None,
):
    UI.vprint(1, "-> Opening download queue with", max_download_slots, "workers.")

    def download_task(*texture_attributes):
        if IMG.build_jpeg_ortho(tile, *texture_attributes):
            payload = (tile, *texture_attributes)
            metrics = getattr(tile, "_performance_metrics", None)
            if metrics is not None:
                metrics.increment("textures_downloaded")
            if conversion_submit is not None:
                return int(bool(conversion_submit(payload)))
            if convert_queue is not None:
                convert_queue.put(payload)
                return 1
            return 0
        return 0

    dico_dl_progress = {"done": 0, "bar": 2, "message": "Downloading textures"}
    dl_workers = parallel_launch(
        download_task,
        download_queue,
        max_download_slots,
        progress=dico_dl_progress,
    )

    download_success = parallel_join(dl_workers)

    if UI.is_cancel_requested():
        UI.vprint(1, "Download process interrupted.")
        return 0

    if not download_success:
        UI.vprint(0, "ERROR: One or more orthophotos could not be downloaded.")
        return 0

    if dico_dl_progress["done"]:
        UI.vprint(1, " *Download of textures completed.")
    return 1

################################################################################
def _finish_standalone_build_transaction(transaction, succeeded):
    """Commit or restore the output snapshot for a standalone Step 3 run."""
    if transaction is None:
        return
    if succeeded:
        try:
            # Mark completion before cleanup so an interruption during
            # cleanup cannot make recovery discard a valid new tile.
            transaction._write_marker("complete", None, None)
            transaction.cleanup()
        except Exception as error:
            UI.logprint(
                "WARNING: Could not remove standalone tile transaction staging:",
                repr(error),
            )
            UI.vprint(
                1,
                "WARNING: Standalone tile transaction staging remains for recovery:",
                error,
            )
        return

    try:
        transaction._write_marker("discarding", None, None, "initial")
        transaction.discard_current("before-restore")
        transaction._write_marker("restoring", None, None, "initial")
        transaction.restore_snapshot_files("initial")
        # The initial hardlink snapshot has been consumed by the restore. A
        # later recovery must only clean the staging directory, not restore it
        # a second time over the already-restored canonical files.
        transaction._write_marker("restored", None, None)
        transaction.cleanup()
    except Exception as error:
        UI.logprint(
            "ERROR: Could not restore the previous standalone tile state:",
            repr(error),
        )
        UI.vprint(
            0,
            UI.ui_text(
                "ERROR: Could not restore the previous tile state safely: {}".format(
                    error
                ),
                "エラー: 以前のタイル状態を安全に復元できません: {}".format(error),
            ),
        )


def build_tile(tile, persist_config=True):
    try:
        IMG.validate_imagery_cache_settings()
    except ValueError as error:
        UI.vprint(
            0,
            UI.ui_text(
                "ERROR: Invalid imagery cache settings: {}".format(error),
                "エラー: 画像キャッシュ設定が不正です: {}".format(error),
            ),
        )
        return 0
    standalone_transaction = None
    if not UI.is_building_all:
        UI.initialize_build_log(tile.build_dir, tile)
        if not _recover_build_transaction(tile):
            UI.exit_message_and_bottom_line(
                UI.ui_text(
                    "ERROR: The previous tile build could not be recovered.",
                    "エラー: 前回のタイルビルドを復元できませんでした。",
                )
            )
            return 0
        try:
            standalone_transaction = _BuildTransaction(
                tile, preserve_inputs=True
            )
        except Exception as error:
            UI.logprint(
                "ERROR: Could not start standalone tile transaction:",
                repr(error),
            )
            UI.vprint(
                0,
                UI.ui_text(
                    "ERROR: Could not start a safe tile build: {}".format(error),
                    "エラー: 安全なタイルビルドを開始できません: {}".format(error),
                ),
            )
            return 0
    result = 0
    try:
        result = _build_tile(tile, persist_config=persist_config)
        return result
    finally:
        _finish_standalone_build_transaction(
            standalone_transaction, bool(result)
        )
        # A DSF can be fully written before a later imagery/download stage
        # fails.  Never leave that unactivated artifact behind: the next run
        # must either rebuild it or activate it atomically.
        if not result:
            dsf_tmp_path = os.path.join(
                tile.build_dir,
                "Earth nav data",
                FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf.tmp",
            )
            try:
                os.remove(dsf_tmp_path)
            except OSError:
                pass
        UI.is_working = 0
        UI.flush_build_log(tile.build_dir)

def _build_tile(tile, persist_config=True):
    if UI.is_working:
        return 0
    UI.is_working = 1
    if not UI.is_building_all and UI.active_cancel_event is None:
        UI.red_flag = False
    UI.logprint(
        "Step 3 for tile lat=", tile.lat, ", lon=", tile.lon, ": starting."
    )
    UI.vprint(
        0,
        "\nStep 3 : Building DSF/Imagery for tile "
        + FNAMES.short_latlon(tile.lat, tile.lon)
        + " : \n--------\n",
    )

    if not os.path.isfile(FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon)):
        UI.lvprint(
            0, "ERROR: A mesh file must first be constructed for the tile!"
        )
        UI.exit_message_and_bottom_line("")
        return 0

    timer = time.time()
    # Only dimensions from DDS files validated in this invocation are cached.
    # This local map is intentionally not persisted on the Tile object.
    validated_dds_dimensions = {}

    if persist_config and not tile.write_to_config():
        UI.exit_message_and_bottom_line("ERROR: Could not save tile configuration.")
        return 0

    if not IMG.initialize_local_combined_providers_dict(tile):
        UI.exit_message_and_bottom_line("")
        return 0

    try:
        if not os.path.exists(
            os.path.join(
                tile.build_dir,
                "Earth nav data",
                FNAMES.round_latlon(tile.lat, tile.lon),
            )
        ):
            os.makedirs(
                os.path.join(
                    tile.build_dir,
                    "Earth nav data",
                    FNAMES.round_latlon(tile.lat, tile.lon),
                )
            )
        if not os.path.isdir(os.path.join(tile.build_dir, "textures")):
            os.makedirs(os.path.join(tile.build_dir, "textures"))
        if UI.cleaning_level > 1 and not tile.grouped:
            for f in os.listdir(os.path.join(tile.build_dir, "textures")):
                if not _MASK_TEXTURE_PATTERN.search(f):
                    continue
                try:
                    os.remove(os.path.join(tile.build_dir, "textures", f))
                except OSError:
                    pass
        if not tile.grouped:
            terrain_dir = os.path.join(tile.build_dir, "terrain")
            if os.path.isdir(terrain_dir):
                for dir_path, _, names in os.walk(terrain_dir):
                    for name in names:
                        if not _is_generated_terrain_name(name):
                            continue
                        try:
                            os.remove(os.path.join(dir_path, name))
                        except OSError:
                            pass
        if not os.path.isdir(os.path.join(tile.build_dir, "terrain")):
            os.makedirs(os.path.join(tile.build_dir, "terrain"))
    except Exception as e:
        UI.lvprint(0, "ERROR: Cannot create tile subdirectories.")
        UI.vprint(3, e)
        UI.exit_message_and_bottom_line("")
        return 0

    download_queue = queue.Queue()
    convert_queue = queue.Queue()
    streaming_enabled = bool(
        getattr(tile, "enable_streaming_conversion", enable_streaming_conversion)
        and not skip_downloads
        and not skip_converts
    )
    streaming_runner = None
    download_launched = False
    convert_launched = False
    conversion_success = True
    dsf_state = {"result": 0, "error": None}
    download_state = {"result": 0, "error": None}

    def run_dsf():
        try:
            dsf_state["result"] = DSF.build_dsf(tile, download_queue)
        except Exception as error:
            dsf_state["error"] = error
            UI.vprint(0, "ERROR: DSF worker failed:", error)

    def run_downloads():
        try:
            download_state["result"] = download_textures(
                tile,
                download_queue,
                convert_queue if not streaming_enabled else None,
                streaming_runner.submit if streaming_runner is not None else None,
            )
        except Exception as error:
            download_state["error"] = error
            UI.vprint(0, "ERROR: Download worker failed:", error)

    build_dsf_thread = threading.Thread(
        target=run_dsf, name="Ortho4XP-DSF"
    )
    download_thread = threading.Thread(
        target=run_downloads, name="Ortho4XP-downloads"
    )
    build_dsf_thread.start()
    if not skip_downloads:
        if streaming_enabled:
            try:
                streaming_runner = _StreamingConversionRunner(tile)
            except Exception as error:
                UI.vprint(
                    0,
                    UI.ui_text(
                        "ERROR: Could not start streaming conversion: {}".format(error),
                        "エラー: ストリーミング変換を開始できません: {}".format(error),
                    ),
                )
                UI.cancel_operation("internal")
                build_dsf_thread.join()
                return 0
        download_thread.start()
        download_launched = True
        if not skip_converts:
            dico_conv_progress = {"done": 0, "bar": 3, "message": "Converting DDS textures"}
            convert_launched = True
    build_dsf_thread.join()
    if download_launched:
        for _ in range(max_download_slots):
            download_queue.put("quit")
        download_thread.join()
    if streaming_runner is not None:
        try:
            conversion_success, streaming_results = streaming_runner.finish()
            if streaming_results:
                UI.vprint(
                    1,
                    " *Streaming DDS conversion completed:"
                    + " success={}/{}".format(
                        sum(result.ok for result in streaming_results),
                        len(streaming_results),
                    ),
                )
        except Exception as error:
            conversion_success = False
            UI.vprint(0, "ERROR: Streaming DDS conversion failed:", error)
    if dsf_state["error"] is not None or not dsf_state["result"]:
        UI.exit_message_and_bottom_line("ERROR: DSF construction failed.")
        return 0
    if download_launched and (
        download_state["error"] is not None or not download_state["result"]
    ):
        UI.exit_message_and_bottom_line("ERROR: Texture download failed.")
        return 0
    if convert_launched and not streaming_enabled:
            dds_converter = getattr(tile, 'dds_converter', getattr(UI, 'dds_converter', 'nvcompress'))
            dds_format = getattr(tile, 'dds_format', getattr(UI, 'dds_format', 'BC3'))
            use_gpu = getattr(tile, 'use_gpu_acceleration', getattr(UI, 'use_gpu_acceleration', True))
            as_helper = os.path.join(UI.Ortho4XP_dir, "Utils", "mac", "ASHelper")
            gpu_converter_requested = (
                use_gpu
                and dds_converter == "TextureConverter"
                and "dar" in sys.platform
            )
            gpu_batch_requested = (
                gpu_converter_requested
                and os.path.isfile(as_helper)
                and os.access(as_helper, os.X_OK)
            )
            requested_upscale_backend = IMG.normalize_upscale_backend(
                getattr(tile, 'upscale_backend', 'none')
            )
            requested_upscale_scope = IMG.normalize_upscale_scope(
                getattr(tile, 'upscale_scope', 'all')
            )
            capabilities = (
                _ashelper_capabilities(as_helper)
                if (
                    gpu_converter_requested
                    or (
                        requested_upscale_backend == "tensorops"
                        and os.path.isfile(as_helper)
                        and os.access(as_helper, os.X_OK)
                    )
                )
                else {
                    "metal_available": False,
                    "metalfx_spatial_available": False,
                    "tensorops_available": False,
                    "fp8_tensorops_available": False,
                }
            )
            metal_available = bool(capabilities.get("metal_available"))
            metalfx_available = bool(
                capabilities.get("metalfx_spatial_available")
            )
            tensorops_available = bool(capabilities.get("tensorops_available"))
            metrics = getattr(tile, "_performance_metrics", None)
            if metrics is not None:
                metrics.set_capabilities(
                    dict(capabilities, **_opencl_capabilities())
                )
            gpu_batch_enabled = gpu_batch_requested and metal_available
            effective_gpu = use_gpu and (
                not gpu_converter_requested or metal_available
            )
            UI.vprint(
                1,
                "-> Starting multiprocessing pool with",
                max_convert_slots,
                f"workers for DDS conversion (Format: {dds_format}).",
            )
            config_data = {
                'use_magick': IMG.use_magick,
                'use_texture_converter': getattr(IMG, 'use_texture_converter', False),
                'dds_convert_cmd': IMG.dds_convert_cmd,
                'gdal_transl_cmd': IMG.gdal_transl_cmd,
                'gdalwarp_cmd': IMG.gdalwarp_cmd,
                'as_helper_cmd': getattr(IMG, 'as_helper_cmd', None),
                'providers_dict': IMG.providers_dict,
                'local_combined_providers_dict': IMG.local_combined_providers_dict,
                'color_filters_dict': IMG.color_filters_dict,
                'extents_dict': IMG.extents_dict,
                'Ortho4XP_dir': UI.Ortho4XP_dir,
                'verbosity': UI.verbosity,
                'cleaning_level': UI.cleaning_level,
                'upscale_backend': requested_upscale_backend,
                'upscale_scope': requested_upscale_scope,
                'fp8_model_path': getattr(tile, 'fp8_model_path', getattr(IMG, 'fp8_model_path', '')),
                'imagery_cache_format': getattr(IMG, 'imagery_cache_format', 'jpg'),
                'imagery_cache_quality': getattr(IMG, 'imagery_cache_quality', ''),
                'dds_converter': getattr(tile, 'dds_converter', dds_converter),
                'dds_format': getattr(tile, 'dds_format', dds_format),
                'use_gpu_acceleration': effective_gpu,
                'use_gpu_for_color_filters': getattr(tile, 'use_gpu_for_color_filters', False),
                'is_worker': True
            }
            # Collect conversion arguments from queue
            convert_list = []
            while not convert_queue.empty():
                item = convert_queue.get()
                if item != "quit":
                    convert_list.append(item)

            def can_defer_to_gpu_batch(item):
                item_tile, item_x, item_y, item_zoomlevel, provider_code = item
                return IMG.can_defer_gpu_batch_for_texture(
                    item_tile, item_x, item_y, item_zoomlevel, provider_code
                )

            def can_defer_to_tensorops_batch(item):
                item_tile, item_x, item_y, item_zoomlevel, provider_code = item
                if not IMG.should_upscale_texture(
                    item_tile, item_x, item_y, item_zoomlevel, provider_code
                ):
                    return False
                if provider_code not in IMG.providers_dict:
                    return False
                if provider_code in IMG.local_combined_providers_dict:
                    return False
                # The FP8 input remains the opaque RGB JPEG. Supported
                # color/mask preprocessing is applied by the existing DDS
                # batch after the model output is produced.
                return IMG.can_defer_gpu_batch_for_texture(
                    item_tile, item_x, item_y, item_zoomlevel, provider_code
                )

            def can_defer_to_metalfx_batch(item):
                item_tile, item_x, item_y, item_zoomlevel, provider_code = item
                if requested_upscale_backend != "metalfx_spatial":
                    return False
                if not gpu_batch_enabled or not metalfx_available:
                    return False
                if not IMG.should_upscale_texture(
                    item_tile, item_x, item_y, item_zoomlevel, provider_code
                ):
                    return False
                # Only direct provider cache images are sent to MetalFX. A
                # prepared/combined/RGBA image remains on the per-image path,
                # where ASHelper can perform its RGBA split safely.
                if provider_code not in IMG.providers_dict:
                    return False
                if provider_code in IMG.local_combined_providers_dict:
                    return False
                return IMG.can_defer_gpu_batch_for_texture(
                    item_tile, item_x, item_y, item_zoomlevel, provider_code
                )

            metalfx_batch_items = [
                item for item in convert_list if can_defer_to_metalfx_batch(item)
            ]
            tensorops_model_path = getattr(
                tile, 'fp8_model_path', getattr(IMG, 'fp8_model_path', '')
            )
            tensorops_pack_configured = bool(
                tensorops_model_path
                and os.path.isdir(tensorops_model_path)
                and os.path.isfile(os.path.join(tensorops_model_path, "manifest.json"))
            )
            tensorops_batch_items = [
                item for item in convert_list if can_defer_to_tensorops_batch(item)
            ]
            defer_tensorops_batch = bool(
                tensorops_available
                and requested_upscale_backend == "tensorops"
                and tensorops_pack_configured
                and tensorops_batch_items
            )
            if not defer_tensorops_batch:
                tensorops_batch_items = []
            direct_batch_items = metalfx_batch_items + tensorops_batch_items
            regular_convert_list = [
                item for item in convert_list if item not in direct_batch_items
            ]

            batch_items_eligible = bool(
                regular_convert_list
                and all(can_defer_to_gpu_batch(item) for item in regular_convert_list)
            )
            defer_gpu_batch = bool(
                gpu_batch_enabled
                and batch_items_eligible
                and requested_upscale_backend != "metalfx_spatial"
                and requested_upscale_backend != "tensorops"
            )
            config_data['defer_gpu_batch'] = defer_gpu_batch
            # TensorOps direct items are removed from the worker list before
            # this pool starts. Never ask the remaining workers to defer
            # again, otherwise an ineligible direct-provider item could be
            # counted as successful without producing a DDS.
            config_data['defer_fp8_batch'] = False

            dds_error = IMG.dds_format_support_error(dds_converter, dds_format)
            if dds_error:
                UI.vprint(1, f"ERROR: {dds_error}")
                success_count = 0
                conversion_success = False
            else:
                if regular_convert_list:
                    pool_success = multiprocessing_pool(
                        IMG.convert_texture,
                        regular_convert_list,
                        max_convert_slots,
                        progress=dico_conv_progress,
                        init_func=IMG.init_worker,
                        init_args=config_data
                    )
                    success_count = len(regular_convert_list) if pool_success else 0
                    conversion_success = bool(pool_success)
                else:
                    success_count = 0
                    conversion_success = True

            # Direct provider JPEGs are processed by ASHelper without creating
            # an 8192x8192 PNG intermediate.  Keep the number of child
            # processes bounded because each MetalFX readback is large.
            direct_metalfx_requested = bool(metalfx_batch_items)
            metalfx_direct_specs = []
            metalfx_direct_failed = []
            metalfx_direct_cleanup = []
            if conversion_success and direct_metalfx_requested:
                UI.vprint(
                    1,
                    "-> Executing MetalFX Spatial direct DDS batch "
                    f"({len(metalfx_batch_items)} images)...",
                )
                try:
                    for item in metalfx_batch_items:
                        spec = _build_metalfx_direct_dds_spec(item, dds_format)
                        metalfx_direct_specs.append(spec)
                        metalfx_direct_cleanup.extend(spec["cleanup_paths"])
                except Exception as error:
                    UI.vprint(1, f"WARNING: MetalFX direct DDS preparation failed: {error}")
                    for prepared_spec in metalfx_direct_specs:
                        for cleanup_path in prepared_spec.get("cleanup_paths", ()):
                            try:
                                os.remove(cleanup_path)
                            except OSError:
                                pass
                        try:
                            os.remove(prepared_spec["temporary_path"])
                        except OSError:
                            pass
                    metalfx_direct_specs = []
                    metalfx_direct_cleanup = []
                    metalfx_direct_failed = list(metalfx_batch_items)

                batch_result = _run_metalfx_direct_dds_batch(
                    as_helper,
                    metalfx_direct_specs,
                    worker_limit=2,
                    chunk_size=8,
                )
                metalfx_direct_failed.extend(batch_result["failed_items"])

                if metalfx_direct_failed:
                    fallback_progress = {
                        "done": 0,
                        "bar": 3,
                        "message": "MetalFX direct DDS fallback",
                    }
                    fallback_config = dict(config_data)
                    fallback_config["upscale_backend"] = "ci_lanczos"
                    fallback_success = _run_cpu_fallback(
                        metalfx_direct_failed,
                        fallback_config,
                        max_convert_slots,
                        fallback_progress,
                    )
                    success_count += batch_result["batch_success"]
                    success_count += len(metalfx_direct_failed) if fallback_success else 0
                    conversion_success = bool(
                        fallback_success and success_count == len(convert_list)
                    )
                else:
                    success_count += batch_result["batch_success"]
                    conversion_success = True

                for cleanup_path in sorted(set(metalfx_direct_cleanup)):
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass
                metalfx_batch_items = []

            # Direct DDS work is already complete.  The existing DDS batch
            # below handles only the remaining non-direct items.
            batch_convert_list = (
                regular_convert_list
                if direct_metalfx_requested or defer_tensorops_batch
                else convert_list
            )

            # TensorOps direct DDS work is split into bounded child processes.
            # The child exit is a hard memory-reclaim boundary; successful
            # chunks remain published when a later chunk fails.  TensorOps is
            # only allowed to use the number of children that fits inside the
            # memory plan, while a failed item gets a MetalFX retry before CI.
            if conversion_success and defer_tensorops_batch:
                UI.vprint(
                    1,
                    "-> Executing TensorOps direct DDS batch "
                    f"({len(tensorops_batch_items)} images)...",
                )
                tensorops_direct_specs = []
                tensorops_direct_failed = []
                tensorops_direct_cleanup = []
                tensorops_direct_failed_specs = []
                try:
                    for item in tensorops_batch_items:
                        spec = _build_tensorops_direct_dds_spec(item, dds_format)
                        tensorops_direct_specs.append(spec)
                        tensorops_direct_cleanup.extend(spec["cleanup_paths"])
                except Exception as error:
                    UI.vprint(1, f"WARNING: TensorOps direct DDS preparation failed: {error}")
                    for prepared_spec in tensorops_direct_specs:
                        for cleanup_path in prepared_spec.get("cleanup_paths", ()):
                            try:
                                os.remove(cleanup_path)
                            except OSError:
                                pass
                        try:
                            os.remove(prepared_spec["temporary_path"])
                        except OSError:
                            pass
                    tensorops_direct_specs = []
                    tensorops_direct_failed = list(tensorops_batch_items)

                memory_plan = _tensorops_memory_plan(
                    tensorops_direct_specs,
                    max_workers=max_convert_slots,
                    chunk_size=8,
                )
                UI.vprint(
                    1,
                    "   TensorOps memory plan: "
                    f"reason={memory_plan['reason']} "
                    f"memory_budget_mb={memory_plan['memory_budget_mb']} "
                    f"parent_rss_mb={memory_plan['parent_rss_mb']} "
                    f"estimated_worker_mb={memory_plan['estimated_worker_mb']} "
                    f"estimated_dds_mb={memory_plan['estimated_dds_mb']} "
                    f"estimated_total_mb={memory_plan['estimated_total_mb']} "
                    f"batch_workers={memory_plan['workers']}",
                )
                if tensorops_direct_specs and memory_plan["can_run"]:
                    batch_result = _run_tensorops_direct_dds_batch(
                        as_helper,
                        tensorops_model_path,
                        tensorops_direct_specs,
                        chunk_size=8,
                        worker_limit=memory_plan["workers"],
                    )
                    tensorops_direct_failed_specs = list(
                        batch_result.get("failed_specs", [])
                    )
                    tensorops_direct_failed = list(batch_result["failed_items"])
                elif tensorops_direct_specs:
                    tensorops_direct_failed_specs = list(tensorops_direct_specs)
                    tensorops_direct_failed = [
                        spec["item"] for spec in tensorops_direct_failed_specs
                    ]
                    batch_result = {
                        "batch_success": 0,
                        "failed_items": tensorops_direct_failed,
                        "failed_specs": tensorops_direct_failed_specs,
                        "batch_fallback": 0,
                        "batch_failed": len(tensorops_direct_failed),
                        "batch_workers": 0,
                        "batch_chunks": memory_plan["chunk_count"],
                        "effective_backend": "memory_budget_exceeded",
                    }
                else:
                    batch_result = {
                        "batch_success": 0,
                        "failed_items": list(tensorops_direct_failed),
                        "failed_specs": [],
                        "batch_fallback": 0,
                        "batch_failed": len(tensorops_direct_failed),
                        "batch_workers": 0,
                        "batch_chunks": 0,
                        "effective_backend": "preparation_failed",
                    }

                # Reuse the prepared opaque-DDS specs for the MetalFX retry so
                # the failed TensorOps item is re-read from its original JPEG.
                # A preparation failure has no reusable spec, so try to build
                # one and leave only the items that still fail for CI.
                metalfx_retry_specs = list(tensorops_direct_failed_specs)
                tensorops_items_without_spec = [
                    item for item in tensorops_direct_failed
                    if not any(spec["item"] == item for spec in metalfx_retry_specs)
                ]
                tensorops_fallback_unprepared = []
                for item in tensorops_items_without_spec:
                    try:
                        spec = _build_metalfx_direct_dds_spec(item, dds_format)
                    except Exception as error:
                        UI.vprint(
                            1,
                            "WARNING: TensorOps fallback preparation failed: "
                            f"{error}",
                        )
                        tensorops_fallback_unprepared.append(item)
                        continue
                    metalfx_retry_specs.append(spec)
                    tensorops_direct_cleanup.extend(spec["cleanup_paths"])

                remaining_specs = []
                if metalfx_retry_specs and metalfx_available:
                    UI.vprint(
                        1,
                        "   TensorOps fallback: retrying failed items with "
                        f"MetalFX ({len(metalfx_retry_specs)} images)...",
                    )
                    metalfx_result = _run_metalfx_direct_dds_batch(
                        as_helper,
                        metalfx_retry_specs,
                        worker_limit=2,
                        chunk_size=8,
                    )
                    failed_items = list(metalfx_result["failed_items"])
                    remaining_specs = [
                        spec for spec in metalfx_retry_specs
                        if spec["item"] in failed_items
                    ]
                    tensorops_direct_failed = tensorops_fallback_unprepared + [
                        spec["item"] for spec in remaining_specs
                    ]
                    metalfx_success_count = (
                        len(metalfx_retry_specs) - len(remaining_specs)
                    )
                else:
                    remaining_specs = list(metalfx_retry_specs)
                    tensorops_direct_failed = tensorops_fallback_unprepared + [
                        spec["item"] for spec in remaining_specs
                    ]
                    metalfx_success_count = 0

                fallback_success = True
                cpu_fallback_items = list(tensorops_direct_failed)
                if cpu_fallback_items:
                    fallback_config = dict(config_data)
                    fallback_config["upscale_backend"] = "ci_lanczos"
                    fallback_config["defer_fp8_batch"] = False
                    fallback_config["defer_gpu_batch"] = False
                    fallback_success = _run_cpu_fallback(
                        cpu_fallback_items,
                        fallback_config,
                        2,
                        {
                            "done": 0,
                            "bar": 3,
                            "message": "TensorOps direct DDS fallback",
                        },
                    )
                success_count += batch_result["batch_success"]
                success_count += metalfx_success_count
                success_count += len(cpu_fallback_items) if fallback_success else 0
                conversion_success = bool(
                    (not cpu_fallback_items or fallback_success)
                    and success_count == len(convert_list)
                )
                defer_gpu_batch = False
                defer_tensorops_batch = False
                for cleanup_path in sorted(set(tensorops_direct_cleanup)):
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass

            # GPU Batch DDS Conversion integration for macOS
            if conversion_success and defer_gpu_batch:
                import O4_RAMDisk_Utils
                UI.vprint(1, "-> Executing ultra-fast GPU Batch DDS Conversion via ASHelper...")
                batch_args = []
                batch_requests = []
                temp_files_to_delete = []
                batch_generated_mask_files = []
                prepared_input_paths = [None] * len(batch_convert_list)
                batch_output_specs = []
                batch_attempted = False

                def cleanup_generated_batch_masks():
                    for temp_file in batch_generated_mask_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

                def cleanup_batch_outputs():
                    for temp_path, _, _, _ in batch_output_specs:
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
                
                for item_index, item in enumerate(batch_convert_list):
                    tile, til_x_left, til_y_top, zoomlevel, provider_code = item
                    out_file_name = FNAMES.dds_file_name_from_attributes(til_x_left, til_y_top, zoomlevel, provider_code)
                    out_file_path = os.path.join(tile.build_dir, "textures", out_file_name)
                    png_file_name = out_file_name.replace("dds", "png")
                    upscale_backend = IMG.normalize_upscale_backend(
                        getattr(tile, "upscale_backend", "none")
                    )
                    upscale_enabled = IMG.should_upscale_texture(
                        tile, til_x_left, til_y_top, zoomlevel, provider_code
                    )
                    effective_upscale_backend = (
                        upscale_backend if upscale_enabled else "none"
                    )
                    metalfx_upscaled_tmp = os.path.join(
                        UI.Ortho4XP_dir,
                        "tmp",
                        out_file_name.replace(
                            ".dds", "_metalfx_spatial_upscaled.png"
                        ),
                    )
                    tensorops_upscaled_tmp = os.path.join(
                        UI.Ortho4XP_dir,
                        "tmp",
                        out_file_name.replace(
                        ".dds", "_tensorops_upscaled.png"
                        ),
                    )
                    ci_lanczos_upscaled_tmp = os.path.join(
                        UI.Ortho4XP_dir,
                        "tmp",
                        out_file_name.replace(".dds", "_ci_lanczos_upscaled.png"),
                    )
                    upscale_candidates = (
                        [ci_lanczos_upscaled_tmp]
                        if effective_upscale_backend == "ci_lanczos"
                        else [tensorops_upscaled_tmp, ci_lanczos_upscaled_tmp]
                        if effective_upscale_backend == "tensorops"
                        else [metalfx_upscaled_tmp, ci_lanczos_upscaled_tmp]
                    )
                    tmp_png = os.path.join(UI.Ortho4XP_dir, "tmp", png_file_name)
                    
                    source_is_cache = False
                    if provider_code in IMG.providers_dict:
                        file_dir = FNAMES.jpeg_file_dir_from_attributes(tile.lat, tile.lon, zoomlevel, IMG.providers_dict[provider_code])
                        jpeg_path = IMG.find_imagery_cache_path(
                            til_x_left, til_y_top, zoomlevel, provider_code, file_dir
                        )
                    else:
                        jpeg_path = None
                    
                    if effective_upscale_backend != "none" and any(
                        os.path.exists(path) for path in upscale_candidates
                    ):
                        input_path = next(
                            path for path in upscale_candidates if os.path.exists(path)
                        )
                        temp_files_to_delete.append(input_path)
                    elif upscale_backend == "none" and os.path.exists(tmp_png):
                        input_path = tmp_png
                        temp_files_to_delete.append(tmp_png)
                    elif jpeg_path and IMG._jpeg_file_is_ready(jpeg_path):
                        input_path = jpeg_path
                        source_is_cache = True
                    else:
                        UI.vprint(1, f"ERROR: Input source image not found for {out_file_name}")
                        conversion_success = False
                        batch_attempted = True
                        break

                    prepared_input_paths[item_index] = input_path
                    
                    fp8_batch_input = (
                        effective_upscale_backend == "tensorops"
                        and input_path == tensorops_upscaled_tmp
                    )
                    metalfx_batch_input = (
                        effective_upscale_backend == "metalfx_spatial"
                        and input_path == metalfx_upscaled_tmp
                    )
                    mask_input = source_is_cache or fp8_batch_input or metalfx_batch_input
                    mask_path = "none"
                    if mask_input and tile.imprint_masks_to_dds:
                        try:
                            mask_path, generated_mask_path = _resolve_gpu_batch_mask(
                                tile,
                                til_x_left,
                                til_y_top,
                                zoomlevel,
                                provider_code,
                                png_file_name,
                            )
                        except Exception as e:
                            UI.vprint(
                                1,
                                f"ERROR: Could not prepare fallback mask for {out_file_name}: {str(e)}",
                            )
                            conversion_success = False
                            batch_attempted = True
                            break
                        if generated_mask_path:
                            batch_generated_mask_files.append(generated_mask_path)
                        elif mask_path != "none":
                            temp_files_to_delete.append(mask_path)

                    if tile.imprint_masks_to_dds and provider_code in IMG.providers_dict:
                        exact_mask_path = os.path.join(
                            tile.build_dir,
                            "textures",
                            FNAMES.mask_file(
                                til_x_left, til_y_top, zoomlevel, provider_code
                            ),
                        )
                        if (
                            os.path.isfile(exact_mask_path)
                            and exact_mask_path not in temp_files_to_delete
                        ):
                            temp_files_to_delete.append(exact_mask_path)
                    
                    r, g, b = 1.0, 1.0, 1.0
                    contrast, brightness, saturation = 1.0, 0.0, 1.0
                    
                    color_code = "none"
                    color_filter_input = (
                        source_is_cache or fp8_batch_input or metalfx_batch_input
                    )
                    if color_filter_input and provider_code in IMG.providers_dict:
                        color_code = IMG.providers_dict[provider_code].get(
                            "color_filters", "none"
                        )
                        if not IMG.gpu_batch_color_filter_supported(color_code):
                            UI.vprint(
                                1,
                                f"WARNING: Using normal color preprocessing for {out_file_name} ({color_code}).",
                            )
                            conversion_success = False
                            batch_attempted = True
                            break
                        for color_filter in IMG.color_filters_dict.get(color_code, []):
                            filter_name = color_filter[0]
                            if filter_name == "brightness-contrast":
                                b_val, c_val = color_filter[1:3]
                                brightness = b_val / 255.0
                                contrast = 1.0 + (c_val / 128.0)
                            elif filter_name == "saturation":
                                s_val = color_filter[1]
                                saturation = 1.0 + (s_val / 100.0)
                    
                    has_alpha = False
                    if mask_path != "none":
                        try:
                            with Image.open(mask_path) as mask_image:
                                has_alpha = (
                                    mask_image.convert("L").getextrema()[0] < 255
                                )
                        except:
                            pass
                    if not has_alpha and not source_is_cache:
                        has_alpha = IMG._image_has_non_opaque_alpha(input_path)
                    
                    target_fmt = IMG.resolve_dds_format(dds_format, has_alpha)
                    batch_tmp_path = out_file_path + ".gpu.tmp.dds"
                    try:
                        os.remove(batch_tmp_path)
                    except OSError:
                        pass
                    batch_args.extend([
                        input_path, 
                        mask_path, 
                        str(r), str(g), str(b), 
                        str(contrast), str(brightness), str(saturation), 
                        batch_tmp_path,
                        target_fmt
                    ])
                    batch_requests.append(
                        {
                            "id": f"dds-{item_index + 1}",
                            "input": input_path,
                            "mask": mask_path,
                            "r": r,
                            "g": g,
                            "b": b,
                            "contrast": contrast,
                            "brightness": brightness,
                            "saturation": saturation,
                            "output": batch_tmp_path,
                            "format": target_fmt,
                        }
                    )
                    batch_output_specs.append(
                        (batch_tmp_path, out_file_path, target_fmt, input_path)
                    )
                
                if conversion_success and batch_args:
                    batch_attempted = True
                    chunk_size = 64
                    batch_started = time.perf_counter()
                    server = getattr(tile, "_ashelper_jsonl_server", None)
                    server_failed = bool(server is not None and server.gpu_disabled)
                    server_used = bool(
                        server is not None
                        and not server_failed
                        and metal_available
                    )
                    if server_failed:
                        # A resident helper that already failed is not retried
                        # through another GPU entry point. The existing CPU
                        # fallback below will reprocess the original inputs.
                        conversion_success = False
                    gpu_worker_count = max(
                        1,
                        min(
                            12,
                            int(
                                getattr(
                                    tile,
                                    "gpu_dds_workers",
                                    gpu_dds_workers,
                                )
                                or 8
                            ),
                        ),
                    )
                    gpu_backend_counts = {"metal": 0, "cpu": 0, "unknown": 0}
                    gpu_route_failed = bool(server_failed)
                    gpu_failure_items = len(batch_requests) if server_failed else 0
                    gpu_telemetry = {
                        "decode_ms": 0.0,
                        "mask_setup_ms": 0.0,
                        "color_setup_ms": 0.0,
                        "preprocess_ms": 0.0,
                        "compression_ms": 0.0,
                        "readback_ms": 0.0,
                        "write_ms": 0.0,
                        "total_ms": 0.0,
                        "rss_after_item_mb": 0.0,
                    }
                    gpu_telemetry_legacy_fields = {
                        "mask_setup_ms": "mask_ms",
                        "color_setup_ms": "color_ms",
                    }
                    if server_failed:
                        UI.vprint(
                            1,
                            "WARNING: Resident ASHelper is disabled; using CPU DDS fallback.",
                        )
                    elif server_used:
                        for start in range(0, len(batch_requests), chunk_size):
                            request_chunk = batch_requests[start : start + chunk_size]
                            try:
                                server_response = server.convert_batch(
                                    request_chunk,
                                    gpu=True,
                                    parallelism=gpu_worker_count,
                                )
                            except Exception as error:
                                UI.vprint(
                                    1,
                                    "ERROR: Resident ASHelper DDS conversion failed: "
                                    + str(error),
                                )
                                gpu_route_failed = True
                                gpu_failure_items += len(request_chunk)
                                conversion_success = False
                                break
                            server_results = server_response.get("results", [])
                            if not isinstance(server_results, list) or len(server_results) != len(request_chunk):
                                UI.vprint(
                                    1,
                                    "ERROR: Resident ASHelper returned an incomplete DDS batch.",
                                )
                                gpu_route_failed = True
                                gpu_failure_items += len(request_chunk)
                                conversion_success = False
                                break
                            chunk_failed_items = 0
                            for result in server_results:
                                if not bool(result.get("ok", False)):
                                    chunk_failed_items += 1
                                backend = str(result.get("backend", "unknown"))
                                if backend not in gpu_backend_counts:
                                    backend = "unknown"
                                gpu_backend_counts[backend] += 1
                                for field in gpu_telemetry:
                                    try:
                                        legacy_field = gpu_telemetry_legacy_fields.get(
                                            field, field
                                        )
                                        value = float(
                                            result.get(
                                                field,
                                                result.get(legacy_field, 0.0),
                                            )
                                            or 0.0
                                        )
                                        if field == "rss_after_item_mb":
                                            gpu_telemetry[field] = max(
                                                gpu_telemetry[field], value
                                            )
                                        else:
                                            gpu_telemetry[field] += value
                                    except (TypeError, ValueError):
                                        pass
                            gpu_failure_items += chunk_failed_items
                            if chunk_failed_items or not server_response.get("ok", False):
                                UI.vprint(
                                    1,
                                    "ERROR: Resident ASHelper returned a failed DDS batch.",
                                )
                                gpu_route_failed = True
                                if chunk_failed_items == 0:
                                    gpu_failure_items += len(request_chunk)
                                conversion_success = False
                                break
                    else:
                        for i in range(0, len(batch_args), chunk_size * 10):
                            chunk = batch_args[i:i + chunk_size * 10]
                            cmd = [as_helper, "--convert-batch-v3", "true"] + chunk
                            try:
                                batch_result = subprocess.run(
                                    cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    check=False,
                                )
                                ret = batch_result.returncode
                                summary_seen = False
                                if batch_result.stdout:
                                    output_level = 0 if ret != 0 else 2
                                    for line in batch_result.stdout.splitlines():
                                        UI.vprint(output_level, "      " + line)
                                        if line.startswith("backend=dds "):
                                            summary_seen = True
                                            fields = dict(
                                                field.split("=", 1)
                                                for field in line.split()
                                                if "=" in field
                                            )
                                            metal_items = int(fields.get("metal_items", 0))
                                            cpu_items = int(fields.get("cpu_fallback_items", 0))
                                            known_items = metal_items + cpu_items
                                            gpu_backend_counts["metal"] += metal_items
                                            gpu_backend_counts["cpu"] += cpu_items
                                            gpu_backend_counts["unknown"] += max(
                                                0,
                                                int(fields.get("batch_tasks", 0)) - known_items,
                                            )
                                            gpu_failure_items += int(fields.get("batch_failed", 0))
                                            for field in gpu_telemetry:
                                                try:
                                                    legacy_field = gpu_telemetry_legacy_fields.get(
                                                        field, field
                                                    )
                                                    value = float(
                                                        fields.get(
                                                            field,
                                                            fields.get(legacy_field, 0.0),
                                                        )
                                                        or 0.0
                                                    )
                                                    if field == "rss_after_item_mb":
                                                        gpu_telemetry[field] = max(
                                                            gpu_telemetry[field], value
                                                        )
                                                    else:
                                                        gpu_telemetry[field] += value
                                                except (TypeError, ValueError):
                                                    pass
                                if ret != 0:
                                    UI.vprint(1, f"ERROR: GPU Batch DDS conversion failed with return code {ret}")
                                    gpu_route_failed = True
                                    if not summary_seen:
                                        gpu_failure_items += len(chunk) // 10
                                    conversion_success = False
                                    break
                            except Exception as e:
                                UI.vprint(1, f"ERROR: Execution of GPU Batch DDS conversion failed: {str(e)}")
                                gpu_route_failed = True
                                gpu_failure_items += len(chunk) // 10
                                conversion_success = False
                                break
                    batch_duration_ms = (time.perf_counter() - batch_started) * 1000.0
                    if metrics is not None:
                        if gpu_route_failed:
                            batch_backend = "failed"
                        elif server_used:
                            batch_backend = (
                                "metal"
                                if gpu_backend_counts["metal"] == len(batch_requests)
                                else "mixed"
                                if gpu_backend_counts["metal"]
                                else "cpu"
                            )
                        else:
                            batch_backend = "metal" if metal_available else "cpu"
                        metrics.record_batch(
                            "gpu_dds",
                            len(batch_requests),
                            duration_ms=batch_duration_ms,
                            status="completed" if conversion_success else "fallback",
                        )
                        metrics.increment(
                            "gpu_dds_metal_items", gpu_backend_counts["metal"]
                        )
                        metrics.increment(
                            "gpu_dds_cpu_fallback_items", gpu_backend_counts["cpu"]
                        )
                        metrics.increment(
                            "gpu_dds_unknown_items", gpu_backend_counts["unknown"]
                        )
                        metrics.set_value(
                            "gpu_dds_backend",
                            batch_backend,
                        )
                        metrics.set_value(
                            "gpu_dds_worker_count",
                            gpu_worker_count if server_used and not gpu_route_failed else (0 if gpu_route_failed else 8),
                        )
                        metrics.set_value(
                            "gpu_dds_chunk_count",
                            (len(batch_requests) + chunk_size - 1) // chunk_size,
                        )
                        metrics.set_value(
                            "gpu_dds_items_per_second",
                            len(batch_requests) / (batch_duration_ms / 1000.0)
                            if batch_duration_ms > 0
                            else 0.0,
                        )
                        for field, value in gpu_telemetry.items():
                            metrics.set_value("gpu_dds_" + field, value)
                        if server is not None:
                            metrics.set_value(
                                "ashelper_restart_count",
                                getattr(server, "restart_count", 0),
                            )
                            metrics.set_value(
                                "ashelper_gpu_disabled",
                                bool(getattr(server, "gpu_disabled", False)),
                            )
                        metrics.increment(
                            "gpu_dds_failures",
                            gpu_failure_items,
                        )
                        UI.vprint(
                            1,
                            "   GPU DDS summary: "
                            f"backend={'failed' if gpu_route_failed else ('resident' if server_used else 'batch-v3')} "
                            f"workers={gpu_worker_count if server_used and not gpu_route_failed else (0 if gpu_route_failed else 8)} "
                            f"items={len(batch_requests)} "
                            f"metal={gpu_backend_counts['metal']} "
                            f"cpu_fallback={gpu_backend_counts['cpu']} "
                            f"duration_ms={batch_duration_ms:.2f}",
                        )
                    if conversion_success:
                        invalid_outputs = []
                        for temp_path, final_path, target_fmt, input_path in batch_output_specs:
                            try:
                                with Image.open(input_path) as source_image:
                                    expected_dimensions = source_image.size
                            except Exception as error:
                                invalid_outputs.append((temp_path, f"input inspection failed: {error}"))
                                continue
                            dds_valid, dds_error = IMG.validate_dds_file(
                                temp_path,
                                expected_format=target_fmt,
                                expected_dimensions=expected_dimensions,
                                require_mipmaps=True,
                            )
                            if not dds_valid:
                                invalid_outputs.append((temp_path, dds_error))
                            else:
                                validated_dds_dimensions[os.path.abspath(final_path)] = (
                                    expected_dimensions
                                )
                        if invalid_outputs:
                            if metrics is not None:
                                metrics.increment(
                                    "gpu_dds_validation_failures",
                                    len(invalid_outputs),
                                )
                            for path, reason in invalid_outputs:
                                UI.vprint(
                                    0,
                                    f"ERROR: GPU batch produced invalid DDS {path}: {reason}",
                                )
                            conversion_success = False
                        else:
                            try:
                                for temp_path, final_path, _, _ in batch_output_specs:
                                    os.replace(temp_path, final_path)
                            except OSError as error:
                                UI.vprint(0, "ERROR: Could not activate GPU batch DDS output:", error)
                                conversion_success = False

                if batch_attempted and not conversion_success:
                    cleanup_batch_outputs()
                    UI.vprint(1, "-> Falling back to CPU DDS conversion via ASHelper...")
                    fallback_convert_list = _cpu_fallback_convert_args(
                        batch_convert_list, prepared_input_paths
                    )
                    for item in batch_convert_list:
                        item_out_name = FNAMES.dds_file_name_from_attributes(
                            item[1], item[2], item[3], item[4]
                        )
                        fallback_tmp_png = os.path.join(
                            UI.Ortho4XP_dir,
                            "tmp",
                            item_out_name.replace("dds", "png"),
                        )
                        if fallback_tmp_png not in temp_files_to_delete:
                            temp_files_to_delete.append(fallback_tmp_png)
                        if item[4] in IMG.providers_dict and item[0].imprint_masks_to_dds:
                            fallback_mask = os.path.join(
                                item[0].build_dir,
                                "textures",
                                FNAMES.mask_file(item[1], item[2], item[3], item[4]),
                            )
                            if os.path.isfile(fallback_mask) and fallback_mask not in temp_files_to_delete:
                                temp_files_to_delete.append(fallback_mask)

                    fallback_progress = {
                        "done": 0,
                        "bar": 3,
                        "message": "CPU fallback DDS conversion",
                    }
                    fallback_success = _run_cpu_fallback(
                        fallback_convert_list,
                        config_data,
                        max_convert_slots,
                        fallback_progress,
                    )
                    success_count = len(batch_convert_list) if fallback_success else 0
                    conversion_success = bool(fallback_success)

                if conversion_success:
                    for temp_file in temp_files_to_delete:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    cleanup_generated_batch_masks()

            if not conversion_success:
                UI.lvprint(0, f"WARNING: {len(convert_list) - success_count} textures failed to convert.")
                UI.lvprint(0, "Skipping cleanup to protect existing data.")

            if UI.is_cancel_requested():
                UI.vprint(1, "DDS conversion process interrupted.")
            elif dico_conv_progress["done"] >= 1:
                UI.vprint(1, " *DDS conversion of textures completed.")
    if convert_launched and not conversion_success:
        UI.exit_message_and_bottom_line("ERROR: DDS conversion failed.")
        return 0
    if UI.is_cancel_requested():
        UI.exit_message_and_bottom_line()
        return 0
    performance_metrics = getattr(tile, "_performance_metrics", None)
    try:
        if performance_metrics is None:
            synced_terrain_count = _sync_terrain_load_centers(
                tile, validated_dds_dimensions
            )
        else:
            with performance_metrics.stage("terrain metadata"):
                synced_terrain_count = _sync_terrain_load_centers(
                    tile, validated_dds_dimensions
                )
        sync_stats = getattr(tile, "_terrain_load_center_stats", {})
        if performance_metrics is not None:
            performance_metrics.increment(
                "terrain_load_center_files",
                sync_stats.get("files", synced_terrain_count),
            )
            performance_metrics.increment(
                "terrain_load_center_rewrites",
                sync_stats.get("rewritten", 0),
            )
            performance_metrics.increment(
                "terrain_load_center_unchanged",
                sync_stats.get("unchanged", 0),
            )
            performance_metrics.increment(
                "terrain_load_center_dimension_cache_hits",
                sync_stats.get("dimension_cache_hits", 0),
            )
        UI.vprint(
            1,
            " *Synchronized LOAD_CENTER metadata for",
            synced_terrain_count,
            "terrain files.",
        )
    except Exception as error:
        UI.vprint(0, "ERROR: Could not synchronize terrain DDS metadata:", error)
        UI.exit_message_and_bottom_line("ERROR: Terrain metadata synchronization failed.")
        return 0
    UI.vprint(1, " *Activating DSF file.")
    dsf_file_name = os.path.join(
        tile.build_dir,
        "Earth nav data",
        FNAMES.long_latlon(tile.lat, tile.lon) + ".dsf",
    )
    try:
        if performance_metrics is None:
            _activate_dsf(dsf_file_name + ".tmp", dsf_file_name)
        else:
            with performance_metrics.stage("DSF activation"):
                _activate_dsf(dsf_file_name + ".tmp", dsf_file_name)
    except Exception as error:
        UI.vprint(0, "ERROR: could not activate DSF file; existing tile was preserved:", error)
        try:
            os.remove(dsf_file_name + ".tmp")
        except OSError:
            pass
        UI.exit_message_and_bottom_line()
        return 0
    if UI.cleaning_level > 1:
        try:
            os.remove(FNAMES.alt_file(tile))
        except:
            pass
        try:
            os.remove(FNAMES.input_node_file(tile))
        except:
            pass
        try:
            os.remove(FNAMES.input_poly_file(tile))
        except:
            pass
    if UI.cleaning_level > 2:
        try:
            os.remove(FNAMES.mesh_file(tile.build_dir, tile.lat, tile.lon))
        except:
            pass
        try:
            os.remove(FNAMES.apt_file(tile))
        except:
            pass
    if UI.cleaning_level > 1 and not tile.grouped and conversion_success:
        remove_unwanted_textures(tile)
    try:
        import O4_RAMDisk_Utils
        flush_result = O4_RAMDisk_Utils.flush_tile_imagery(tile.lat, tile.lon)
        if flush_result is False:
            UI.vprint(
                0,
                UI.ui_text(
                    "ERROR: RAM disk imagery flush failed; RAM data was retained.",
                    "エラー: RAMディスクの画像flushに失敗したため、RAMデータを保持しました。",
                ),
            )
            return 0
    except Exception as e:
        UI.vprint(
            0,
            UI.ui_text(
                f"ERROR: RAM disk imagery flush failed: {e}",
                f"エラー: RAMディスクの画像flushに失敗しました: {e}",
            ),
        )
        return 0
    UI.timings_and_bottom_line(timer)
    UI.logprint(
        "Step 3 for tile lat=", tile.lat, ", lon=", tile.lon, ": normal exit."
    )
    return 1

################################################################################
def _report_pipeline_failure(tile, stage_name, error=None, traceback_text=None):
    cancelled = bool(UI.is_cancel_requested())
    failure = {
        "stage": stage_name,
        "cancelled": cancelled,
        "error": repr(error) if error is not None else None,
    }
    tile.last_pipeline_failure = failure
    if error is not None:
        UI.logprint(
            "ERROR: Pipeline stage",
            stage_name,
            "raised:",
            repr(error),
            "\n",
            traceback_text or repr(error),
        )

    if cancelled:
        message = UI.ui_text(
            "ERROR: Tile build cancelled during {}.".format(stage_name),
            "エラー: {} の実行中にタイルビルドをキャンセルしました。".format(
                stage_name
            ),
        )
    elif error is not None:
        message = UI.ui_text(
            "ERROR: {} stage failed: {}".format(stage_name, error),
            "エラー: {} ステージに失敗しました: {}".format(stage_name, error),
        )
    else:
        message = UI.ui_text(
            "ERROR: {} stage failed.".format(stage_name),
            "エラー: {} ステージに失敗しました。".format(stage_name),
        )
    UI.vprint(0, message)
    UI.exit_message_and_bottom_line(message)


def _report_metrics_failure(tile):
    failure = {
        "stage": "DSF metrics",
        "cancelled": bool(UI.is_cancel_requested()),
        "error": "missing or structurally invalid DSF metrics",
    }
    tile.last_pipeline_failure = failure
    message = UI.ui_text(
        "ERROR: DSF metrics were missing or structurally invalid after a successful build.",
        "エラー: ビルド成功後のDSFメトリクスがないか、構造的に不正です。",
    )
    UI.logprint(message)
    UI.vprint(0, message)
    UI.exit_message_and_bottom_line(message)


################################################################################
def _start_full_pipeline(tile, include_overlays):
    metrics = PERF.PerformanceMetrics(tile, "all_in_one")
    tile._performance_metrics = metrics
    metrics.set_config(
        {
            "enable_streaming_conversion": bool(
                getattr(tile, "enable_streaming_conversion", enable_streaming_conversion)
            ),
            "conversion_queue_size": int(
                getattr(tile, "conversion_queue_size", conversion_queue_size)
            ),
            "gpu_batch_size": int(getattr(tile, "gpu_batch_size", gpu_batch_size)),
            "gpu_dds_workers": int(
                getattr(tile, "gpu_dds_workers", gpu_dds_workers)
            ),
            "gpu_batch_wait_ms": int(
                getattr(tile, "gpu_batch_wait_ms", gpu_batch_wait_ms)
            ),
            "enable_shared_memory_handoff": bool(
                getattr(
                    tile,
                    "enable_shared_memory_handoff",
                    enable_shared_memory_handoff,
                )
            ),
            "shared_memory_budget_gb": int(
                getattr(tile, "shared_memory_budget_gb", shared_memory_budget_gb)
            ),
            "max_convert_slots": int(getattr(tile, "max_convert_slots", max_convert_slots)),
            "max_download_slots": int(getattr(tile, "max_download_slots", max_download_slots)),
            "enable_parallel_overlay": bool(
                getattr(tile, "enable_parallel_overlay", enable_parallel_overlay)
            ),
            "max_parallel_tiles": int(
                getattr(tile, "max_parallel_tiles", max_parallel_tiles)
            ),
            "dds_converter": getattr(
                tile, "dds_converter", getattr(UI, "dds_converter", "nvcompress")
            ),
            "dds_format": getattr(tile, "dds_format", getattr(UI, "dds_format", "BC3")),
            "use_gpu_acceleration": bool(
                getattr(tile, "use_gpu_acceleration", getattr(UI, "use_gpu_acceleration", True))
            ),
            "use_gpu_for_masks": bool(getattr(tile, "use_gpu_for_masks", False)),
            "use_gpu_for_color_filters": bool(
                getattr(
                    tile,
                    "use_gpu_for_color_filters",
                    getattr(UI, "use_gpu_for_color_filters", False),
                )
            ),
            "use_gpu_for_dem_smoothing": bool(
                getattr(tile, "use_gpu_for_dem_smoothing", False)
            ),
            "write_build_log": bool(
                getattr(tile, "write_build_log", getattr(UI, "write_build_log", False))
            ),
            "build_overlays_in_all_in_one": bool(
                getattr(tile, "build_overlays_in_all_in_one", False)
            ),
            "custom_overlay_src": getattr(
                tile, "custom_overlay_src", getattr(OVL, "custom_overlay_src", "")
            ),
            "ovl_exclude_pol": list(
                getattr(tile, "ovl_exclude_pol", getattr(OVL, "ovl_exclude_pol", []))
            ),
            "ovl_exclude_net": list(
                getattr(tile, "ovl_exclude_net", getattr(OVL, "ovl_exclude_net", []))
            ),
        }
    )
    cancel_event = UI.active_cancel_event or threading.Event()
    UI.begin_operation(cancel_event, metrics.data["config"])
    tile._cancel_event = cancel_event
    try:
        try:
            IMG.validate_imagery_cache_settings()
        except ValueError as error:
            UI.vprint(
                0,
                UI.ui_text(
                    "ERROR: Invalid imagery cache settings: {}".format(error),
                    "エラー: 画像キャッシュ設定が不正です: {}".format(error),
                ),
            )
            metrics.fail(error)
            return 0
        if not _recover_build_transaction(tile):
            UI.exit_message_and_bottom_line(
                UI.ui_text(
                    "ERROR: The previous tile build could not be recovered.",
                    "エラー: 前回のタイルビルドを復元できませんでした。",
                )
            )
            metrics.fail(RuntimeError("transaction recovery failed"))
            return 0
        _start_ashelper_jsonl_server(tile, metrics=metrics)
        UI.is_building_all = True
        UI.initialize_build_log(tile.build_dir, tile)
        result = _build_all(tile, include_overlays=include_overlays)
        if not result:
            failure = getattr(tile, "last_pipeline_failure", None) or {}
            metrics.fail(
                RuntimeError(
                    "{}: {}".format(
                        failure.get("stage", "tile pipeline"),
                        failure.get("error") or "stage returned failure",
                    )
                )
            )
        return result
    finally:
        UI.is_building_all = False
        UI.is_working = 0
        try:
            UI.flush_build_log(tile.build_dir)
        except Exception as error:
            metrics.fail(error)
            UI.vprint(1, "WARNING: Could not flush tile build log:", error)
        try:
            metrics.set_value("cancel_reason", getattr(UI, "cancel_reason", None))
            metrics.write(
                os.path.join(tile.build_dir, "Ortho4XP_performance.json"),
                finished=True,
            )
        except Exception as error:
            UI.vprint(1, "WARNING: Could not write performance metrics:", error)
        server = getattr(tile, "_ashelper_jsonl_server", None)
        if server is not None:
            server.close()
            try:
                delattr(tile, "_ashelper_jsonl_server")
            except AttributeError:
                pass
        try:
            delattr(tile, "_cancel_event")
        except AttributeError:
            pass
        UI.end_operation()


def build_all(tile):
    return _start_full_pipeline(tile, include_overlays=True)


def build_continuous(tile):
    """Build all core stages with DSF-budget retries for the CLI path."""
    return _start_full_pipeline(
        tile,
        include_overlays=bool(
            getattr(tile, "build_overlays_in_all_in_one", False)
        ),
    )


def _run_pipeline_once(tile, start_stage="vector data"):
    stages = (
        ("vector data", VMAP.build_poly_file),
        ("mesh", MESH.build_mesh),
        ("water masks", MASK.build_masks),
        (
            "imagery/DSF",
            lambda current_tile: build_tile(current_tile, persist_config=False),
        ),
    )
    stage_names = [stage_name for stage_name, _ in stages]
    if start_stage not in stage_names:
        raise ValueError("unknown pipeline start stage: {}".format(start_stage))
    stages = stages[stage_names.index(start_stage):]
    tile.last_pipeline_failure = None
    pipeline_result = OSM.OSM_COMPLETE
    for stage_name, stage in stages:
        stage_error = None
        stage_traceback = None
        metrics = getattr(tile, "_performance_metrics", None)
        stage_context = metrics.stage(stage_name) if metrics is not None else nullcontext()
        try:
            with stage_context:
                stage_result = stage(tile)
                stage_succeeded = bool(stage_result)
        except Exception as error:
            stage_result = OSM.OSM_FAILED
            stage_succeeded = False
            stage_error = error
            stage_traceback = traceback.format_exc()
        if metrics is not None:
            if UI.is_cancel_requested():
                result_name = "cancelled"
            elif stage_result == OSM.OSM_DEGRADED:
                result_name = "degraded"
            elif stage_succeeded:
                result_name = "success"
            else:
                result_name = "failed"
            metrics.set_value("stage_result.{}".format(stage_name), result_name)
        if stage_result == OSM.OSM_DEGRADED:
            pipeline_result = OSM.OSM_DEGRADED
            UI.vprint(
                0,
                UI.ui_text(
                    "WARNING: {} completed as degraded; final output will not be published.".format(
                        stage_name
                    ),
                    "警告: {} はdegraded状態で完了しました。最終出力は公開されません。".format(
                        stage_name
                    ),
                ),
            )
            continue
        if not stage_succeeded or UI.is_cancel_requested():
            _report_pipeline_failure(
                tile, stage_name, stage_error, stage_traceback
            )
            return OSM.OSM_FAILED
    return pipeline_result


def _snapshot_auto_reduce_settings(tile):
    return {
        "max_levelled_segs": int(tile.max_levelled_segs),
        "water_simplification": float(tile.water_simplification),
        "cover_zl": int(tile.cover_zl),
        "curvature_tol": float(tile.curvature_tol),
        "limit_tris": float(tile.limit_tris),
    }


def _snapshot_tile_config_settings(tile):
    """Capture config values needed to finish recovery after a restart."""
    try:
        from O4_Config_Utils import list_tile_vars
    except (ImportError, AttributeError):
        list_tile_vars = _AUTO_REDUCE_SETTING_NAMES
    return {
        name: getattr(tile, name)
        for name in list_tile_vars
        if hasattr(tile, name)
    }


def _apply_auto_reduce_attempt(tile, base_settings, attempt):
    updates = DSF_BUDGET.retry_settings(
        base_settings, attempt, int(tile.mesh_zl)
    )
    for name, value in updates.items():
        setattr(tile, name, value)
    return updates


def _restore_auto_reduce_settings(tile, base_settings):
    for name, value in base_settings.items():
        setattr(tile, name, value)


def _build_all(tile, include_overlays=True):
    """Run the transactional pipeline with degraded vector files visible."""
    missing = object()
    previous_allow_degraded = getattr(
        tile, "_allow_degraded_intermediate", missing
    )
    tile._allow_degraded_intermediate = True
    try:
        return _build_all_transactional(tile, include_overlays=include_overlays)
    finally:
        if previous_allow_degraded is missing:
            try:
                delattr(tile, "_allow_degraded_intermediate")
            except AttributeError:
                pass
        else:
            tile._allow_degraded_intermediate = previous_allow_degraded


def _build_all_transactional(tile, include_overlays=True):
    base_settings = _snapshot_auto_reduce_settings(tile)
    budget = DSF_BUDGET.normalize_budget(
        getattr(tile, "dsf_node_budget", DSF_BUDGET.DEFAULT_DSF_NODE_BUDGET)
    )
    transaction = None
    best_candidate = None
    performance_metrics = getattr(tile, "_performance_metrics", None)
    overlay_enabled = bool(
        include_overlays and getattr(tile, "build_overlays_in_all_in_one", False)
    )
    if performance_metrics is not None:
        performance_metrics.set_value("overlay_enabled", overlay_enabled)
        if not overlay_enabled:
            performance_metrics.set_value("overlay_result", "disabled")

    def restore_snapshot(snapshot_name):
        if transaction is None:
            return True
        try:
            best_settings = (
                best_candidate["settings"] if best_candidate is not None else None
            )
            transaction.best_config = (
                best_candidate.get("config")
                if best_candidate is not None
                else None
            )
            transaction._write_marker(
                "discarding",
                best_candidate["snapshot"] if best_candidate is not None else None,
                best_settings,
                snapshot_name,
            )
            transaction.discard_current("before-restore")
            transaction._write_marker(
                "restoring",
                best_candidate["snapshot"] if best_candidate is not None else None,
                best_settings,
                snapshot_name,
            )
            transaction.restore_snapshot_files(snapshot_name)
            if snapshot_name == "initial":
                transaction.restore_initial_config()
            return True
        except Exception as error:
            UI.logprint(
                "ERROR: Could not finalize tile transaction:",
                repr(error),
                "\n",
                traceback.format_exc(),
            )
            UI.vprint(
                0,
                UI.ui_text(
                    "ERROR: Could not restore the tile output safely: {}".format(
                        error
                    ),
                    "エラー: タイル出力を安全に復元できません: {}".format(error),
                ),
            )
            return False

    def restore_and_cleanup(snapshot_name):
        if not restore_snapshot(snapshot_name):
            return False
        try:
            transaction.cleanup()
            if performance_metrics is not None:
                performance_metrics.set_value("transaction_state", "rolled_back")
            return True
        except Exception as error:
            UI.logprint("ERROR: Could not remove tile transaction staging:", repr(error))
            UI.vprint(0, "ERROR: Could not remove tile transaction staging:", error)
            return False

    try:
        transaction = _BuildTransaction(tile, include_overlay=overlay_enabled)
        if performance_metrics is not None:
            performance_metrics.set_value("transaction_state", "active")
        previous_candidate = None
        for attempt in range(DSF_BUDGET.MAX_AUTO_REDUCE_ATTEMPTS + 1):
            if attempt:
                updates = _apply_auto_reduce_attempt(tile, base_settings, attempt)
                retry_stage = DSF_BUDGET.retry_stage_for_settings(updates)
                UI.vprint(
                    0,
                    "[Auto-Reduce] Partial pipeline attempt {}/{} from {} onward; "
                    "updated settings: {}".format(
                        attempt,
                        DSF_BUDGET.MAX_AUTO_REDUCE_ATTEMPTS,
                        retry_stage,
                        ", ".join(
                            "{}={}".format(name, value)
                            for name, value in updates.items()
                        ),
                    ),
                )
                if previous_candidate is None:
                    raise RuntimeError("partial retry has no previous candidate")
                previous_snapshot = previous_candidate["snapshot"]
                transaction._write_marker(
                    "restoring",
                    best_candidate["snapshot"] if best_candidate is not None else None,
                    best_candidate["settings"] if best_candidate is not None else None,
                    previous_snapshot,
                )
                transaction.discard_current("before-partial-attempt-{}".format(attempt))
                transaction.restore_snapshot_files(previous_snapshot)
                retry_base_snapshot = "retry-base-{}".format(attempt)
                transaction._link_current_to(retry_base_snapshot)
                previous_candidate["snapshot"] = retry_base_snapshot
                transaction._write_marker(
                    "active",
                    best_candidate["snapshot"] if best_candidate is not None else None,
                    best_candidate["settings"] if best_candidate is not None else None,
                )
            else:
                _restore_auto_reduce_settings(tile, base_settings)
                retry_stage = "vector data"
                transaction.discard_current("before-attempt-{}".format(attempt))
                transaction.prepare_attempt()
                UI.vprint(0, "[Auto-Reduce] Full pipeline baseline attempt.")

            if performance_metrics is not None:
                performance_metrics.begin_attempt(
                    attempt,
                    _snapshot_auto_reduce_settings(tile),
                )

            # A verified-cache recovery can leave a diagnostic failure from a
            # previous attempt on the tile.  Only failures recorded during
            # this attempt may affect auto-reduce classification.
            tile.osm_failures = []
            pipeline_result = (
                _run_pipeline_once(tile)
                if retry_stage == "vector data"
                else _run_pipeline_once(tile, start_stage=retry_stage)
            )
            if pipeline_result == OSM.OSM_DEGRADED:
                degraded_snapshot = transaction.capture_candidate(
                    "degraded-{}".format(attempt)
                )
                status = {
                    "status": "DEGRADED",
                    "tile": FNAMES.short_latlon(tile.lat, tile.lon),
                    "missing_layers": sorted(
                        getattr(tile, "osm_degraded_layers", set())
                    ),
                    "failures": getattr(tile, "osm_failures", []),
                    "cache": [
                        failure.get("cache", {"used": False})
                        for failure in getattr(tile, "osm_failures", [])
                    ],
                    "snapshot": degraded_snapshot,
                    "created_at": time.time(),
                }
                staging_path = transaction.export_snapshot(
                    degraded_snapshot, status
                )
                status["staging_path"] = staging_path
                with open(
                    os.path.join(staging_path, "status.json"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(status, stream, ensure_ascii=False, sort_keys=True, indent=2)
                    stream.write("\n")
                transaction._write_marker("restoring", None, None, "initial")
                transaction.restore_snapshot_files("initial")
                transaction.restore_initial_config()
                transaction.cleanup()
                if performance_metrics is not None:
                    performance_metrics.set_value(
                        "overlay_result",
                        "skipped_degraded" if overlay_enabled else "disabled",
                    )
                    performance_metrics.set_value(
                        "transaction_state", "rolled_back"
                    )
                _restore_auto_reduce_settings(tile, base_settings)
                tile.last_pipeline_result = OSM.OSM_DEGRADED
                UI.vprint(
                    0,
                    UI.ui_text(
                        "WARNING: Degraded build was kept for inspection at {}. Existing tile output was restored.",
                        "警告: degradedビルドを検証用に{}へ保存し、既存タイル出力を復元しました。",
                    ).format(staging_path),
                )
                if performance_metrics is not None:
                    performance_metrics.end_attempt("degraded")
                return OSM.OSM_DEGRADED
            if not pipeline_result:
                failure = getattr(tile, "last_pipeline_failure", {})
                osm_failures = getattr(tile, "osm_failures", None) or []
                if best_candidate is None or failure.get("cancelled") or osm_failures:
                    if performance_metrics is not None:
                        performance_metrics.end_attempt(
                            "cancelled" if failure.get("cancelled") else "failed"
                        )
                    _restore_auto_reduce_settings(tile, base_settings)
                    restore_and_cleanup(
                        "initial"
                        if overlay_enabled or best_candidate is None
                        else best_candidate["snapshot"]
                    )
                    return OSM.OSM_FAILED
                UI.vprint(
                    0,
                    UI.ui_text(
                        "WARNING: Reduced pipeline failed; retaining the last successful tile state.",
                        "警告: 削減後の全工程に失敗したため、最後に成功したタイル状態を保持します。",
                    ),
                )
                _restore_auto_reduce_settings(tile, best_candidate["settings"])
                if performance_metrics is not None:
                    performance_metrics.end_attempt("failed_reduced")
                break

            metrics = getattr(tile, "last_dsf_metrics", None)
            metrics_valid = bool(
                metrics
                and hasattr(metrics, "get")
                and metrics.get("structurally_valid", False)
            )
            try:
                point_count = int(metrics["point_count"])
                metrics_valid = metrics_valid and point_count >= 0
            except (KeyError, TypeError, ValueError):
                metrics_valid = False
                point_count = 0
            if not metrics_valid:
                _report_metrics_failure(tile)
                if best_candidate is None:
                    if performance_metrics is not None:
                        performance_metrics.end_attempt("invalid_metrics")
                    _restore_auto_reduce_settings(tile, base_settings)
                    restore_and_cleanup("initial")
                    return 0
                UI.vprint(
                    0,
                    UI.ui_text(
                        "WARNING: Reduced pipeline produced invalid DSF metrics; retaining the last successful tile state.",
                        "警告: 削減後のDSFメトリクスが不正なため、最後に成功したタイル状態を保持します。",
                    ),
                )
                _restore_auto_reduce_settings(tile, best_candidate["settings"])
                if performance_metrics is not None:
                    performance_metrics.end_attempt("invalid_metrics_reduced")
                break

            metrics = dict(metrics)
            metrics["point_count"] = point_count
            candidate = {
                "attempt": attempt,
                "metrics": metrics,
                "settings": _snapshot_auto_reduce_settings(tile),
                "config": _snapshot_tile_config_settings(tile),
                "snapshot": transaction.capture_candidate(attempt),
            }
            previous_candidate = candidate
            if (
                best_candidate is None
                or metrics["point_count"] < best_candidate["metrics"]["point_count"]
            ):
                best_candidate = candidate
                transaction.set_best_snapshot(
                    candidate["snapshot"],
                    candidate["settings"],
                    candidate["config"],
                )
                UI.vprint(
                    1,
                    "[Auto-Reduce] Candidate from attempt {} is the current best.".format(
                        attempt
                    ),
                )

            UI.vprint(
                0,
                "[Auto-Reduce] Attempt {} produced {:,} DSF point instances "
                "(budget {:,}).".format(
                    attempt,
                    metrics["point_count"],
                    budget,
                ),
            )
            budget_exceeded = bool(
                metrics.get("budget_exceeded", metrics["point_count"] > budget)
            )
            if not budget_exceeded:
                if performance_metrics is not None:
                    performance_metrics.end_attempt("success")
                break
            if attempt >= DSF_BUDGET.MAX_AUTO_REDUCE_ATTEMPTS:
                if performance_metrics is not None:
                    performance_metrics.end_attempt("budget_exceeded")
                break
            if performance_metrics is not None:
                performance_metrics.end_attempt("budget_exceeded")
            UI.vprint(0, "[Auto-Reduce] Rebuilding all stages to reduce DSF density.")

        if best_candidate is None:
            _restore_auto_reduce_settings(tile, base_settings)
            restore_and_cleanup("initial")
            return 0

        _restore_auto_reduce_settings(tile, best_candidate["settings"])
        if not restore_snapshot(best_candidate["snapshot"]):
            _restore_auto_reduce_settings(tile, base_settings)
            return 0
        transaction._write_marker(
            "config-pending",
            best_candidate["snapshot"],
            best_candidate["settings"],
        )
        tile.last_dsf_metrics = dict(best_candidate["metrics"])

        if UI.is_cancel_requested():
            _report_pipeline_failure(tile, "tile publication")
            _restore_auto_reduce_settings(tile, base_settings)
            restore_and_cleanup("initial")
            return 0

        if best_candidate["metrics"].get(
            "budget_exceeded", best_candidate["metrics"]["point_count"] > budget
        ):
            UI.vprint(
                0,
                UI.ui_text(
                    "WARNING: DSF point budget remains exceeded after the reduced "
                    "attempt; keeping the structurally valid best build.",
                    "警告: 削減後もDSFポイント予算を超過しています。構造的に有効な最良ビルドを保持します。",
                ),
            )

        if not overlay_enabled:
            if UI.is_cancel_requested():
                _report_pipeline_failure(tile, "tile configuration publication")
                _restore_auto_reduce_settings(tile, base_settings)
                restore_and_cleanup("initial")
                return 0
            if not tile.write_to_config():
                UI.vprint(
                    0,
                    UI.ui_text(
                        "ERROR: Could not save final tile configuration; restoring the previous tile state.",
                        "エラー: 最終タイル設定を保存できないため、以前のタイル状態に戻します。",
                    ),
                )
                _restore_auto_reduce_settings(tile, base_settings)
                restore_and_cleanup("initial")
                return 0
            if UI.is_cancel_requested():
                _report_pipeline_failure(tile, "tile configuration publication")
                _restore_auto_reduce_settings(tile, base_settings)
                restore_and_cleanup("initial")
                return 0
            UI.vprint(
                1,
                "[Auto-Reduce] Saved settings from selected full-pipeline attempt {} to tile config.".format(
                    best_candidate["attempt"]
                ),
            )
        if best_candidate["attempt"] > 0 and not tile.grouped:
            try:
                removed_textures = remove_unwanted_textures(tile)
                if removed_textures:
                    UI.vprint(
                        1,
                        UI.ui_text(
                            "[Auto-Reduce] Removed {} unreferenced generated texture(s).".format(
                                len(removed_textures)
                            ),
                            "[自動削減] 未参照の生成テクスチャを{}個整理しました。".format(
                                len(removed_textures)
                            ),
                        ),
                    )
            except Exception as error:
                UI.logprint(
                    "WARNING: Could not remove unreferenced reduced-attempt textures:",
                    repr(error),
                )
                UI.vprint(
                    1,
                    UI.ui_text(
                        "WARNING: Could not remove unreferenced reduced-attempt textures; keeping the valid tile.",
                        "警告: 自動削減後の未参照テクスチャを整理できないため、有効なタイルを保持します。",
                    ),
                )
    except Exception as error:
        if performance_metrics is not None:
            performance_metrics.end_attempt("exception")
        UI.logprint(
            "ERROR: Full tile pipeline failed unexpectedly:",
            repr(error),
            "\n",
            traceback.format_exc(),
        )
        UI.vprint(
            0,
            UI.ui_text(
                "ERROR: Full tile pipeline failed unexpectedly: {}".format(error),
                "エラー: タイル全工程で予期しない失敗が発生しました: {}".format(
                    error
                ),
            ),
        )
        recovered_settings = base_settings
        if transaction is not None:
            try:
                if overlay_enabled:
                    _restore_auto_reduce_settings(tile, base_settings)
                    restore_and_cleanup("initial")
                elif best_candidate is not None:
                    _restore_auto_reduce_settings(tile, best_candidate["settings"])
                    if restore_snapshot(best_candidate["snapshot"]):
                        transaction._write_marker(
                            "config-pending",
                            best_candidate["snapshot"],
                            best_candidate["settings"],
                        )
                        if tile.write_to_config():
                            recovered_settings = best_candidate["settings"]
                        else:
                            UI.vprint(
                                0,
                                UI.ui_text(
                                    "ERROR: Could not save the recovered tile configuration; restoring the pre-build state.",
                                    "エラー: 復元したタイル設定を保存できないため、ビルド前の状態に戻します。",
                                ),
                            )
                            restore_and_cleanup("initial")
                    else:
                        restore_and_cleanup("initial")
                else:
                    restore_and_cleanup("initial")
            except Exception as recovery_error:
                UI.logprint(
                    "ERROR: Automatic recovery after unexpected pipeline failure failed:",
                    repr(recovery_error),
                )
                try:
                    restore_and_cleanup("initial")
                except Exception as rollback_error:
                    UI.logprint(
                        "ERROR: Rollback to the pre-build tile state also failed:",
                        repr(rollback_error),
                    )
        _restore_auto_reduce_settings(tile, recovered_settings)
        UI.exit_message_and_bottom_line(
            UI.ui_text(
                "ERROR: Tile build failed.",
                "エラー: タイルビルドに失敗しました。",
            )
        )
        return 0

    UI.is_working = 0
    if overlay_enabled:
        UI.vprint(0, "-> Automatically extracting overlays (All in one)...")
        transaction._write_marker(
            "overlay-pending",
            best_candidate["snapshot"],
            best_candidate["settings"],
        )
        if performance_metrics is not None:
            performance_metrics.set_value("transaction_state", "overlay-pending")
        overlay_candidate = transaction.overlay_candidate_path()
        overlay_context = (
            performance_metrics.stage("overlay")
            if performance_metrics is not None
            else nullcontext()
        )
        overlay_error = None
        overlay_traceback = None
        try:
            with overlay_context:
                overlay_result = _build_overlay_stage(
                    tile, output_path=overlay_candidate
                )
        except Exception as error:
            overlay_result = 0
            overlay_error = error
            overlay_traceback = traceback.format_exc()
        if not overlay_result or UI.is_cancel_requested():
            _report_pipeline_failure(
                tile, "overlay extraction", overlay_error, overlay_traceback
            )
            if performance_metrics is not None:
                performance_metrics.set_value(
                    "overlay_result",
                    "cancelled" if UI.is_cancel_requested() else "failed",
                )
            _restore_auto_reduce_settings(tile, base_settings)
            restore_and_cleanup("initial")
            return 0
        transaction._write_marker(
            "committing",
            best_candidate["snapshot"],
            best_candidate["settings"],
        )
        if performance_metrics is not None:
            performance_metrics.set_value("transaction_state", "committing")
        try:
            if not transaction.activate_overlay(overlay_candidate):
                raise OSError("overlay candidate was not produced")
            if not tile.write_to_config():
                raise OSError("could not save final tile configuration")
        except Exception as error:
            UI.vprint(
                0,
                UI.ui_text(
                    "ERROR: Could not publish the overlay and tile configuration; restoring the previous tile state: {}".format(
                        error
                    ),
                    "エラー: Overlayとタイル設定を公開できないため、以前のタイル状態に戻します: {}".format(
                        error
                    ),
                ),
            )
            _restore_auto_reduce_settings(tile, base_settings)
            restore_and_cleanup("initial")
            return 0
        UI.vprint(
            1,
            "[Auto-Reduce] Saved settings from selected full-pipeline attempt {} to tile config.".format(
                best_candidate["attempt"]
            ),
        )
        transaction._write_marker(
            "complete",
            best_candidate["snapshot"],
            best_candidate["settings"],
        )
        if performance_metrics is not None:
            performance_metrics.set_value("overlay_result", "success")
            performance_metrics.set_value("transaction_state", "complete")
    else:
        transaction._write_marker(
            "complete",
            best_candidate["snapshot"],
            best_candidate["settings"],
        )
        if performance_metrics is not None:
            performance_metrics.set_value("transaction_state", "complete")
    try:
        transaction.cleanup()
    except Exception as error:
        UI.logprint("WARNING: Could not remove tile transaction staging:", repr(error))
        UI.vprint(1, "WARNING: Tile transaction staging remains for recovery:", error)
    return 1

################################################################################
def _run_batch_stage(tile, stage_name, stage):
    stage_error = None
    try:
        stage_result = stage(tile)
        succeeded = stage_result == OSM.OSM_COMPLETE
    except Exception as error:
        stage_result = OSM.OSM_FAILED
        succeeded = False
        stage_error = error
        UI.logprint(
            "ERROR: Batch stage",
            stage_name,
            "raised for",
            FNAMES.short_latlon(tile.lat, tile.lon),
            ":",
            repr(error),
            "\n",
            traceback.format_exc(),
        )
    if not UI.is_cancel_requested() and stage_result == OSM.OSM_DEGRADED:
        return OSM.OSM_DEGRADED
    if succeeded and not UI.is_cancel_requested():
        return OSM.OSM_COMPLETE
    if UI.is_cancel_requested():
        UI.exit_message_and_bottom_line(
            UI.ui_text(
                "ERROR: Batch build cancelled during {}.".format(stage_name),
                "エラー: {} の実行中にバッチビルドをキャンセルしました。".format(
                    stage_name
                ),
            )
        )
        return None

    detail = ": {}".format(stage_error) if stage_error is not None else "."
    message = UI.ui_text(
        "ERROR: {} stage failed for tile {}; continuing with the next tile{}".format(
            stage_name, FNAMES.short_latlon(tile.lat, tile.lon), detail
        ),
        "エラー: タイル {} の{}ステージに失敗しました。次のタイルへ進みます{}".format(
            FNAMES.short_latlon(tile.lat, tile.lon), stage_name, detail
        ),
    )
    UI.lvprint(0, message)
    UI.is_working = 0
    return False


def _overlay_worker_entry(
    lat,
    lon,
    custom_overlay_src,
    ovl_exclude_pol,
    ovl_exclude_net,
    result_queue,
    output_path=None,
):
    """Run overlay extraction with process-local UI/module state."""
    try:
        import O4_Overlay_Utils as overlay_utils
        import O4_UI_Utils as ui_utils
        import signal

        ui_utils.is_working = 0
        config = {
            "custom_overlay_src": custom_overlay_src,
            "ovl_exclude_pol": list(ovl_exclude_pol),
            "ovl_exclude_net": list(ovl_exclude_net),
        }
        cancel_event = threading.Event()
        ui_utils.begin_operation(cancel_event, config)

        def request_worker_cancel(signum, frame):
            ui_utils.cancel_operation("parent")

        previous_sigterm = signal.signal(signal.SIGTERM, request_worker_cancel)
        try:
            if output_path:
                result = overlay_utils.build_overlay_to_path(
                    lat,
                    lon,
                    output_path,
                    config=config,
                    cancel_check=ui_utils.is_cancel_requested,
                )
            else:
                result = overlay_utils.build_overlay(lat, lon)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            ui_utils.end_operation()
        result_queue.put(int(bool(result)))
    except Exception:
        result_queue.put(0)


def _build_overlay_stage(tile, output_path=None):
    """Use the historical serial overlay path unless explicitly opted in."""
    config = {
        "custom_overlay_src": getattr(
            tile, "custom_overlay_src", getattr(OVL, "custom_overlay_src", "")
        ),
        "ovl_exclude_pol": list(
            getattr(tile, "ovl_exclude_pol", getattr(OVL, "ovl_exclude_pol", []))
        ),
        "ovl_exclude_net": list(
            getattr(tile, "ovl_exclude_net", getattr(OVL, "ovl_exclude_net", []))
        ),
    }
    cancel_check = lambda: UI.is_cancel_requested() or bool(
        getattr(getattr(tile, "_cancel_event", None), "is_set", lambda: False)()
    )
    if not getattr(tile, "enable_parallel_overlay", enable_parallel_overlay):
        if output_path:
            return OVL.build_overlay_to_path(
                tile.lat, tile.lon, output_path, config=config, cancel_check=cancel_check
            )
        return OVL.build_overlay(tile.lat, tile.lon)
    # A parallel tile worker already has isolated process state. Nesting a
    # second child here would only add startup cost and lose useful logging.
    if getattr(tile, "_parallel_tile_worker", False):
        if output_path:
            return OVL.build_overlay_to_path(
                tile.lat, tile.lon, output_path, config=config, cancel_check=cancel_check
            )
        return OVL.build_overlay(tile.lat, tile.lon)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_overlay_worker_entry,
        args=(
            tile.lat,
            tile.lon,
            config["custom_overlay_src"],
            config["ovl_exclude_pol"],
            config["ovl_exclude_net"],
            result_queue,
            output_path,
        ),
        name="Ortho4XP-overlay-{}".format(FNAMES.short_latlon(tile.lat, tile.lon)),
    )
    process.start()
    try:
        while process.is_alive():
            if UI.is_cancel_requested():
                process.terminate()
                process.join(timeout=2)
                return 0
            process.join(timeout=0.1)
        try:
            return int(result_queue.get(timeout=1))
        except Exception:
            return 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        result_queue.close()


@contextmanager
def _host_gpu_semaphore(enabled):
    """Serialize GPU-owning work across opt-in tile worker processes."""
    if not enabled:
        yield
        return
    file_descriptor = None
    try:
        import fcntl

        semaphore_path = os.path.join(tempfile.gettempdir(), ".ortho4xp-gpu-semaphore")
        file_descriptor = os.open(semaphore_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        yield
    except (ImportError, OSError):
        # The normal single-tile path remains usable on platforms without
        # advisory file locks; the child processes still have isolated state.
        yield
    finally:
        if file_descriptor is not None:
            try:
                import fcntl

                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            try:
                os.close(file_descriptor)
            except OSError:
                pass


def _parallel_tile_stage_uses_gpu(tile, stage_name):
    """Return whether one parallel-tile stage can own the host GPU."""
    if stage_name == "vector data":
        # Airport DEM smoothing is invoked while vector data is assembled.
        return bool(getattr(tile, "use_gpu_for_dem_smoothing", False))
    if stage_name == "water masks":
        return bool(
            getattr(tile, "use_gpu_for_masks", False)
            or getattr(tile, "use_gpu_for_dem_smoothing", False)
        )
    if stage_name != "imagery/DSF":
        return False
    use_gpu_acceleration = bool(getattr(tile, "use_gpu_acceleration", False))
    use_gpu_for_color_filters = bool(
        getattr(tile, "use_gpu_for_color_filters", False)
    )
    if not use_gpu_acceleration and not use_gpu_for_color_filters:
        return False
    dds_converter = getattr(tile, "dds_converter", "nvcompress")
    upscale_backend = IMG.normalize_upscale_backend(
        getattr(tile, "upscale_backend", "none")
    )
    return bool(
        use_gpu_for_color_filters
        or (
            use_gpu_acceleration
            and (
                dds_converter == "TextureConverter"
                or upscale_backend in ("metalfx_spatial", "tensorops")
            )
        )
    )


def _parallel_tile_worker(payload):
    import O4_UI_Utils as worker_ui

    tile_values = payload.get("tile_values", {})
    effective_config = {
        "write_build_log": bool(tile_values.get("write_build_log", False)),
        "use_gpu_for_color_filters": bool(
            tile_values.get("use_gpu_for_color_filters", False)
        ),
        "custom_overlay_src": tile_values.get(
            "custom_overlay_src", getattr(OVL, "custom_overlay_src", "")
        ),
        "ovl_exclude_pol": list(
            tile_values.get("ovl_exclude_pol", getattr(OVL, "ovl_exclude_pol", []))
        ),
        "ovl_exclude_net": list(
            tile_values.get("ovl_exclude_net", getattr(OVL, "ovl_exclude_net", []))
        ),
    }
    worker_ui.begin_operation(threading.Event(), effective_config)
    try:
        return _parallel_tile_worker_impl(payload)
    finally:
        worker_ui.end_operation()


def _parallel_tile_worker_impl(payload):
    """Build one tile in a spawn-isolated process for opt-in batch mode."""
    try:
        import O4_Config_Utils as CFG
        import O4_UI_Utils as worker_ui
        import O4_OSM_Utils as worker_osm

        lat = int(payload["lat"])
        lon = int(payload["lon"])
        tile = CFG.Tile(lat, lon, payload.get("custom_build_dir", ""))
        for name, value in payload.get("tile_values", {}).items():
            if hasattr(tile, name):
                setattr(tile, name, value)
        if payload.get("do_ptc") and not tile.read_from_config():
            return lat, lon, False, "tile config could not be read"
        tile.build_dir = FNAMES.build_dir(lat, lon, tile.custom_build_dir)
        tile.make_dirs()
        tile._parallel_tile_worker = True
        worker_ui.update_operation_config(
            {
                "write_build_log": bool(
                    getattr(tile, "write_build_log", getattr(worker_ui, "write_build_log", False))
                ),
                "use_gpu_for_color_filters": bool(
                    getattr(tile, "use_gpu_for_color_filters", False)
                ),
                "custom_overlay_src": getattr(
                    tile, "custom_overlay_src", getattr(OVL, "custom_overlay_src", "")
                ),
                "ovl_exclude_pol": list(
                    getattr(tile, "ovl_exclude_pol", getattr(OVL, "ovl_exclude_pol", []))
                ),
                "ovl_exclude_net": list(
                    getattr(tile, "ovl_exclude_net", getattr(OVL, "ovl_exclude_net", []))
                ),
            }
        )
        worker_osm.osm_download_failure_policy = worker_osm.normalize_osm_failure_policy(
            payload.get("osm_download_failure_policy", worker_osm.osm_download_failure_policy)
        )
        worker_ui.is_working = 0
        worker_ui.red_flag = False

        transaction = None
        if any(
            payload.get(name)
            for name in ("do_osm", "do_mesh", "do_mask", "do_dsf", "do_ovl")
        ):
            transaction = _BuildTransaction(
                tile,
                preserve_inputs=True,
                include_overlay=bool(payload.get("do_ovl")),
            )
            tile._allow_degraded_intermediate = True

        stages = []
        if payload.get("do_osm"):
            stages.append(("vector data", VMAP.build_poly_file))
        if payload.get("do_mesh"):
            stages.append(("mesh", MESH.build_mesh))
        if payload.get("do_mask"):
            stages.append(("water masks", MASK.build_masks))
        if payload.get("do_dsf"):
            stages.append(("imagery/DSF", lambda current_tile: build_tile(current_tile)))
        if payload.get("do_ovl"):
            stages.append(
                (
                    "overlay extraction",
                    lambda current_tile: _build_overlay_stage(
                        current_tile,
                        output_path=transaction.overlay_candidate_path(),
                    ),
                )
            )

        tile_degraded = False
        for stage_name, stage in stages:
            if stage_name == "overlay extraction" and tile_degraded:
                UI.vprint(
                    0,
                    UI.ui_text(
                        "WARNING: Skipping overlay publication for degraded tile.",
                        "警告: degradedタイルのオーバーレイ公開をスキップします。",
                    ),
                )
                continue
            stage_context = _host_gpu_semaphore(
                _parallel_tile_stage_uses_gpu(tile, stage_name)
            )
            with stage_context:
                try:
                    stage_result = stage(tile)
                except Exception as error:
                    if transaction is not None:
                        transaction.restore_snapshot("initial")
                        transaction.cleanup()
                    return lat, lon, OSM.OSM_FAILED, "{}: {}".format(stage_name, error)
            if worker_ui.is_cancel_requested():
                if transaction is not None:
                    transaction.restore_snapshot("initial")
                    transaction.cleanup()
                return lat, lon, OSM.OSM_FAILED, "{} cancelled".format(stage_name)
            if stage_result == OSM.OSM_DEGRADED:
                tile_degraded = True
                continue
            if stage_result != OSM.OSM_COMPLETE:
                if transaction is not None:
                    transaction.restore_snapshot("initial")
                    transaction.cleanup()
                return lat, lon, OSM.OSM_FAILED, "{} failed".format(stage_name)

        if transaction is not None:
            if tile_degraded:
                degraded_snapshot = transaction.capture_candidate("degraded")
                status = {
                    "status": "DEGRADED",
                    "tile": FNAMES.short_latlon(tile.lat, tile.lon),
                    "missing_layers": sorted(
                        getattr(tile, "osm_degraded_layers", set())
                    ),
                    "failures": getattr(tile, "osm_failures", []),
                    "cache": [
                        failure.get("cache", {"used": False})
                        for failure in getattr(tile, "osm_failures", [])
                    ],
                    "snapshot": degraded_snapshot,
                    "created_at": time.time(),
                }
                staging_path = transaction.export_snapshot(
                    degraded_snapshot, status
                )
                status["staging_path"] = staging_path
                with open(
                    os.path.join(staging_path, "status.json"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(status, stream, ensure_ascii=False, sort_keys=True, indent=2)
                    stream.write("\n")
                transaction.restore_snapshot("initial")
                transaction.cleanup()
                tile._allow_degraded_intermediate = False
                return lat, lon, OSM.OSM_DEGRADED, staging_path
            if payload.get("do_ovl"):
                if not transaction.activate_overlay(transaction.overlay_candidate_path()):
                    transaction.restore_snapshot("initial")
                    transaction.cleanup()
                    return lat, lon, OSM.OSM_FAILED, "overlay publication failed"
            transaction.cleanup()
            tile._allow_degraded_intermediate = False
        return lat, lon, OSM.OSM_COMPLETE, None
    except Exception as error:
        return int(payload.get("lat", 0)), int(payload.get("lon", 0)), False, str(error)


def _build_tile_list_parallel(
    tile,
    list_lat_lon,
    do_osm,
    do_mesh,
    do_mask,
    do_dsf,
    do_ovl,
    do_ptc,
    worker_count,
):
    """Run explicit multi-tile batches with isolated UI/module state.

    The tile workers are direct children of the caller rather than Pool
    workers.  A tile may itself create the historical DDS conversion pool (or
    the streaming CPU pool), and Python's daemon Pool workers are prohibited
    from creating children on macOS spawn.
    """
    try:
        import O4_Config_Utils as CFG

        tile_values = {
            name: getattr(tile, name)
            for name in CFG.list_tile_vars
            if hasattr(tile, name)
        }
    except Exception:
        tile_values = {}
    payloads = [
        {
            "lat": lat,
            "lon": lon,
            "custom_build_dir": tile.custom_build_dir,
            "tile_values": tile_values,
            "do_osm": do_osm,
            "do_mesh": do_mesh,
            "do_mask": do_mask,
            "do_dsf": do_dsf,
            "do_ovl": do_ovl,
            "do_ptc": do_ptc,
            "osm_download_failure_policy": getattr(
                OSM, "osm_download_failure_policy", "abort"
            ),
        }
        for lat, lon in list_lat_lon
    ]
    context = multiprocessing.get_context("spawn")
    failed = False
    degraded = False
    completed = 0
    for batch_start in range(0, len(payloads), worker_count):
        batch = payloads[batch_start : batch_start + worker_count]
        result_queue = context.Queue()
        processes = []
        raw_results = []
        try:
            for payload in batch:
                process = context.Process(
                    target=_parallel_tile_worker,
                    args=(payload,),
                    name="Ortho4XP-tile-{}".format(
                        FNAMES.short_latlon(payload["lat"], payload["lon"])
                    ),
                )
                process.start()
                processes.append(process)

            active = list(processes)
            while active:
                if UI.is_cancel_requested():
                    for process in active:
                        if process.is_alive():
                            process.terminate()
                    for process in active:
                        process.join(timeout=2)
                    UI.exit_message_and_bottom_line(
                        UI.ui_text(
                            "ERROR: Parallel batch build cancelled.",
                            "エラー: 並列バッチビルドをキャンセルしました。",
                        )
                    )
                    return 0
                try:
                    raw_results.append(result_queue.get(timeout=0.2))
                except queue.Empty:
                    pass
                for process in list(active):
                    if not process.is_alive():
                        process.join()
                        active.remove(process)

            while True:
                try:
                    raw_results.append(result_queue.get_nowait())
                except queue.Empty:
                    break

            for payload in batch:
                key = (payload["lat"], payload["lon"])
                matching = next(
                    (result for result in raw_results if (result[0], result[1]) == key),
                    None,
                )
                if matching is None:
                    result = (
                        payload["lat"],
                        payload["lon"],
                        False,
                        "worker exited without a result",
                    )
                else:
                    result = matching
                lat, lon, stage_result, detail = result
                completed += 1
                UI.vprint(
                    1,
                    "Parallel tile {}/{}: {}".format(
                        completed, len(payloads), FNAMES.short_latlon(lat, lon)
                    ),
                )
                if stage_result == OSM.OSM_DEGRADED:
                    degraded = True
                    UI.lvprint(
                        0,
                        UI.ui_text(
                            "WARNING: Parallel tile {} completed as degraded: {}".format(
                                FNAMES.short_latlon(lat, lon), detail or "staging"
                            ),
                            "警告: 並列タイル{}はdegraded状態で完了しました: {}".format(
                                FNAMES.short_latlon(lat, lon), detail or "staging"
                            ),
                        ),
                    )
                elif stage_result != OSM.OSM_COMPLETE:
                    failed = True
                    UI.lvprint(
                        0,
                        UI.ui_text(
                            "ERROR: Parallel tile {} failed: {}".format(
                                FNAMES.short_latlon(lat, lon), detail or "unknown error"
                            ),
                            "エラー: 並列タイル {} に失敗しました: {}".format(
                                FNAMES.short_latlon(lat, lon), detail or "不明なエラー"
                            ),
                        ),
                    )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
            result_queue.close()
            result_queue.join_thread()
    if failed:
        return OSM.OSM_FAILED
    if degraded:
        return OSM.OSM_DEGRADED
    return OSM.OSM_COMPLETE


################################################################################
def build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
):
    """Build a batch with one operation-local cancellation token."""
    if UI.active_cancel_event is not None:
        return _build_tile_list_impl(
            tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
        )
    effective_config = {
        "write_build_log": bool(
            getattr(tile, "write_build_log", getattr(UI, "write_build_log", False))
        ),
        "use_gpu_for_color_filters": bool(
            getattr(tile, "use_gpu_for_color_filters", False)
        ),
        "custom_overlay_src": getattr(
            tile, "custom_overlay_src", getattr(OVL, "custom_overlay_src", "")
        ),
        "ovl_exclude_pol": list(
            getattr(tile, "ovl_exclude_pol", getattr(OVL, "ovl_exclude_pol", []))
        ),
        "ovl_exclude_net": list(
            getattr(tile, "ovl_exclude_net", getattr(OVL, "ovl_exclude_net", []))
        ),
    }
    UI.begin_operation(threading.Event(), effective_config)
    try:
        return _build_tile_list_impl(
            tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
        )
    finally:
        UI.end_operation()


def _build_tile_list_impl(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
):
    if UI.is_working:
        return 0
    if not UI.is_cancel_requested() and UI.active_cancel_event is None:
        UI.red_flag = 0
    requested_parallel_tiles = int(
        getattr(tile, "max_parallel_tiles", max_parallel_tiles) or 1
    )
    if requested_parallel_tiles > 1 and len(list_lat_lon) > 1:
        worker_count = min(2, requested_parallel_tiles, len(list_lat_lon))
        UI.vprint(
            0,
            "-> Starting isolated parallel tile workers:",
            worker_count,
        )
        return _build_tile_list_parallel(
            tile,
            list_lat_lon,
            do_osm,
            do_mesh,
            do_mask,
            do_dsf,
            do_ovl,
            do_ptc,
            worker_count,
        )
    timer = time.time()
    UI.lvprint(
        0, "Batch build launched for a number of", len(list_lat_lon), "tiles."
    )
    batch_failed = False
    batch_degraded = False
    k = 0
    for (lat, lon) in list_lat_lon:
        k += 1
        if UI.is_cancel_requested():
            UI.exit_message_and_bottom_line(
                UI.ui_text(
                    "ERROR: Batch build cancelled before the next tile.",
                    "エラー: 次のタイルへ進む前にバッチビルドをキャンセルしました。",
                )
            )
            return 0
        UI.vprint(
            1,
            "Dealing with tile ",
            k,
            "/",
            len(list_lat_lon),
            ":",
            FNAMES.short_latlon(lat, lon),
        )
        (tile.lat, tile.lon) = (lat, lon)
        tile.build_dir = FNAMES.build_dir(
            tile.lat, tile.lon, tile.custom_build_dir
        )
        tile.dem = None
        if do_ptc:
            try:
                config_loaded = bool(tile.read_from_config())
            except Exception as error:
                batch_failed = True
                UI.logprint(
                    "ERROR: Could not read per-tile config for",
                    FNAMES.short_latlon(lat, lon),
                    ":",
                    repr(error),
                    "\n",
                    traceback.format_exc(),
                )
                UI.lvprint(
                    0,
                    UI.ui_text(
                        "CFG error: Could not read settings for tile {}; skipping it: {}".format(
                            FNAMES.short_latlon(lat, lon), error
                        ),
                        "CFGエラー: タイル {} の設定を読み込めないためスキップします: {}".format(
                            FNAMES.short_latlon(lat, lon), error
                        ),
                    ),
                )
                continue
            if not config_loaded:
                batch_failed = True
                UI.lvprint(
                    0,
                    UI.ui_text(
                        "CFG error: Skipping tile {} because no per-tile config was found.".format(
                            FNAMES.short_latlon(lat, lon)
                        ),
                        "CFGエラー: タイル {} のタイル別設定がないためスキップします。".format(
                            FNAMES.short_latlon(lat, lon)
                        ),
                    ),
                )
                continue

        UI.update_operation_config(
            {
                "write_build_log": bool(
                    getattr(tile, "write_build_log", getattr(UI, "write_build_log", False))
                ),
                "use_gpu_for_color_filters": bool(
                    getattr(tile, "use_gpu_for_color_filters", False)
                ),
                "custom_overlay_src": getattr(
                    tile, "custom_overlay_src", getattr(OVL, "custom_overlay_src", "")
                ),
                "ovl_exclude_pol": list(
                    getattr(tile, "ovl_exclude_pol", getattr(OVL, "ovl_exclude_pol", []))
                ),
                "ovl_exclude_net": list(
                    getattr(tile, "ovl_exclude_net", getattr(OVL, "ovl_exclude_net", []))
                ),
            }
        )

        if do_osm or do_mesh or do_dsf or do_ovl:
            try:
                tile.make_dirs()
            except Exception as error:
                batch_failed = True
                UI.lvprint(
                    0,
                    UI.ui_text(
                        "ERROR: Could not prepare tile {}; continuing with the next tile: {}".format(
                            FNAMES.short_latlon(lat, lon), error
                        ),
                        "エラー: タイル {} の準備に失敗しました。次のタイルへ進みます: {}".format(
                            FNAMES.short_latlon(lat, lon), error
                        ),
                    ),
                )
                UI.is_working = 0
                continue

        batch_transaction = None
        if do_osm or do_mesh or do_mask or do_dsf or do_ovl:
            try:
                batch_transaction = _BuildTransaction(
                    tile,
                    preserve_inputs=True,
                    include_overlay=bool(do_ovl),
                )
                tile._allow_degraded_intermediate = True
            except Exception as error:
                batch_failed = True
                UI.lvprint(
                    0,
                    UI.ui_text(
                        "ERROR: Could not start the safe OSM build transaction for tile {}: {}".format(
                            FNAMES.short_latlon(lat, lon), error
                        ),
                        "エラー: タイル {} のOSM安全トランザクションを開始できません: {}".format(
                            FNAMES.short_latlon(lat, lon), error
                        ),
                    ),
                )
                continue

        stages = []
        if do_osm:
            stages.append(("vector data", VMAP.build_poly_file))
        if do_mesh:
            stages.append(("mesh", MESH.build_mesh))
        if do_mask:
            stages.append(("water masks", MASK.build_masks))
        if do_dsf:
            stages.append(
                (
                    "imagery/DSF",
                    lambda current_tile: build_tile(current_tile),
                )
            )
        if do_ovl:
            stages.append(
                (
                    "overlay extraction",
                    lambda current_tile: _build_overlay_stage(
                        current_tile,
                        output_path=batch_transaction.overlay_candidate_path(),
                    ),
                )
            )

        tile_succeeded = True
        tile_degraded = False
        for stage_name, stage in stages:
            if stage_name == "overlay extraction" and tile_degraded:
                UI.vprint(
                    0,
                    UI.ui_text(
                        "WARNING: Skipping overlay publication for degraded tile.",
                        "警告: degradedタイルのオーバーレイ公開をスキップします。",
                    ),
                )
                continue
            stage_result = _run_batch_stage(tile, stage_name, stage)
            if stage_result is None:
                tile_succeeded = False
                batch_failed = True
                break
            if stage_result == OSM.OSM_DEGRADED:
                tile_degraded = True
                continue
            if stage_result != OSM.OSM_COMPLETE:
                batch_failed = True
                tile_succeeded = False
                break

        if batch_transaction is not None:
            try:
                if tile_succeeded and tile_degraded:
                    degraded_snapshot = batch_transaction.capture_candidate(
                        "degraded"
                    )
                    status = {
                        "status": "DEGRADED",
                        "tile": FNAMES.short_latlon(tile.lat, tile.lon),
                        "missing_layers": sorted(
                            getattr(tile, "osm_degraded_layers", set())
                        ),
                        "failures": getattr(tile, "osm_failures", []),
                        "cache": [
                            failure.get("cache", {"used": False})
                            for failure in getattr(tile, "osm_failures", [])
                        ],
                        "snapshot": degraded_snapshot,
                        "created_at": time.time(),
                    }
                    staging_path = batch_transaction.export_snapshot(
                        degraded_snapshot, status
                    )
                    status["staging_path"] = staging_path
                    with open(
                        os.path.join(staging_path, "status.json"),
                        "w",
                        encoding="utf-8",
                    ) as stream:
                        json.dump(
                            status,
                            stream,
                            ensure_ascii=False,
                            sort_keys=True,
                            indent=2,
                        )
                        stream.write("\n")
                    batch_transaction.restore_snapshot("initial")
                    batch_transaction.cleanup()
                    batch_degraded = True
                    tile_succeeded = False
                    UI.vprint(
                        0,
                        UI.ui_text(
                            "WARNING: Degraded tile {} was kept for inspection at {}.".format(
                                FNAMES.short_latlon(tile.lat, tile.lon), staging_path
                            ),
                            "警告: degradedタイル{}を検証用に{}へ保存しました。".format(
                                FNAMES.short_latlon(tile.lat, tile.lon), staging_path
                            ),
                        ),
                    )
                elif tile_succeeded:
                    if do_ovl:
                        overlay_candidate = batch_transaction.overlay_candidate_path()
                        if not batch_transaction.activate_overlay(overlay_candidate):
                            raise OSError("overlay candidate was not produced")
                    batch_transaction.cleanup()
                else:
                    batch_transaction.restore_snapshot("initial")
                    batch_transaction.cleanup()
            except Exception as error:
                batch_failed = True
                tile_succeeded = False
                UI.logprint(
                    "ERROR: Could not finalize batch OSM transaction:",
                    repr(error),
                    "\n",
                    traceback.format_exc(),
                )
                UI.lvprint(0, "ERROR: Could not finalize safe batch output:", error)
            finally:
                tile._allow_degraded_intermediate = False

        if tile_succeeded:
            try:
                UI.gui.earth_window.canvas.delete(
                    UI.gui.earth_window.dico_tiles_todo[(lat, lon)]
                )
                UI.gui.earth_window.dico_tiles_todo.pop((lat, lon), None)
            except Exception:
                pass
    if batch_failed:
        UI.lvprint(
            0,
            UI.ui_text(
                "Batch process completed in {} with skipped or failed tiles.".format(
                    UI.nicer_timer(time.time() - timer)
                ),
                "{} にバッチ処理が完了しましたが、スキップまたは失敗したタイルがあります。".format(
                    UI.nicer_timer(time.time() - timer)
                ),
            ),
        )
        return 0
    if batch_degraded:
        UI.lvprint(
            0,
            UI.ui_text(
                "Batch process completed with degraded tiles; no degraded tile was published.",
                "バッチ処理はdegradedタイルを含んで完了しました。degradedタイルは公開していません。",
            ),
        )
        return OSM.OSM_DEGRADED
    UI.lvprint(
        0, "Batch process completed in", UI.nicer_timer(time.time() - timer)
    )
    return 1

################################################################################
def remove_unwanted_textures(tile):
    """Remove generated DDS files not referenced by the tile's terrain files.

    Grouped builds share terrain/textures between tiles, so callers must not
    prune those outputs as if they belonged exclusively to one tile.
    """
    if getattr(tile, "grouped", False):
        return []

    terrain_dir = os.path.join(tile.build_dir, "terrain")
    texture_dir = os.path.join(tile.build_dir, "textures")
    referenced_textures = set()
    if os.path.isdir(terrain_dir):
        for dir_path, _, names in os.walk(terrain_dir):
            for name in names:
                if not name.endswith(".ter"):
                    continue
                base_name = name[:-4]
                for suffix in (
                    "_water_overlay",
                    "_sea_overlay",
                    "_water",
                    "_sea",
                    "_overlay",
                ):
                    if base_name.endswith(suffix):
                        base_name = base_name[: -len(suffix)]
                        break
                referenced_textures.add(base_name + ".dds")

    removed_textures = []
    if not os.path.isdir(texture_dir):
        return removed_textures
    for name in os.listdir(texture_dir):
        if not _is_generated_dds_name(name):
            continue
        if name in referenced_textures:
            continue
        path = os.path.join(texture_dir, name)
        try:
            os.remove(path)
            removed_textures.append(path)
        except OSError:
            pass
    return removed_textures
