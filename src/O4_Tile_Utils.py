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
from contextlib import contextmanager, nullcontext
from itertools import count
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_DSF_Utils as DSF
import O4_Overlay_Utils as OVL
import O4_DSF_Budget as DSF_BUDGET
import O4_Performance_Utils as PERF
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
gpu_batch_wait_ms = 50
enable_parallel_overlay = False
max_parallel_tiles = 1


_BUILD_TRANSACTION_MARKER = ".Ortho4XP_build_recovery.json"
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
    """Keep each full-pipeline attempt recoverable without copying tile data.

    The pipeline writes to the canonical tile paths.  This transaction moves
    only known Ortho4XP outputs to a sibling staging directory, so an attempt
    can run from a clean output set and a later failure cannot mix its files
    with a previous successful attempt.  Moves stay on the same filesystem
    and therefore do not duplicate multi-gigabyte DDS assets.
    """

    def __init__(self, tile, preserve_inputs=False):
        self.build_dir = os.path.abspath(tile.build_dir)
        self.mask_dir = os.path.abspath(FNAMES.mask_dir(tile.lat, tile.lon))
        self.grouped = bool(getattr(tile, "grouped", False))
        self.preserve_inputs = bool(preserve_inputs)
        self.parent_dir = os.path.dirname(self.build_dir) or os.curdir
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.parent_dir, exist_ok=True)
        self.root = tempfile.mkdtemp(
            prefix=".o4xp-build-transaction-", dir=self.parent_dir
        )
        self.marker_path = os.path.join(
            self.build_dir, _BUILD_TRANSACTION_MARKER
        )
        self.tile_lat = int(tile.lat)
        self.tile_lon = int(tile.lon)
        try:
            if self.preserve_inputs:
                # A standalone Step 3 still needs the existing mesh, masks,
                # and reusable DDS files visible at their canonical paths.
                # Hardlinks provide a rollback snapshot without duplicating
                # multi-gigabyte texture payloads.
                self._write_marker("snapshotting", None, None)
                self._link_current_to("initial")
                self._write_marker("active", None, None)
            else:
                self._write_marker("active", None, None)
                self._move_current_to("initial")
                self._write_marker("active", None, None)
        except Exception:
            if self.preserve_inputs:
                shutil.rmtree(self.root, ignore_errors=True)
                for path in (self.marker_path, self.marker_path + ".tmp"):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            raise

    def _write_marker(
        self,
        state,
        best_snapshot,
        best_settings=None,
        target_snapshot=None,
    ):
        marker = {
            "version": 1,
            "state": state,
            "transaction_root": self.root,
            "build_dir": self.build_dir,
            "mask_dir": self.mask_dir,
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
        return self._tile_output_paths() + self._mask_output_paths()

    def _link_current_to(self, snapshot_name):
        """Snapshot current files while leaving standalone inputs visible."""
        snapshot_root = os.path.join(self.root, snapshot_name)
        for kind, source_path in self._current_output_paths():
            source_root = self.mask_dir if kind == "mask" else self.build_dir
            relative_path = os.path.relpath(source_path, source_root)
            destination_path = os.path.join(
                snapshot_root, kind, relative_path
            )
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            if source_path.lower().endswith(".dds"):
                # DDS conversion activates output with os.replace(), so a
                # hardlink preserves the old multi-gigabyte payload without a
                # second copy. The build directory and its sibling snapshot
                # are deliberately on the same filesystem.
                os.link(source_path, destination_path)
            else:
                # Terrain, mesh, and mask files can be opened in-place by
                # legacy code; a hardlink would let those writes corrupt the
                # rollback snapshot. These files are small enough to copy,
                # and mask files may live on another filesystem.
                shutil.copy2(source_path, destination_path)

    def _move_current_to(self, snapshot_name):
        snapshot_root = os.path.join(self.root, snapshot_name)
        for kind, source_path in self._current_output_paths():
            source_root = self.mask_dir if kind == "mask" else self.build_dir
            relative_path = os.path.relpath(source_path, source_root)
            destination_path = os.path.join(
                snapshot_root, kind, relative_path
            )
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            os.replace(source_path, destination_path)

    def capture_candidate(self, attempt):
        snapshot_name = "candidate-{}".format(attempt)
        self._move_current_to(snapshot_name)
        return snapshot_name

    def clone_snapshot(self, source_snapshot, target_snapshot):
        """Clone a candidate without consuming it.

        DDS payloads are hardlinked while small text/raster outputs are copied,
        matching the transaction's existing snapshot policy.  This gives a
        partial retry a private rollback point even though restoring a
        candidate moves its files back to canonical paths.
        """
        source_root = os.path.join(self.root, source_snapshot)
        target_root = os.path.join(self.root, target_snapshot)
        if not os.path.isdir(source_root):
            raise FileNotFoundError(source_root)
        for kind in ("tile", "shared", "mask"):
            source_kind_root = os.path.join(source_root, kind)
            if not os.path.isdir(source_kind_root):
                continue
            for dir_path, _, names in os.walk(source_kind_root):
                for name in names:
                    source_path = os.path.join(dir_path, name)
                    relative_path = os.path.relpath(source_path, source_kind_root)
                    destination_path = os.path.join(
                        target_root, kind, relative_path
                    )
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    if source_path.lower().endswith(".dds"):
                        os.link(source_path, destination_path)
                    else:
                        shutil.copy2(source_path, destination_path)
        return target_snapshot

    def discard_current(self, label):
        self._move_current_to("discarded-{}".format(label))

    def prepare_attempt(self):
        """Restore shared grouped assets before the next clean attempt."""
        if not self.grouped:
            return
        source_root = os.path.join(self.root, "initial", "shared")
        if not os.path.isdir(source_root):
            return
        for dir_path, _, names in os.walk(source_root):
            for name in names:
                source_path = os.path.join(dir_path, name)
                relative_path = os.path.relpath(source_path, source_root)
                destination_path = os.path.join(
                    self.build_dir, relative_path
                )
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                os.replace(source_path, destination_path)

    def restore_snapshot(self, snapshot_name):
        self.discard_current("before-restore")
        self.restore_snapshot_files(snapshot_name)

    def restore_snapshot_files(self, snapshot_name):
        """Move the remaining files of a snapshot into canonical paths.

        This operation is intentionally idempotent: a recovery run can finish
        a restore after a process stopped between two file moves.
        """
        snapshot_root = os.path.join(self.root, snapshot_name)
        for kind, destination_root in (
            ("tile", self.build_dir),
            ("shared", self.build_dir),
            ("mask", self.mask_dir),
        ):
            source_root = os.path.join(snapshot_root, kind)
            if not os.path.isdir(source_root):
                continue
            for dir_path, _, names in os.walk(source_root):
                for name in names:
                    source_path = os.path.join(dir_path, name)
                    relative_path = os.path.relpath(source_path, source_root)
                    destination_path = os.path.join(
                        destination_root, relative_path
                    )
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    os.replace(source_path, destination_path)

    def snapshot_has_files(self, snapshot_name):
        snapshot_root = os.path.join(self.root, snapshot_name)
        for _, _, names in os.walk(snapshot_root):
            if names:
                return True
        return False

    def cleanup(self):
        shutil.rmtree(self.root)
        try:
            os.remove(self.marker_path)
        except OSError:
            pass
        try:
            os.remove(self.marker_path + ".tmp")
        except OSError:
            pass


def _transaction_marker_path(tile):
    return os.path.join(
        os.path.abspath(tile.build_dir), _BUILD_TRANSACTION_MARKER
    )


def _recover_build_transaction(tile):
    """Restore a left-over full-pipeline transaction after a hard stop."""
    marker_path = _transaction_marker_path(tile)
    if not os.path.isfile(marker_path):
        return True
    try:
        with open(marker_path, "r", encoding="utf-8") as stream:
            marker = json.load(stream)
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
        if not os.path.isdir(root):
            os.remove(marker_path)
            return True

        transaction = _BuildTransaction.__new__(_BuildTransaction)
        transaction.build_dir = build_dir
        transaction.mask_dir = mask_dir
        transaction.grouped = bool(marker.get("grouped", False))
        transaction.preserve_inputs = bool(marker.get("preserve_inputs", False))
        transaction.parent_dir = os.path.abspath(parent_dir)
        transaction.root = root
        transaction.marker_path = marker_path
        transaction.tile_lat = int(tile.lat)
        transaction.tile_lon = int(tile.lon)
        transaction.best_config = marker.get("best_config")
        state = marker.get("state", "active")
        best_snapshot = marker.get("best_snapshot")
        best_settings = marker.get("best_settings")
        best_config = marker.get("best_config")
        target_snapshot = marker.get("target_snapshot")

        if state == "snapshotting":
            # Standalone snapshot creation only links/copies into staging; the
            # canonical outputs have not been changed yet. Discard a partial
            # snapshot rather than attempting to restore incomplete inputs.
            transaction.cleanup()
            return True

        if state in ("complete", "restored"):
            transaction.cleanup()
            return True

        if state == "discarding":
            target_snapshot = target_snapshot or best_snapshot or "initial"
            transaction.discard_current("before-restore")
            transaction._write_marker(
                "restoring",
                best_snapshot,
                best_settings,
                target_snapshot,
            )
            state = "restoring"

        if state == "restoring":
            target_snapshot = target_snapshot or best_snapshot or "initial"
            transaction.restore_snapshot_files(target_snapshot)
            if target_snapshot == "initial":
                best_settings = None
                best_config = None
            else:
                state = "config-pending"

        elif state == "config-pending":
            if best_snapshot and transaction.snapshot_has_files(best_snapshot):
                transaction.restore_snapshot_files(best_snapshot)
        else:
            snapshot_name = best_snapshot or "initial"
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
            "probe_error": type(error).__name__,
        }
    else:
        values = {}
        for line in (result.stdout or "").splitlines():
            if "=" not in line:
                continue
            name, value = line.strip().split("=", 1)
            values[name] = value.lower() == "true"
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
    return max(1, requested if requested > 0 else 32)


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
    streaming_requested = bool(
        getattr(tile, "enable_streaming_conversion", enable_streaming_conversion)
    )
    dds_converter = getattr(
        tile, "dds_converter", getattr(UI, "dds_converter", "nvcompress")
    )
    wants_server = bool(
        use_gpu
        and (
            (streaming_requested and dds_converter == "TextureConverter")
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
                )
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
        if self.gpu_server is None or self.gpu_server.gpu_disabled:
            raise RuntimeError("ASHelper GPU server is disabled")
        specs = []
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
            specs.append(spec)

        if self.metrics is not None:
            self.metrics.increment("conversion_batches_gpu")
        response = self.gpu_server.convert_batch(
            [spec["request"] for spec in specs],
            gpu=True,
        )
        response_by_id = {
            result.get("id"): result
            for result in response.get("results", [])
            if isinstance(result, dict)
        }
        normalized = {}
        for spec in specs:
            task_id = spec["task_id"]
            result = response_by_id.get(task_id, {})
            ok = bool(result.get("ok"))
            error = result.get("error")
            if ok:
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
                        error = "atomic_publish:{}".format(type(publish_error).__name__)
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

    def finish(self):
        if UI.red_flag:
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
            if UI.red_flag or self.scheduler.error is not None:
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


def _sync_terrain_load_centers(tile):
    """Synchronize generated terrain metadata with the final DDS headers."""
    terrain_dir = os.path.join(tile.build_dir, "terrain")
    if not os.path.isdir(terrain_dir):
        raise FileNotFoundError(terrain_dir)

    staged = []
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
                width, height = IMG.read_dds_dimensions(dds_path)
                tokens = lines[load_center_index].split()
                if len(tokens) != 5:
                    raise ValueError(
                        f"Invalid LOAD_CENTER in generated terrain: {terrain_path}"
                    )
                tokens[-1] = str(width)
                lines[load_center_index] = " ".join(tokens) + "\n"

                temporary_path = terrain_path + ".tmp"
                with open(temporary_path, "w", encoding="utf-8") as stream:
                    stream.writelines(lines)
                    stream.flush()
                    os.fsync(stream.fileno())
                staged.append((temporary_path, terrain_path))

        for temporary_path, terrain_path in staged:
            os.replace(temporary_path, terrain_path)
    except Exception:
        for temporary_path, _ in staged:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        raise

    return len(staged)


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


def _run_tensorops_direct_dds_batch(as_helper, pack_path, specs, chunk_size=8):
    """Run TensorOps in one child per bounded chunk and publish valid DDS atomically."""
    if not specs:
        return {
            "failed_items": [],
            "batch_tasks": 0,
            "batch_success": 0,
            "batch_fallback": 0,
            "batch_failed": 0,
            "batch_workers": 1,
            "batch_chunks": 0,
            "chunk_size": max(1, int(chunk_size)),
            "effective_backend": "none",
            "tensorops_dispatch_observed": False,
            "peak_rss_mb": 0,
            "rss_after_item_mb": 0,
            "temporary_bytes": 0,
            "duration_ms": 0.0,
            "fallback_reasons": {},
        }

    chunk_size = max(1, int(chunk_size))
    chunks = [
        specs[index : index + chunk_size]
        for index in range(0, len(specs), chunk_size)
    ]
    started = time.perf_counter()
    stats = {
        "failed_items": [],
        "batch_tasks": len(specs),
        "batch_success": 0,
        "batch_fallback": 0,
        "batch_failed": 0,
        "batch_workers": 1,
        "batch_chunks": len(chunks),
        "chunk_size": chunk_size,
        "effective_backend": "failed",
        "tensorops_dispatch_observed": False,
        "peak_rss_mb": 0,
        "rss_after_item_mb": 0,
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
        stats["batch_failed"] += 1
        stats["failed_items"].append(spec["item"])
        record_reason(reason)
        try:
            os.remove(spec["temporary_path"])
        except OSError:
            pass

    os.makedirs(os.path.join(UI.Ortho4XP_dir, "tmp"), exist_ok=True)
    for chunk_index, chunk in enumerate(chunks, start=1):
        request_fd, request_path = tempfile.mkstemp(
            prefix=".tensorops-dds-",
            suffix=".json",
            dir=os.path.join(UI.Ortho4XP_dir, "tmp"),
        )
        request = {
            "version": 1,
            "pack": pack_path,
            "items": [spec["request"] for spec in chunk],
        }
        try:
            with os.fdopen(request_fd, "w", encoding="utf-8") as stream:
                json.dump(request, stream, separators=(",", ":"))
            try:
                result = subprocess.run(
                    [as_helper, "--tensorops-dds-batch", request_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                output = result.stdout or ""
            except Exception as error:
                result = None
                output = ""
                reason = f"ashelper_exception:{type(error).__name__}"
                for spec in chunk:
                    mark_failed(spec, reason)
                UI.vprint(1, f"   TensorOps direct DDS: chunk {chunk_index}/{len(chunks)} failed ({reason})")
                continue

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
                record_reason(signal_reason)

            for index, spec in enumerate(chunk):
                fields = {}
                if index < len(item_lines):
                    fields = dict(
                        field.split("=", 1)
                        for field in item_lines[index].split()
                        if "=" in field
                    )
                try:
                    item_rss_mb = int(float(fields.get("rss_mb", 0) or 0))
                except (TypeError, ValueError):
                    item_rss_mb = 0
                stats["peak_rss_mb"] = max(stats["peak_rss_mb"], item_rss_mb)
                stats["rss_after_item_mb"] = item_rss_mb
                if fields.get("tensorops_dispatch_observed", "").lower() == "true":
                    stats["tensorops_dispatch_observed"] = True

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
                        or ("process_signal_9" if result.returncode == -9 else None)
                        or (f"ashelper_exit_{result.returncode}" if result.returncode else None)
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
                        record_reason(fields["fallback_reason"])

            if result.returncode < 0:
                for spec in chunk[len(item_lines):]:
                    if spec["item"] not in stats["failed_items"]:
                        mark_failed(spec, signal_reason or f"process_signal_{abs(result.returncode)}")
        finally:
            try:
                os.close(request_fd)
            except OSError:
                pass
            try:
                os.remove(request_path)
            except OSError:
                pass
        completed = min(chunk_index * chunk_size, len(specs))
        UI.vprint(1, f"   TensorOps direct DDS: {completed}/{len(specs)}")

    stats["duration_ms"] = (time.perf_counter() - started) * 1000.0
    if stats["batch_success"] == 0:
        stats["effective_backend"] = "failed"
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
        f"batch_failed={stats['batch_failed']} batch_workers=1 "
        f"batch_chunks={stats['batch_chunks']} chunk_size={stats['chunk_size']} "
        f"tensorops_dispatch_observed={str(stats['tensorops_dispatch_observed']).lower()} "
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

    if UI.red_flag:
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
                UI.red_flag = True
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
                if int(item_zoomlevel) >= 18:
                    return False
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

            # TensorOps direct DDS work is split into one child process per
            # bounded chunk. The child exit is a hard memory-reclaim boundary;
            # successful chunks remain published when a later chunk fails.
            if conversion_success and defer_tensorops_batch:
                UI.vprint(
                    1,
                    "-> Executing TensorOps direct DDS batch "
                    f"({len(tensorops_batch_items)} images)...",
                )
                tensorops_direct_specs = []
                tensorops_direct_failed = []
                tensorops_direct_cleanup = []
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

                batch_result = _run_tensorops_direct_dds_batch(
                    as_helper,
                    tensorops_model_path,
                    tensorops_direct_specs,
                    chunk_size=8,
                )
                tensorops_direct_failed.extend(batch_result["failed_items"])
                fallback_success = True
                if tensorops_direct_failed:
                    fallback_config = dict(config_data)
                    fallback_config["upscale_backend"] = "ci_lanczos"
                    fallback_config["defer_fp8_batch"] = False
                    fallback_config["defer_gpu_batch"] = False
                    fallback_success = _run_cpu_fallback(
                        tensorops_direct_failed,
                        fallback_config,
                        2,
                        {
                            "done": 0,
                            "bar": 3,
                            "message": "TensorOps direct DDS fallback",
                        },
                    )
                success_count += batch_result["batch_success"]
                success_count += len(tensorops_direct_failed) if fallback_success else 0
                conversion_success = bool(
                    fallback_success and success_count == len(convert_list)
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
                    batch_output_specs.append(
                        (batch_tmp_path, out_file_path, target_fmt, input_path)
                    )
                
                if conversion_success and batch_args:
                    batch_attempted = True
                    chunk_size = 64
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
                            if batch_result.stdout:
                                output_level = 0 if ret != 0 else 2
                                for line in batch_result.stdout.splitlines():
                                    UI.vprint(output_level, "      " + line)
                            if ret != 0:
                                UI.vprint(1, f"ERROR: GPU Batch DDS conversion failed with return code {ret}")
                                conversion_success = False
                                break
                        except Exception as e:
                            UI.vprint(1, f"ERROR: Execution of GPU Batch DDS conversion failed: {str(e)}")
                            conversion_success = False
                            break
                    if conversion_success:
                        invalid_outputs = []
                        for temp_path, _, target_fmt, input_path in batch_output_specs:
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
                        if invalid_outputs:
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

            if UI.red_flag:
                UI.vprint(1, "DDS conversion process interrupted.")
            elif dico_conv_progress["done"] >= 1:
                UI.vprint(1, " *DDS conversion of textures completed.")
    if convert_launched and not conversion_success:
        UI.exit_message_and_bottom_line("ERROR: DDS conversion failed.")
        return 0
    if UI.red_flag:
        UI.exit_message_and_bottom_line()
        return 0
    try:
        synced_terrain_count = _sync_terrain_load_centers(tile)
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
    cancelled = bool(UI.red_flag)
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
        "cancelled": bool(UI.red_flag),
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
            "gpu_batch_wait_ms": int(
                getattr(tile, "gpu_batch_wait_ms", gpu_batch_wait_ms)
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
            "use_gpu_for_dem_smoothing": bool(
                getattr(tile, "use_gpu_for_dem_smoothing", False)
            ),
        }
    )
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


def build_all(tile):
    return _start_full_pipeline(tile, include_overlays=True)


def build_continuous(tile):
    """Build all core stages with DSF-budget retries for the CLI path."""
    return _start_full_pipeline(tile, include_overlays=False)


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
    for stage_name, stage in stages:
        stage_error = None
        stage_traceback = None
        metrics = getattr(tile, "_performance_metrics", None)
        stage_context = metrics.stage(stage_name) if metrics is not None else nullcontext()
        try:
            with stage_context:
                stage_succeeded = bool(stage(tile))
        except Exception as error:
            stage_succeeded = False
            stage_error = error
            stage_traceback = traceback.format_exc()
        if not stage_succeeded or UI.red_flag:
            _report_pipeline_failure(
                tile, stage_name, stage_error, stage_traceback
            )
            return 0
    return 1


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
    base_settings = _snapshot_auto_reduce_settings(tile)
    budget = DSF_BUDGET.normalize_budget(
        getattr(tile, "dsf_node_budget", DSF_BUDGET.DEFAULT_DSF_NODE_BUDGET)
    )
    transaction = None
    best_candidate = None
    performance_metrics = getattr(tile, "_performance_metrics", None)

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
            return True
        except Exception as error:
            UI.logprint("ERROR: Could not remove tile transaction staging:", repr(error))
            UI.vprint(0, "ERROR: Could not remove tile transaction staging:", error)
            return False

    try:
        transaction = _BuildTransaction(tile)
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

            pipeline_result = (
                _run_pipeline_once(tile)
                if retry_stage == "vector data"
                else _run_pipeline_once(tile, start_stage=retry_stage)
            )
            if not pipeline_result:
                failure = getattr(tile, "last_pipeline_failure", {})
                if best_candidate is None or failure.get("cancelled"):
                    if performance_metrics is not None:
                        performance_metrics.end_attempt(
                            "cancelled" if failure.get("cancelled") else "failed"
                        )
                    _restore_auto_reduce_settings(tile, base_settings)
                    restore_and_cleanup(
                        best_candidate["snapshot"]
                        if best_candidate is not None
                        else "initial"
                    )
                    return 0
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
        UI.vprint(
            1,
            "[Auto-Reduce] Saved settings from selected full-pipeline attempt {} to tile config.".format(
                best_candidate["attempt"]
            ),
        )
        try:
            transaction.cleanup()
        except Exception as error:
            UI.logprint("WARNING: Could not remove tile transaction staging:", repr(error))
            UI.vprint(1, "WARNING: Tile transaction staging remains for recovery:", error)

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
                if best_candidate is not None:
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
    if include_overlays and getattr(tile, "build_overlays_in_all_in_one", False):
        UI.vprint(0, "-> Automatically extracting overlays (All in one)...")
        overlay_context = (
            performance_metrics.stage("overlay")
            if performance_metrics is not None
            else nullcontext()
        )
        with overlay_context:
            overlay_result = _build_overlay_stage(tile)
        if not overlay_result or UI.red_flag:
            _report_pipeline_failure(tile, "overlay extraction")
            return 0
    return 1

################################################################################
def _run_batch_stage(tile, stage_name, stage):
    stage_error = None
    try:
        succeeded = bool(stage(tile))
    except Exception as error:
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
    if succeeded and not UI.red_flag:
        return True
    if UI.red_flag:
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
):
    """Run overlay extraction with process-local UI/module state."""
    try:
        import O4_Overlay_Utils as overlay_utils
        import O4_UI_Utils as ui_utils

        ui_utils.is_working = 0
        ui_utils.red_flag = False
        overlay_utils.custom_overlay_src = custom_overlay_src
        overlay_utils.ovl_exclude_pol = list(ovl_exclude_pol)
        overlay_utils.ovl_exclude_net = list(ovl_exclude_net)
        result_queue.put(int(bool(overlay_utils.build_overlay(lat, lon))))
    except Exception:
        result_queue.put(0)


def _build_overlay_stage(tile):
    """Use the historical serial overlay path unless explicitly opted in."""
    if not getattr(tile, "enable_parallel_overlay", enable_parallel_overlay):
        return OVL.build_overlay(tile.lat, tile.lon)
    # A parallel tile worker already has isolated process state. Nesting a
    # second child here would only add startup cost and lose useful logging.
    if getattr(tile, "_parallel_tile_worker", False):
        return OVL.build_overlay(tile.lat, tile.lon)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_overlay_worker_entry,
        args=(
            tile.lat,
            tile.lon,
            getattr(OVL, "custom_overlay_src", ""),
            getattr(OVL, "ovl_exclude_pol", []),
            getattr(OVL, "ovl_exclude_net", []),
            result_queue,
        ),
        name="Ortho4XP-overlay-{}".format(FNAMES.short_latlon(tile.lat, tile.lon)),
    )
    process.start()
    try:
        while process.is_alive():
            if UI.red_flag:
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
    if not getattr(tile, "use_gpu_acceleration", False):
        return False
    dds_converter = getattr(tile, "dds_converter", "nvcompress")
    upscale_backend = IMG.normalize_upscale_backend(
        getattr(tile, "upscale_backend", "none")
    )
    return bool(
        dds_converter == "TextureConverter"
        or upscale_backend in ("metalfx_spatial", "tensorops")
    )


def _parallel_tile_worker(payload):
    """Build one tile in a spawn-isolated process for opt-in batch mode."""
    try:
        import O4_Config_Utils as CFG
        import O4_UI_Utils as worker_ui

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
        worker_ui.is_working = 0
        worker_ui.red_flag = False

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
                    lambda current_tile: _build_overlay_stage(current_tile),
                )
            )

        for stage_name, stage in stages:
            stage_context = _host_gpu_semaphore(
                _parallel_tile_stage_uses_gpu(tile, stage_name)
            )
            with stage_context:
                try:
                    succeeded = bool(stage(tile))
                except Exception as error:
                    return lat, lon, False, "{}: {}".format(stage_name, error)
            if not succeeded or worker_ui.red_flag:
                return lat, lon, False, "{} failed".format(stage_name)
        return lat, lon, True, None
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
        }
        for lat, lon in list_lat_lon
    ]
    context = multiprocessing.get_context("spawn")
    failed = False
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
                if UI.red_flag:
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
                lat, lon, succeeded, detail = result
                completed += 1
                UI.vprint(
                    1,
                    "Parallel tile {}/{}: {}".format(
                        completed, len(payloads), FNAMES.short_latlon(lat, lon)
                    ),
                )
                if not succeeded:
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
    return 0 if failed else 1


################################################################################
def build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
):
    if UI.is_working:
        return 0
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
    k = 0
    for (lat, lon) in list_lat_lon:
        k += 1
        if UI.red_flag:
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

        if do_osm or do_mesh or do_dsf:
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
                    _build_overlay_stage,
                )
            )

        tile_succeeded = True
        for stage_name, stage in stages:
            stage_result = _run_batch_stage(tile, stage_name, stage)
            if stage_result is None:
                return 0
            if not stage_result:
                batch_failed = True
                tile_succeeded = False
                break

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
    UI.lvprint(
        0, "Batch process completed in", UI.nicer_timer(time.time() - timer)
    )
    return 1

################################################################################
def remove_unwanted_textures(tile):
    texture_list = []
    for f in os.listdir(os.path.join(tile.build_dir, "terrain")):
        if f[-4:] != ".ter":
            continue
        # Extract base texture name by removing suffixes
        base_name = f[:-4].replace("_water", "").replace("_sea", "").replace("_overlay", "")
        texture_list.append(base_name + ".dds")
    for f in os.listdir(os.path.join(tile.build_dir, "textures")):
        if not _is_generated_dds_name(f):
            continue
        if f not in texture_list:
            print("Removing obsolete texture", f)
            try:
                os.remove(os.path.join(tile.build_dir, "textures", f))
            except OSError:
                pass
