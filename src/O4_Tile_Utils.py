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
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_Imagery_Utils as IMG
import O4_Vector_Map as VMAP
import O4_Mesh_Utils as MESH
import O4_Mask_Utils as MASK
import O4_DSF_Utils as DSF
import O4_Overlay_Utils as OVL
import O4_DSF_Budget as DSF_BUDGET
from O4_Parallel_Utils import parallel_launch, parallel_join, multiprocessing_pool
from PIL import Image

max_convert_slots = 8
max_download_slots = 8
skip_downloads = False
skip_converts = False


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


def _ashelper_metal_available(as_helper):
    """Probe ASHelper once before deferring work to the Metal batch path."""
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
        UI.vprint(1, f"WARNING: Could not probe ASHelper Metal capability: {error}")
        return False

    available = (
        result.returncode == 0
        and "metal_available=true" in (result.stdout or "").splitlines()
    )
    if not available:
        detail = (result.stdout or "").strip()
        if detail:
            UI.vprint(1, "WARNING: ASHelper Metal is unavailable; using CPU conversion.", detail)
        else:
            UI.vprint(1, "WARNING: ASHelper Metal is unavailable; using CPU conversion.")
    return available


def _ashelper_tensorops_available(as_helper):
    """Probe the macOS 27 TensorOps capability before batch deferral."""
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
        UI.vprint(1, f"WARNING: Could not probe ASHelper TensorOps capability: {error}")
        return False

    available = (
        result.returncode == 0
        and "tensorops_available=true" in (result.stdout or "").splitlines()
    )
    if not available:
        UI.vprint(
            1,
            "WARNING: ASHelper TensorOps is unavailable; using the requested fallback path.",
        )
    return available


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
            ) == "tensorops":
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

################################################################################
def download_textures(tile, download_queue, convert_queue):
    UI.vprint(1, "-> Opening download queue with", max_download_slots, "workers.")

    def download_task(*texture_attributes):
        if IMG.build_jpeg_ortho(tile, *texture_attributes):
            convert_queue.put((tile, *texture_attributes))
            return 1
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
                tile, download_queue, convert_queue
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
    if dsf_state["error"] is not None or not dsf_state["result"]:
        UI.exit_message_and_bottom_line("ERROR: DSF construction failed.")
        return 0
    if download_launched and (
        download_state["error"] is not None or not download_state["result"]
    ):
        UI.exit_message_and_bottom_line("ERROR: Texture download failed.")
        return 0
    if convert_launched:
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
            metal_available = (
                _ashelper_metal_available(as_helper)
                if gpu_converter_requested
                else False
            )
            tensorops_available = (
                _ashelper_tensorops_available(as_helper)
                if gpu_converter_requested
                and requested_upscale_backend == "tensorops"
                else False
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

            batch_items_eligible = bool(
                convert_list
                and all(can_defer_to_gpu_batch(item) for item in convert_list)
            )
            tensorops_model_path = getattr(
                tile, 'fp8_model_path', getattr(IMG, 'fp8_model_path', '')
            )
            tensorops_pack_configured = bool(
                tensorops_model_path
                and os.path.isdir(tensorops_model_path)
                and os.path.isfile(os.path.join(tensorops_model_path, "manifest.json"))
            )
            defer_tensorops_batch = bool(
                gpu_batch_enabled
                and tensorops_available
                and requested_upscale_backend == "tensorops"
                and tensorops_pack_configured
                and convert_list
                and all(can_defer_to_tensorops_batch(item) for item in convert_list)
            )
            defer_gpu_batch = bool(
                gpu_batch_enabled
                and batch_items_eligible
                and (
                    requested_upscale_backend != "tensorops"
                    or defer_tensorops_batch
                )
            )
            config_data['defer_gpu_batch'] = defer_gpu_batch
            config_data['defer_fp8_batch'] = defer_tensorops_batch

            dds_error = IMG.dds_format_support_error(dds_converter, dds_format)
            if dds_error:
                UI.vprint(1, f"ERROR: {dds_error}")
                success_count = 0
                conversion_success = False
            else:
                pool_success = multiprocessing_pool(
                    IMG.convert_texture,
                    convert_list,
                    max_convert_slots,
                    progress=dico_conv_progress,
                    init_func=IMG.init_worker,
                    init_args=config_data
                )
                success_count = len(convert_list) if pool_success else 0
                conversion_success = bool(pool_success)

            # TensorOps image work is intentionally batched in one ASHelper
            # process.  FP8SRRuntime caches the validated pack in that
            # process, so a large tile set does not reload the model for each
            # worker or each image. This narrow path accepts direct JPEG
            # inputs; supported color filters and masks are applied by the
            # existing DDS batch after FP8 inference.
            if conversion_success and defer_tensorops_batch:
                UI.vprint(1, "-> Executing TensorOps upscale batch via ASHelper...")
                tensorops_batch_args = [tensorops_model_path]
                tensorops_batch_outputs = []
                tensorops_batch_error = None
                for item in convert_list:
                    item_tile, item_x, item_y, item_z, item_provider = item
                    out_file_name = FNAMES.dds_file_name_from_attributes(
                        item_x, item_y, item_z, item_provider
                    )
                    if item_provider not in IMG.providers_dict:
                        tensorops_batch_error = f"provider source unavailable for {out_file_name}"
                        break
                    file_dir = FNAMES.jpeg_file_dir_from_attributes(
                        item_tile.lat,
                        item_tile.lon,
                        item_z,
                        IMG.providers_dict[item_provider],
                    )
                    jpeg_path = IMG.find_imagery_cache_path(
                        item_x, item_y, item_z, item_provider, file_dir
                    )
                    if not jpeg_path:
                        tensorops_batch_error = f"input source not found for {out_file_name}"
                        break
                    output_path = os.path.join(
                        UI.Ortho4XP_dir,
                        "tmp",
                        out_file_name.replace(
                            ".dds", "_tensorops_upscaled.png"
                        ),
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                    tensorops_batch_args.extend([jpeg_path, output_path])
                    tensorops_batch_outputs.append((jpeg_path, output_path))

                if tensorops_batch_error is None and tensorops_batch_outputs:
                    try:
                        fp8_result = subprocess.run(
                            [as_helper, "--tensorops-upscale-batch"]
                            + tensorops_batch_args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                        if fp8_result.stdout:
                            output_level = 0 if fp8_result.returncode != 0 else 2
                            for line in fp8_result.stdout.splitlines():
                                UI.vprint(output_level, "      " + line)
                        if fp8_result.returncode != 0:
                                tensorops_batch_error = (
                                    f"ASHelper returned {fp8_result.returncode}"
                                )
                    except Exception as error:
                        tensorops_batch_error = f"ASHelper execution failed: {error}"

                if tensorops_batch_error is None:
                    invalid_tensorops_outputs = [
                        output_path
                        for input_path, output_path in tensorops_batch_outputs
                        if not IMG._valid_upscale_output(input_path, output_path)
                    ]
                    if invalid_tensorops_outputs:
                        tensorops_batch_error = (
                            "invalid output: " + ", ".join(invalid_tensorops_outputs)
                        )

                if tensorops_batch_error is not None:
                    for _, output_path in tensorops_batch_outputs:
                        try:
                            os.remove(output_path)
                        except OSError:
                            pass
                    UI.vprint(
                        1,
                        f"WARNING: TensorOps batch failed ({tensorops_batch_error}); "
                        "falling back to per-texture processing.",
                    )
                    fallback_progress = {
                        "done": 0,
                        "bar": 3,
                        "message": "FP8 fallback DDS conversion",
                    }
                    fallback_success = _run_cpu_fallback(
                        convert_list,
                        config_data,
                        max_convert_slots,
                        fallback_progress,
                    )
                    success_count = len(convert_list) if fallback_success else 0
                    conversion_success = bool(fallback_success)
                    defer_gpu_batch = False
                    defer_tensorops_batch = False

            # GPU Batch DDS Conversion integration for macOS
            if conversion_success and defer_gpu_batch:
                import O4_RAMDisk_Utils
                UI.vprint(1, "-> Executing ultra-fast GPU Batch DDS Conversion via ASHelper...")
                batch_args = []
                temp_files_to_delete = []
                batch_generated_mask_files = []
                prepared_input_paths = [None] * len(convert_list)
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
                
                for item_index, item in enumerate(convert_list):
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
                    mask_input = source_is_cache or fp8_batch_input
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
                        source_is_cache or fp8_batch_input
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
                        convert_list, prepared_input_paths
                    )
                    for item in convert_list:
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
                    success_count = len(convert_list) if fallback_success else 0
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
    if not _recover_build_transaction(tile):
        UI.exit_message_and_bottom_line(
            UI.ui_text(
                "ERROR: The previous tile build could not be recovered.",
                "エラー: 前回のタイルビルドを復元できませんでした。",
            )
        )
        return 0
    UI.is_building_all = True
    UI.initialize_build_log(tile.build_dir, tile)
    try:
        return _build_all(tile, include_overlays=include_overlays)
    finally:
        UI.is_building_all = False
        UI.is_working = 0
        UI.flush_build_log(tile.build_dir)


def build_all(tile):
    return _start_full_pipeline(tile, include_overlays=True)


def build_continuous(tile):
    """Build all core stages with DSF-budget retries for the CLI path."""
    return _start_full_pipeline(tile, include_overlays=False)


def _run_pipeline_once(tile):
    stages = (
        ("vector data", VMAP.build_poly_file),
        ("mesh", MESH.build_mesh),
        ("water masks", MASK.build_masks),
        (
            "imagery/DSF",
            lambda current_tile: build_tile(current_tile, persist_config=False),
        ),
    )
    tile.last_pipeline_failure = None
    for stage_name, stage in stages:
        stage_error = None
        stage_traceback = None
        try:
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
        for attempt in range(DSF_BUDGET.MAX_AUTO_REDUCE_ATTEMPTS + 1):
            transaction.discard_current("before-attempt-{}".format(attempt))
            transaction.prepare_attempt()
            if attempt:
                updates = _apply_auto_reduce_attempt(tile, base_settings, attempt)
                UI.vprint(
                    0,
                    "[Auto-Reduce] Full pipeline attempt {}/{}; "
                    "updated settings: {}".format(
                        attempt,
                        DSF_BUDGET.MAX_AUTO_REDUCE_ATTEMPTS,
                        ", ".join(
                            "{}={}".format(name, value)
                            for name, value in updates.items()
                        ),
                    ),
                )
            else:
                _restore_auto_reduce_settings(tile, base_settings)
                UI.vprint(0, "[Auto-Reduce] Full pipeline baseline attempt.")

            if not _run_pipeline_once(tile):
                failure = getattr(tile, "last_pipeline_failure", {})
                if best_candidate is None or failure.get("cancelled"):
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
                break
            if attempt >= DSF_BUDGET.MAX_AUTO_REDUCE_ATTEMPTS:
                break
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
        if not OVL.build_overlay(tile.lat, tile.lon) or UI.red_flag:
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


################################################################################
def build_tile_list(
    tile, list_lat_lon, do_osm, do_mesh, do_mask, do_dsf, do_ovl, do_ptc
):
    if UI.is_working:
        return 0
    UI.red_flag = 0
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
                    lambda current_tile: OVL.build_overlay(
                        current_tile.lat, current_tile.lon
                    ),
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
