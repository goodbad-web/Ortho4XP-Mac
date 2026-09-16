import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX hosts
    fcntl = None

import O4_File_Names as FNAMES
import O4_UI_Utils as UI


RAM_DISK_PATH = "/Volumes/Ortho4XP_RAM_Disk"
RAM_VOLUME_NAME = "Ortho4XP_RAM_Disk"
RAM_STATE_FILE_NAME = ".Ortho4XP_ramdisk_state.json"
RAM_LOCK_FILE_NAME = ".Ortho4XP_ramdisk.lock"
RAM_SIZE_SECTORS_PER_GB = 2097152

_ACTIVE_SESSION = None
_LOCK_HANDLE = None


class RamDiskError(RuntimeError):
    """Base error for a RAM disk lifecycle operation."""


class RamDiskConflict(RamDiskError):
    """Raised when an existing state cannot be proven to be ours."""


class RamDiskMergeError(RamDiskError):
    """Raised when a RAM-to-SSD merge cannot be completed safely."""


@dataclass
class MergeResult:
    copied: int = 0
    skipped: int = 0
    failures: list = field(default_factory=list)

    @property
    def success(self):
        return not self.failures


@dataclass
class _RamDiskSession:
    ram_disk_path: str
    tmp_path: str
    ortho_path: str
    tmp_backup: str
    ortho_backup: str
    volume_uuid: str
    device_node: str
    use_orthophotos: bool
    session_id: str
    prepared_paths: set = field(default_factory=set)
    restoring_paths: set = field(default_factory=set)
    restored_paths: set = field(default_factory=set)

    @property
    def ram_ortho_path(self):
        return os.path.join(self.ram_disk_path, "Orthophotos")

    def to_dict(self):
        return {
            "version": 1,
            "session_id": self.session_id,
            "ram_disk_path": self.ram_disk_path,
            "tmp_path": self.tmp_path,
            "ortho_path": self.ortho_path,
            "tmp_backup": self.tmp_backup,
            "ortho_backup": self.ortho_backup,
            "volume_uuid": self.volume_uuid,
            "device_node": self.device_node,
            "volume_name": RAM_VOLUME_NAME,
            "use_orthophotos": self.use_orthophotos,
            "prepared_paths": sorted(self.prepared_paths),
            "restoring_paths": sorted(self.restoring_paths),
            "restored_paths": sorted(self.restored_paths),
            "pid": os.getpid(),
        }


def _ui_text(english, japanese):
    return UI.ui_text(english, japanese)


def _project_root():
    return os.path.abspath(FNAMES.Ortho4XP_dir)


def _state_path():
    return os.path.join(_project_root(), RAM_STATE_FILE_NAME)


def _lock_path():
    # The mount point is global (/Volumes/...), so the lock must also be
    # global; a project-local lock would not protect two projects from each
    # other.
    return os.path.join(tempfile.gettempdir(), RAM_LOCK_FILE_NAME)


def _path_exists(path):
    return os.path.lexists(path)


def _normalise_device_node(value):
    if not value:
        return ""
    value = str(value)
    return value if value.startswith("/dev/") else "/dev/" + value


def _read_volume_info(ram_disk_path=RAM_DISK_PATH):
    """Return diskutil's plist information, or None if it is unavailable."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["diskutil", "info", "-plist", ram_disk_path],
            capture_output=True,
            text=False,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        return plistlib.loads(result.stdout)
    except (OSError, ValueError, plistlib.InvalidFileException, TypeError):
        return None


def _volume_identity(info):
    if not info:
        return "", ""
    volume_uuid = info.get("VolumeUUID") or info.get("DiskUUID") or ""
    device = info.get("DeviceNode") or info.get("DeviceIdentifier") or ""
    return str(volume_uuid), _normalise_device_node(device)


def is_ram_disk_active(ram_disk_path=RAM_DISK_PATH):
    if sys.platform != "darwin" or not _path_exists(ram_disk_path):
        return False
    if os.path.ismount(ram_disk_path):
        return True
    info = _read_volume_info(ram_disk_path)
    if info:
        mounted = info.get("Mounted")
        mount_point = info.get("MountPoint")
        if mounted is True and (
            not mount_point
            or os.path.abspath(mount_point) == os.path.abspath(ram_disk_path)
        ):
            return True
    try:
        result = subprocess.run(
            ["diskutil", "info", ram_disk_path],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and "Mounted: Yes" in result.stdout
    except OSError:
        return False


def _acquire_lock():
    global _LOCK_HANDLE
    if _LOCK_HANDLE is not None:
        return
    if fcntl is None:
        raise RamDiskConflict(
            _ui_text(
                "RAM disk locking is unavailable on this platform.",
                "このプラットフォームではRAMディスクの排他ロックを利用できません。",
            )
        )
    os.makedirs(_project_root(), exist_ok=True)
    handle = open(_lock_path(), "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        handle.close()
        raise RamDiskConflict(
            _ui_text(
                "Another Ortho4XP RAM disk operation is already running.",
                "別のOrtho4XP RAMディスク処理が実行中です。",
            )
        ) from error
    _LOCK_HANDLE = handle


def _release_lock():
    global _LOCK_HANDLE
    if _LOCK_HANDLE is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
    finally:
        _LOCK_HANDLE.close()
        _LOCK_HANDLE = None


def _write_state(session):
    state_path = _state_path()
    fd, temporary_path = tempfile.mkstemp(
        prefix="." + RAM_STATE_FILE_NAME + ".",
        suffix=".tmp",
        dir=_project_root(),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(session.to_dict(), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _read_state():
    state_path = _state_path()
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError, TypeError) as error:
        raise RamDiskConflict(
            _ui_text(
                f"RAM disk state is unreadable: {state_path}",
                f"RAMディスク状態ファイルを読み取れません: {state_path}",
            )
        ) from error
    required = {
        "session_id",
        "ram_disk_path",
        "tmp_path",
        "ortho_path",
        "tmp_backup",
        "ortho_backup",
        "volume_uuid",
        "device_node",
        "volume_name",
        "use_orthophotos",
    }
    if (
        not required.issubset(state)
        or state.get("ram_disk_path") != RAM_DISK_PATH
        or state.get("volume_name") != RAM_VOLUME_NAME
    ):
        raise RamDiskConflict(
            _ui_text(
                "RAM disk state does not match this project.",
                "RAMディスク状態がこのプロジェクトと一致しません。",
            )
        )
    return state


def _remove_state():
    try:
        os.remove(_state_path())
    except FileNotFoundError:
        pass


def _session_from_state(state):
    return _RamDiskSession(
        ram_disk_path=state["ram_disk_path"],
        tmp_path=state["tmp_path"],
        ortho_path=state["ortho_path"],
        tmp_backup=state["tmp_backup"],
        ortho_backup=state["ortho_backup"],
        volume_uuid=str(state["volume_uuid"]),
        device_node=_normalise_device_node(state["device_node"]),
        use_orthophotos=bool(state["use_orthophotos"]),
        session_id=str(state["session_id"]),
        prepared_paths=set(state.get("prepared_paths", [])),
        restoring_paths=set(state.get("restoring_paths", [])),
        restored_paths=set(state.get("restored_paths", [])),
    )


def _validate_state_paths(session):
    expected_tmp = os.path.abspath(FNAMES.Tmp_dir)
    expected_ortho = os.path.abspath(FNAMES.Imagery_dir)
    if session.tmp_path != expected_tmp or session.ortho_path != expected_ortho:
        raise RamDiskConflict(
            _ui_text(
                "RAM disk state points outside the current project paths.",
                "RAMディスク状態が現在のプロジェクトパス外を指しています。",
            )
        )


def _validate_owned_volume(session, require_active=True):
    active = is_ram_disk_active(session.ram_disk_path)
    if not active:
        if require_active:
            return False
        return True
    info = _read_volume_info(session.ram_disk_path)
    volume_uuid, device_node = _volume_identity(info)
    if not volume_uuid or volume_uuid != session.volume_uuid:
        raise RamDiskConflict(
            _ui_text(
                "The mounted RAM volume cannot be proven to belong to Ortho4XP.",
                "マウント済みRAMボリュームがOrtho4XP所有だと確認できません。",
            )
        )
    if info.get("VolumeName") and info.get("VolumeName") != RAM_VOLUME_NAME:
        raise RamDiskConflict(
            _ui_text(
                "The mounted volume name does not match the Ortho4XP RAM disk.",
                "マウント済みボリューム名がOrtho4XPのRAMディスクと一致しません。",
            )
        )
    if device_node and session.device_node and device_node != session.device_node:
        raise RamDiskConflict(
            _ui_text(
                "The mounted RAM device does not match the saved Ortho4XP state.",
                "マウント済みRAMデバイスが保存済みOrtho4XP状態と一致しません。",
            )
        )
    return True


def _expected_link(path, target):
    return os.path.islink(path) and os.path.realpath(path) == os.path.realpath(target)


def _ensure_owned_link_or_missing(path, target):
    if _path_exists(path) and not _expected_link(path, target):
        raise RamDiskConflict(
            _ui_text(
                f"Existing path is not an Ortho4XP RAM link: {path}",
                f"既存パスがOrtho4XPのRAMリンクではありません: {path}",
            )
        )


def _prepare_path(path, backup_path, target):
    """Prepare one SSD path without deleting any unmanaged data."""
    if _path_exists(backup_path):
        raise RamDiskConflict(
            _ui_text(
                f"Existing RAM disk backup requires recovery: {backup_path}",
                f"既存のRAMディスクバックアップを先に復旧してください: {backup_path}",
            )
        )
    if not _path_exists(path):
        os.symlink(target, path)
        return
    if os.path.islink(path):
        raise RamDiskConflict(
            _ui_text(
                f"Existing symlink requires manual recovery: {path}",
                f"既存symlinkは手動復旧が必要です: {path}",
            )
        )
    if not os.path.isdir(path):
        raise RamDiskConflict(
            _ui_text(
                f"Existing file blocks RAM disk setup: {path}",
                f"既存ファイルがRAMディスク設定を妨げています: {path}",
            )
        )
    if os.listdir(path):
        os.rename(path, backup_path)
    else:
        os.rmdir(path)
    os.symlink(target, path)


def _restore_owned_path(path, backup_path, target, source_path=None):
    """Restore one path after all source data has been merged successfully."""
    _ensure_owned_link_or_missing(path, target)
    if os.path.islink(path):
        os.unlink(path)
    elif _path_exists(path):
        raise RamDiskConflict(
            _ui_text(
                f"Path changed while RAM disk was active: {path}",
                f"RAMディスク使用中にパスが変更されました: {path}",
            )
        )
    if _path_exists(backup_path):
        os.rename(backup_path, path)
    else:
        os.makedirs(path, exist_ok=True)
        if source_path and os.path.isdir(source_path):
            result = merge_directories(source_path, path)
            if not result.success:
                raise RamDiskMergeError("; ".join(result.failures))


def restore_path_from_backup(path, backup_path):
    """Compatibility helper that never overwrites an existing path."""
    if not _path_exists(backup_path) or _path_exists(path):
        return False
    try:
        os.rename(backup_path, path)
        return True
    except OSError as error:
        UI.vprint(0, f"[RAMDisk] Failed to restore backup {backup_path}: {error}")
        return False


def restore_path_after_failed_mount(path, backup_path):
    """Compatibility rollback helper with the same no-overwrite guarantee."""
    if restore_path_from_backup(path, backup_path):
        return True
    if not _path_exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except OSError as error:
            UI.vprint(0, f"[RAMDisk] Failed to create rollback path {path}: {error}")
    return False


def _copy_file_atomic(src_file, dest_file):
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix="." + os.path.basename(dest_file) + ".",
        suffix=".tmp",
        dir=os.path.dirname(dest_file),
    )
    os.close(fd)
    try:
        shutil.copy2(src_file, temporary_path)
        if not os.path.isfile(temporary_path):
            raise OSError("temporary copy was not created")
        if os.path.getsize(src_file) != os.path.getsize(temporary_path):
            raise OSError("temporary copy size differs from source")
        os.replace(temporary_path, dest_file)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def merge_directories(src_dir, dest_dir, skip_names=()):
    """Merge a directory without exposing partial destination files."""
    result = MergeResult()
    if not src_dir or not os.path.isdir(src_dir):
        result.failures.append(f"source directory is unavailable: {src_dir}")
        return result
    try:
        os.makedirs(dest_dir, exist_ok=True)
        for root, dirs, files in os.walk(src_dir, followlinks=False):
            if os.path.abspath(root) == os.path.abspath(src_dir):
                dirs[:] = [
                    name for name in dirs
                    if name not in skip_names and not name.endswith(".tmp")
                ]
            else:
                dirs[:] = [name for name in dirs if not name.endswith(".tmp")]
            rel_path = os.path.relpath(root, src_dir)
            target_dir = os.path.join(dest_dir, rel_path) if rel_path != "." else dest_dir
            os.makedirs(target_dir, exist_ok=True)
            for file_name in files:
                if file_name.endswith(".tmp"):
                    result.skipped += 1
                    continue
                src_file = os.path.join(root, file_name)
                dest_file = os.path.join(target_dir, file_name)
                if os.path.islink(src_file):
                    result.failures.append(f"source file is a symlink: {src_file}")
                    continue
                try:
                    _copy_file_atomic(src_file, dest_file)
                    result.copied += 1
                except Exception as error:
                    result.failures.append(f"{src_file}: {error}")
    except Exception as error:
        result.failures.append(f"merge setup failed: {error}")
    for error in result.failures:
        UI.vprint(
            0,
            _ui_text(
                f"[RAMDisk] Copy failed; source retained: {error}",
                f"[RAMDisk] コピーに失敗したためRAM側を保持します: {error}",
            ),
        )
    return result


def safe_merge_directories(src_dir, dest_dir):
    return merge_directories(src_dir, dest_dir).success


def _merge_entry(src_path, dest_path):
    if os.path.isdir(src_path) and not os.path.islink(src_path):
        return merge_directories(src_path, dest_path)
    result = MergeResult()
    if os.path.islink(src_path):
        result.failures.append(f"source entry is a symlink: {src_path}")
        return result
    if os.path.basename(src_path).endswith(".tmp"):
        result.skipped = 1
        return result
    try:
        _copy_file_atomic(src_path, dest_path)
        result.copied = 1
    except Exception as error:
        result.failures.append(f"{src_path}: {error}")
    return result


def _merge_ram_tree_to_backup(source_dir, backup_dir, skip_names=()):
    if not _path_exists(source_dir):
        return MergeResult()
    result = MergeResult()
    try:
        os.makedirs(backup_dir, exist_ok=True)
        for entry in os.scandir(source_dir):
            if entry.name in skip_names:
                result.skipped += 1
                continue
            entry_result = _merge_entry(
                entry.path,
                os.path.join(backup_dir, entry.name),
            )
            result.copied += entry_result.copied
            result.skipped += entry_result.skipped
            result.failures.extend(entry_result.failures)
            if entry_result.success:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        os.remove(entry.path)
                except Exception as error:
                    result.failures.append(
                        f"failed to remove merged source {entry.path}: {error}"
                    )
    except Exception as error:
        result.failures.append(f"RAM tree scan failed: {error}")
    return result


def _detach_owned_session(session):
    if not is_ram_disk_active(session.ram_disk_path):
        return True
    command_target = session.device_node or session.ram_disk_path
    try:
        subprocess.run(
            ["hdiutil", "detach", "-force", command_target],
            capture_output=True,
            text=True,
            check=True,
        )
        UI.vprint(
            1,
            _ui_text(
                "[RAMDisk] RAM disk detached successfully.",
                "[RAMDisk] RAMディスクを正常にdetachしました。",
            ),
        )
        return True
    except Exception as error:
        UI.vprint(
            0,
            _ui_text(
                f"[RAMDisk] RAM disk remains mounted: {error}",
                f"[RAMDisk] RAMディスクはマウント状態のままです: {error}",
            ),
        )
        return False


def detach_ram_disk(ram_disk_path=RAM_DISK_PATH, log_prefix="[RAMDisk]"):
    """Detach only an already validated current session volume."""
    session = _ACTIVE_SESSION
    if session is None or session.ram_disk_path != ram_disk_path:
        UI.vprint(
            0,
            _ui_text(
                f"{log_prefix} Refusing to detach an unowned RAM disk.",
                f"{log_prefix} 所有確認できないRAMディスクのdetachを拒否しました。",
            ),
        )
        return False
    return _detach_owned_session(session)


def _rollback_mount(session):
    for path, backup, target in (
        (session.ortho_path, session.ortho_backup, session.ram_ortho_path),
        (session.tmp_path, session.tmp_backup, session.ram_disk_path),
    ):
        if path == session.ortho_path and not session.use_orthophotos:
            continue
        try:
            if _expected_link(path, target):
                os.unlink(path)
            if _path_exists(backup) and not _path_exists(path):
                os.rename(backup, path)
            elif not _path_exists(path):
                os.makedirs(path, exist_ok=True)
        except Exception as error:
            UI.vprint(
                0,
                _ui_text(
                    f"[RAMDisk] Rollback is incomplete for {path}: {error}",
                    f"[RAMDisk] {path} のロールバックが未完了です: {error}",
                ),
            )


def _preflight_unmanaged_state(session):
    state_path_exists = os.path.isfile(_state_path())
    active = is_ram_disk_active(session.ram_disk_path)
    mount_path_exists = _path_exists(session.ram_disk_path)
    artifacts = (
        (os.path.islink(session.tmp_path) or _path_exists(session.tmp_backup))
        or (os.path.islink(session.ortho_path) or _path_exists(session.ortho_backup))
    )
    if state_path_exists or active or mount_path_exists or artifacts:
        raise RamDiskConflict(
            _ui_text(
                "An existing RAM disk state or symlink requires recovery before setup.",
                "設定前に既存のRAMディスク状態またはsymlinkを復旧する必要があります。",
            )
        )


def mount_ram_disk(size_gb=4, use_orthophotos=False):
    global _ACTIVE_SESSION
    if sys.platform != "darwin":
        UI.vprint(
            1,
            _ui_text(
                "[RAMDisk] RAM disk is only supported on macOS.",
                "[RAMDisk] RAMディスクはmacOSでのみ利用できます。",
            ),
        )
        return False
    if _ACTIVE_SESSION is not None:
        return True

    _acquire_lock()
    ram_disk_path = RAM_DISK_PATH
    tmp_path = os.path.abspath(FNAMES.Tmp_dir)
    ortho_path = os.path.abspath(FNAMES.Imagery_dir)
    session = _RamDiskSession(
        ram_disk_path=ram_disk_path,
        tmp_path=tmp_path,
        ortho_path=ortho_path,
        tmp_backup=tmp_path + "_backup",
        ortho_backup=ortho_path + "_backup",
        volume_uuid="",
        device_node="",
        use_orthophotos=bool(use_orthophotos),
        session_id=uuid.uuid4().hex,
    )
    try:
        _preflight_unmanaged_state(session)
    except BaseException:
        _release_lock()
        raise
    try:
        UI.vprint(
            1,
            _ui_text(
                f"[RAMDisk] Creating {size_gb}GB RAM disk on macOS...",
                f"[RAMDisk] macOS上に{size_gb}GBのRAMディスクを作成しています...",
            ),
        )
        result = subprocess.run(
            [
                "hdiutil",
                "attach",
                "-nomount",
                f"ram://{int(size_gb) * RAM_SIZE_SECTORS_PER_GB}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        output_lines = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        if not output_lines:
            raise RamDiskError("hdiutil did not return a device node")
        session.device_node = _normalise_device_node(output_lines[-1])
        subprocess.run(
            ["diskutil", "erasevolume", "HFS+", RAM_VOLUME_NAME, session.device_node],
            capture_output=True,
            text=True,
            check=True,
        )
        if not is_ram_disk_active(ram_disk_path):
            raise RamDiskError("formatted RAM disk is not mounted")
        volume_uuid, detected_device = _volume_identity(_read_volume_info(ram_disk_path))
        if not volume_uuid:
            raise RamDiskError("could not determine the mounted RAM volume UUID")
        session.volume_uuid = volume_uuid
        if detected_device:
            session.device_node = detected_device
        _write_state(session)

        os.makedirs(session.ram_ortho_path, exist_ok=True)
        _prepare_path(session.tmp_path, session.tmp_backup, session.ram_disk_path)
        session.prepared_paths.add(session.tmp_path)
        _write_state(session)
        if session.use_orthophotos:
            _prepare_path(
                session.ortho_path,
                session.ortho_backup,
                session.ram_ortho_path,
            )
            session.prepared_paths.add(session.ortho_path)
            _write_state(session)
        _ACTIVE_SESSION = session
        UI.vprint(
            1,
            _ui_text(
                f"[RAMDisk] RAM disk mounted at {ram_disk_path}.",
                f"[RAMDisk] RAMディスクを{ram_disk_path}にマウントしました。",
            ),
        )
        return True
    except BaseException as error:
        UI.vprint(
            0,
            _ui_text(
                f"[RAMDisk] Setup failed: {error}",
                f"[RAMDisk] セットアップに失敗しました: {error}",
            ),
        )
        _rollback_mount(session)
        if is_ram_disk_active(session.ram_disk_path):
            _detach_owned_session(session)
        if not is_ram_disk_active(session.ram_disk_path) and not any(
            _path_exists(path) for path in (session.tmp_backup, session.ortho_backup)
        ):
            _remove_state()
        _release_lock()
        if isinstance(error, (KeyboardInterrupt, SystemExit, RamDiskConflict)):
            raise
        return False


def _cleanup_session(session):
    _validate_state_paths(session)
    active = _validate_owned_volume(session, require_active=False)
    path_specs = (
        (
            session.tmp_path,
            session.ram_disk_path,
            session.tmp_backup,
            session.ram_disk_path,
        ),
        (
            session.ortho_path,
            session.ram_ortho_path,
            session.ortho_backup,
            session.ram_ortho_path,
        ),
    )
    plans = []
    for path, target, backup, source in path_specs:
        if path == session.ortho_path and not session.use_orthophotos:
            continue
        if path not in session.prepared_paths:
            if os.path.islink(path) and not _expected_link(path, target):
                raise RamDiskConflict(
                    _ui_text(
                        f"Path changed before RAM disk setup completed: {path}",
                        f"RAMディスク設定完了前にパスが変更されました: {path}",
                    )
                )
            if _expected_link(path, target) or _path_exists(backup):
                session.prepared_paths.add(path)
                _write_state(session)
            else:
                # The durable state may have been written just before this
                # path was prepared. Leave an untouched normal/missing path
                # alone and only detach the owned volume during recovery.
                continue
        if path in session.restored_paths and _path_exists(path) and not os.path.islink(path):
            continue
        if (
            path in session.restoring_paths
            and _path_exists(path)
            and not os.path.islink(path)
            and not _path_exists(backup)
        ):
            session.restored_paths.add(path)
            session.restoring_paths.discard(path)
            _write_state(session)
            continue
        _ensure_owned_link_or_missing(path, target)
        destination = backup
        staged = False
        if active and os.path.isdir(source):
            if not _path_exists(destination):
                destination = tempfile.mkdtemp(
                    prefix=".Ortho4XP_restore_",
                    dir=os.path.dirname(path),
                )
                staged = True
            if path == session.tmp_path:
                result = merge_directories(
                    source,
                    destination,
                    skip_names={"Orthophotos"},
                )
            else:
                result = merge_directories(source, destination)
            if not result.success:
                for plan in plans:
                    if plan[4]:
                        shutil.rmtree(plan[3], ignore_errors=True)
                if staged:
                    shutil.rmtree(destination, ignore_errors=True)
                raise RamDiskMergeError("; ".join(result.failures))
        plans.append((path, target, backup, destination, staged))

    for path, target, backup, destination, staged in plans:
        session.restoring_paths.add(path)
        _write_state(session)
        if staged:
            _ensure_owned_link_or_missing(path, target)
            if os.path.islink(path):
                os.unlink(path)
            elif _path_exists(path):
                raise RamDiskConflict(f"path changed while restoring: {path}")
            os.rename(destination, path)
        else:
            _restore_owned_path(path, backup, target, source_path=None)
        session.restored_paths.add(path)
        session.restoring_paths.discard(path)
        _write_state(session)

    if active and not _detach_owned_session(session):
        return False
    _remove_state()
    return True


def unmount_ram_disk(use_orthophotos=False):
    global _ACTIVE_SESSION
    if sys.platform != "darwin":
        return False
    session = _ACTIVE_SESSION
    if session is None:
        state = _read_state()
        if state is None:
            return True
        session = _session_from_state(state)
    try:
        success = _cleanup_session(session)
    except (RamDiskError, OSError) as error:
        UI.vprint(
            0,
            _ui_text(
                f"[RAMDisk] Cleanup incomplete; RAM state retained: {error}",
                f"[RAMDisk] cleanupが未完了のためRAM状態を保持します: {error}",
            ),
        )
        return False
    if success:
        _ACTIVE_SESSION = None
        _release_lock()
    return success


def recover_orphaned_symlinks():
    """Recover only a state previously recorded by this project."""
    global _ACTIVE_SESSION
    if sys.platform != "darwin":
        return False
    _acquire_lock()
    try:
        state = _read_state()
        tmp_path = os.path.abspath(FNAMES.Tmp_dir)
        ortho_path = os.path.abspath(FNAMES.Imagery_dir)
        artifacts = (
            os.path.islink(tmp_path)
            or _path_exists(tmp_path + "_backup")
            or os.path.islink(ortho_path)
            or _path_exists(ortho_path + "_backup")
            or is_ram_disk_active(RAM_DISK_PATH)
        )
        if state is None:
            if artifacts:
                raise RamDiskConflict(
                    _ui_text(
                        "Unowned RAM disk artifacts were found; no paths were changed.",
                        "所有確認できないRAMディスク残骸を検出したため、パスを変更しませんでした。",
                    )
                )
            return False
        session = _session_from_state(state)
        _validate_state_paths(session)
        _validate_owned_volume(session, require_active=False)
        _cleanup_session(session)
        _ACTIVE_SESSION = None
        UI.vprint(
            1,
            _ui_text(
                "[RAMDisk] Previous RAM disk state recovered.",
                "[RAMDisk] 前回のRAMディスク状態を復旧しました。",
            ),
        )
        return True
    finally:
        _release_lock()


def _owned_session_for_orthophotos():
    session = _ACTIVE_SESSION
    if session is None or not session.use_orthophotos:
        return None
    _validate_owned_volume(session, require_active=True)
    if not _expected_link(session.ortho_path, session.ram_ortho_path):
        raise RamDiskConflict(
            _ui_text(
                "Orthophotos is no longer the owned RAM disk link.",
                "Orthophotosが所有RAMディスクのリンクではなくなっています。",
            )
        )
    return session


def check_and_restore_cached_image(file_path):
    """Restore one SSD cache file into an owned active RAM cache."""
    if not file_path or os.path.exists(file_path):
        return bool(file_path and os.path.exists(file_path))
    try:
        session = _owned_session_for_orthophotos()
    except RamDiskError as error:
        UI.vprint(2, f"[RAMDisk] Cached image restore refused: {error}")
        return False
    if session is None:
        return False
    abs_file_path = os.path.abspath(file_path)
    try:
        if os.path.commonpath([abs_file_path, session.ortho_path]) != session.ortho_path:
            return False
    except ValueError:
        return False
    rel_path = os.path.relpath(abs_file_path, session.ortho_path)
    backup_file_path = os.path.join(session.ortho_backup, rel_path)
    if not os.path.isfile(backup_file_path):
        return False
    os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
    temporary_path = abs_file_path + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copy2(backup_file_path, temporary_path)
        os.replace(temporary_path, abs_file_path)
        UI.vprint(
            2,
            _ui_text(
                f"[RAMDisk] Restored cached image on-demand: {os.path.basename(file_path)}",
                f"[RAMDisk] キャッシュ画像をRAMへ復旧しました: {os.path.basename(file_path)}",
            ),
        )
        return True
    except Exception as error:
        UI.vprint(2, f"[RAMDisk] Warning: Failed to restore cached image {file_path}: {error}")
        return False
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def flush_tile_imagery(lat, lon):
    """Flush all active RAM imagery entries safely after a completed tile."""
    try:
        session = _owned_session_for_orthophotos()
    except RamDiskError as error:
        UI.vprint(0, f"[RAMDisk] Imagery flush refused: {error}")
        return False
    if session is None:
        return None
    if not os.path.isdir(session.ram_ortho_path):
        return True

    # The provider layout may be normal, grouped, code, or custom. Flushing
    # every top-level entry is intentionally conservative: no layout can
    # exhaust RAM simply because its directory convention is unknown.
    result = _merge_ram_tree_to_backup(session.ram_ortho_path, session.ortho_backup)
    if not result.success:
        UI.vprint(
            0,
            _ui_text(
                f"[RAMDisk] Imagery flush for {lat:+d}{lon:+d} failed; RAM data retained.",
                f"[RAMDisk] {lat:+d}{lon:+d} の画像flushに失敗したためRAMデータを保持します。",
            ),
        )
        return False
    UI.vprint(
        1,
        _ui_text(
            f"[RAMDisk] Imagery cache flushed after tile {lat:+d}{lon:+d}.",
            f"[RAMDisk] タイル{lat:+d}{lon:+d}完了後に画像キャッシュをflushしました。",
        ),
    )
    return True


def remove_tile_imagery(lat, lon):
    """Remove tile imagery from both the active RAM tree and SSD backup."""
    session = _owned_session_for_orthophotos()
    if session is None:
        return False
    relatives = {FNAMES.long_latlon(lat, lon), FNAMES.short_latlon(lat, lon)}
    for relative in relatives:
        for root in (session.ram_ortho_path, session.ortho_backup):
            target = os.path.join(root, relative)
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
    return True
