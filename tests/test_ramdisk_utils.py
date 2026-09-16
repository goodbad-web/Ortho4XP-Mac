import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_RAMDisk_Utils as RAM  # noqa: E402


@pytest.fixture
def ram_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(RAM.sys, "platform", "darwin")
    monkeypatch.setattr(RAM.FNAMES, "Ortho4XP_dir", str(tmp_path))
    monkeypatch.setattr(RAM.FNAMES, "Tmp_dir", str(tmp_path / "tmp"))
    monkeypatch.setattr(RAM.FNAMES, "Imagery_dir", str(tmp_path / "Orthophotos"))
    RAM._ACTIVE_SESSION = None
    if RAM._LOCK_HANDLE is not None:
        RAM._release_lock()
    yield tmp_path
    RAM._ACTIVE_SESSION = None
    if RAM._LOCK_HANDLE is not None:
        RAM._release_lock()


def _session(tmp_path, use_orthophotos=True):
    ram_root = tmp_path / "ramdisk"
    return RAM._RamDiskSession(
        ram_disk_path=str(ram_root),
        tmp_path=str(tmp_path / "tmp"),
        ortho_path=str(tmp_path / "Orthophotos"),
        tmp_backup=str(tmp_path / "tmp_backup"),
        ortho_backup=str(tmp_path / "Orthophotos_backup"),
        volume_uuid="volume-1",
        device_node="/dev/disk1s1",
        use_orthophotos=use_orthophotos,
        session_id="session-1",
    )


def _mock_owned_volume(monkeypatch, active=True):
    monkeypatch.setattr(RAM, "is_ram_disk_active", lambda path: active)
    monkeypatch.setattr(
        RAM,
        "_read_volume_info",
        lambda path: {
            "Mounted": True,
            "MountPoint": path,
            "VolumeUUID": "volume-1",
            "DeviceIdentifier": "disk1s1",
        },
    )


def test_merge_failure_keeps_source_and_existing_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "image.jpg").write_bytes(b"new")
    (destination / "image.jpg").write_bytes(b"old")

    def fail_copy(source_path, destination_path):
        raise OSError("simulated disk full")

    monkeypatch.setattr(RAM, "_copy_file_atomic", fail_copy)

    result = RAM.merge_directories(str(source), str(destination))

    assert not result.success
    assert (source / "image.jpg").read_bytes() == b"new"
    assert (destination / "image.jpg").read_bytes() == b"old"


def test_merge_skips_transient_files(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"complete")
    (source / "image.jpg.tmp").write_bytes(b"partial")

    result = RAM.merge_directories(str(source), str(destination))

    assert result.success
    assert result.skipped == 1
    assert (destination / "image.jpg").read_bytes() == b"complete"
    assert not (destination / "image.jpg.tmp").exists()


def test_mount_refuses_unowned_volume_without_mutating_paths(ram_paths, monkeypatch):
    monkeypatch.setattr(RAM, "is_ram_disk_active", lambda path: True)
    tmp_path = Path(RAM.FNAMES.Tmp_dir)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.symlink_to("/unowned/location")

    with pytest.raises(RAM.RamDiskConflict):
        RAM.mount_ram_disk(size_gb=4, use_orthophotos=False)

    assert tmp_path.is_symlink()
    assert tmp_path.readlink() == Path("/unowned/location")


def test_mount_refuses_existing_backup_without_deleting_it(ram_paths, monkeypatch):
    monkeypatch.setattr(RAM, "is_ram_disk_active", lambda path: False)
    backup = Path(RAM.FNAMES.Tmp_dir + "_backup")
    backup.mkdir(parents=True)
    sentinel = backup / "keep.cache"
    sentinel.write_bytes(b"keep")

    with pytest.raises(RAM.RamDiskConflict):
        RAM.mount_ram_disk(size_gb=4, use_orthophotos=False)

    assert sentinel.read_bytes() == b"keep"


def test_detach_refuses_unowned_ram_disk(monkeypatch):
    called = []
    monkeypatch.setattr(RAM.subprocess, "run", lambda *args, **kwargs: called.append(args))

    assert RAM.detach_ram_disk() is False
    assert called == []


def test_recovery_detaches_volume_only_state(ram_paths, monkeypatch):
    session = _session(ram_paths, use_orthophotos=False)
    monkeypatch.setattr(RAM, "RAM_DISK_PATH", session.ram_disk_path)
    monkeypatch.setattr(RAM, "is_ram_disk_active", lambda path: True)
    monkeypatch.setattr(
        RAM,
        "_read_volume_info",
        lambda path: {
            "Mounted": True,
            "MountPoint": path,
            "VolumeUUID": "volume-1",
            "DeviceIdentifier": "disk1s1",
        },
    )
    detached = []

    def fake_detach(current_session):
        detached.append(current_session.ram_disk_path)
        return True

    monkeypatch.setattr(RAM, "_detach_owned_session", fake_detach)
    RAM._write_state(session)

    assert RAM.recover_orphaned_symlinks() is True
    assert detached == [session.ram_disk_path]
    assert not Path(RAM._state_path()).exists()
    assert Path(session.tmp_path).is_dir()


def test_recovery_keeps_state_when_merge_fails(ram_paths, monkeypatch):
    session = _session(ram_paths, use_orthophotos=False)
    monkeypatch.setattr(RAM, "RAM_DISK_PATH", session.ram_disk_path)
    Path(session.ram_disk_path).mkdir()
    Path(session.tmp_path).symlink_to(session.ram_disk_path)
    (Path(session.ram_disk_path) / "work").mkdir()
    (Path(session.ram_disk_path) / "work" / "image.jpg").write_bytes(b"image")
    _mock_owned_volume(monkeypatch)
    RAM._write_state(session)
    monkeypatch.setattr(
        RAM,
        "merge_directories",
        lambda source, destination, **kwargs: RAM.MergeResult(
            failures=["simulated copy failure"]
        ),
    )

    with pytest.raises(RAM.RamDiskMergeError):
        RAM.recover_orphaned_symlinks()

    assert Path(RAM._state_path()).exists()
    assert Path(session.tmp_path).is_symlink()
    assert (Path(session.ram_disk_path) / "work" / "image.jpg").exists()


def test_unmount_merges_tmp_and_orthophotos_without_cross_contamination(
    ram_paths, monkeypatch
):
    session = _session(ram_paths, use_orthophotos=True)
    ram_root = Path(session.ram_disk_path)
    ram_ortho = Path(session.ram_ortho_path)
    ram_root.mkdir()
    ram_ortho.mkdir()
    Path(session.tmp_path).symlink_to(session.ram_disk_path)
    Path(session.ortho_path).symlink_to(session.ram_ortho_path)
    Path(session.tmp_backup).mkdir()
    Path(session.ortho_backup).mkdir()
    (Path(session.tmp_backup) / "old.cache").write_bytes(b"old tmp")
    (Path(session.ortho_backup) / "old.jpg").write_bytes(b"old image")
    (ram_root / "new.cache").write_bytes(b"new tmp")
    (ram_ortho / "new.jpg").write_bytes(b"new image")
    _mock_owned_volume(monkeypatch)
    monkeypatch.setattr(RAM, "_detach_owned_session", lambda current_session: True)
    RAM._ACTIVE_SESSION = session
    RAM._write_state(session)

    assert RAM.unmount_ram_disk(use_orthophotos=True) is True

    assert not Path(session.tmp_path).is_symlink()
    assert not Path(session.ortho_path).is_symlink()
    assert (Path(session.tmp_path) / "old.cache").read_bytes() == b"old tmp"
    assert (Path(session.tmp_path) / "new.cache").read_bytes() == b"new tmp"
    assert (Path(session.ortho_path) / "old.jpg").read_bytes() == b"old image"
    assert (Path(session.ortho_path) / "new.jpg").read_bytes() == b"new image"
    assert not (Path(session.tmp_path) / "Orthophotos").exists()
    assert not Path(RAM._state_path()).exists()


def test_flush_handles_normal_grouped_code_and_custom_layout(ram_paths, monkeypatch):
    session = _session(ram_paths, use_orthophotos=True)
    Path(session.ram_disk_path).mkdir()
    Path(session.ram_ortho_path).mkdir()
    Path(session.ortho_path).symlink_to(session.ram_ortho_path)
    layout_dirs = {
        "normal": Path(session.ram_ortho_path) / "+01+002",
        "grouped": Path(session.ram_ortho_path) / "+01+002+01+003",
        "code": Path(session.ram_ortho_path) / "BI",
        "custom": Path(session.ram_ortho_path) / "custom-provider",
    }
    for label, directory in layout_dirs.items():
        directory.mkdir()
        (directory / f"{label}.jpg").write_bytes(label.encode())
    _mock_owned_volume(monkeypatch)
    RAM._ACTIVE_SESSION = session

    assert RAM.flush_tile_imagery(1, 2) is True

    for label, directory in layout_dirs.items():
        assert not directory.exists()
        assert (Path(session.ortho_backup) / directory.name / f"{label}.jpg").read_bytes() == label.encode()


def test_flush_failure_keeps_ram_tree(ram_paths, monkeypatch):
    session = _session(ram_paths, use_orthophotos=True)
    Path(session.ram_disk_path).mkdir()
    Path(session.ram_ortho_path).mkdir()
    Path(session.ortho_path).symlink_to(session.ram_ortho_path)
    source = Path(session.ram_ortho_path) / "BI"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"image")
    _mock_owned_volume(monkeypatch)
    RAM._ACTIVE_SESSION = session
    monkeypatch.setattr(
        RAM,
        "_copy_file_atomic",
        lambda source_path, destination_path: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert RAM.flush_tile_imagery(1, 2) is False
    assert (source / "image.jpg").exists()


def test_unmount_failure_keeps_links_backups_and_ram_data(ram_paths, monkeypatch):
    session = _session(ram_paths, use_orthophotos=False)
    ram_root = Path(session.ram_disk_path)
    ram_root.mkdir()
    Path(session.tmp_path).symlink_to(session.ram_disk_path)
    Path(session.tmp_backup).mkdir()
    source = ram_root / "work"
    source.mkdir()
    (source / "image.cache").write_bytes(b"image")
    _mock_owned_volume(monkeypatch)
    RAM._ACTIVE_SESSION = session
    RAM._write_state(session)
    monkeypatch.setattr(
        RAM,
        "_copy_file_atomic",
        lambda source_path, destination_path: (_ for _ in ()).throw(OSError("disk full")),
    )

    assert RAM.unmount_ram_disk() is False
    assert Path(session.tmp_path).is_symlink()
    assert Path(session.tmp_backup).is_dir()
    assert (source / "image.cache").exists()
    assert Path(RAM._state_path()).exists()


def test_gui_exit_waits_for_worker(monkeypatch):
    import O4_GUI_Utils as GUI

    class Worker:
        def is_alive(self):
            return True

    class Status:
        def set(self, value):
            self.value = value

    gui = GUI.Ortho4XP_GUI.__new__(GUI.Ortho4XP_GUI)
    gui.working_thread = Worker()
    gui.status_var = Status()
    scheduled = []
    destroyed = []
    gui.after = lambda delay, callback: scheduled.append((delay, callback))
    gui.destroy = lambda: destroyed.append(True)

    gui._finish_exit()

    assert scheduled and scheduled[0][0] == 100
    assert destroyed == []


def test_atomic_jpeg_save_removes_partial_file(tmp_path, monkeypatch):
    from PIL import Image
    import O4_Imagery_Utils as IMG

    output = tmp_path / "image.jpg"
    output.write_bytes(b"previous")
    image = Image.new("RGB", (2, 2), "red")

    def interrupted_save(path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted save")

    monkeypatch.setattr(image, "save", interrupted_save)
    with pytest.raises(OSError):
        IMG._save_image_atomically(image, str(output), format="JPEG")

    assert output.read_bytes() == b"previous"
    assert list(tmp_path.glob("*.tmp")) == []
