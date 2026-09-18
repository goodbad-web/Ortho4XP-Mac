import hashlib
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_DSF_Utils as DSF  # noqa: E402


def _uncompressed_dsf():
    nmed_payload = b"elevation data"
    nmed = b"NMED" + struct.pack("<I", 8 + len(nmed_payload)) + nmed_payload
    nfed = b"NFED" + struct.pack("<I", 8 + len(nmed)) + nmed
    body = b"XPLNEDSF" + struct.pack("<I", 1) + nfed
    return body + hashlib.md5(body).digest()


def _configure_extraction(monkeypatch, tmp_path, source):
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    monkeypatch.setattr(DSF.FNAMES, "Tmp_dir", str(tmp_dir))
    monkeypatch.setattr(
        DSF.FNAMES,
        "resolve_global_scenery_dsf",
        lambda _root, _lat, _lon: str(source),
    )
    monkeypatch.setattr(DSF.UI, "vprint", lambda *_args: None)
    monkeypatch.setattr(
        DSF.UI, "exit_message_and_bottom_line", lambda *_args: None
    )
    return tmp_dir


def test_global_scenery_dsf_extraction_isolated_for_concurrent_calls(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.dsf"
    source.write_bytes(b"7z compressed archive")
    tmp_dir = _configure_extraction(monkeypatch, tmp_path, source)
    work_dirs = []
    barrier = threading.Barrier(2)

    def fake_run(command, **_kwargs):
        archive_path = Path(command[-1])
        work_dir = archive_path.parent
        work_dirs.append(work_dir)
        assert command[3] == "-o" + str(work_dir)
        barrier.wait(timeout=5)
        archive_path.with_suffix("").write_bytes(_uncompressed_dsf())
        return subprocess.CompletedProcess(command, 0, "Everything is Ok\n")

    monkeypatch.setattr(DSF.subprocess, "run", fake_run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(DSF.extract_elevation_and_bathymetry_data, 34, 132)
            for _ in range(2)
        ]
        results = [future.result() for future in futures]

    assert results == [(b"elevation data", b""), (b"elevation data", b"")]
    assert len({str(path) for path in work_dirs}) == 2
    assert all(path.parent == tmp_dir for path in work_dirs)
    assert list(tmp_dir.iterdir()) == []


def test_global_scenery_dsf_extraction_logs_7zip_failure_and_cleans_up(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.dsf"
    source.write_bytes(b"7z compressed archive")
    tmp_dir = _configure_extraction(monkeypatch, tmp_path, source)
    log_messages = []
    errors = []
    monkeypatch.setattr(DSF.UI, "logprint", lambda *args: log_messages.append(args))
    monkeypatch.setattr(
        DSF.UI,
        "exit_message_and_bottom_line",
        lambda *args: errors.append(args),
    )

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 7, "archive error\n")

    monkeypatch.setattr(DSF.subprocess, "run", fake_run)

    assert DSF.extract_elevation_and_bathymetry_data(34, 132) is None

    diagnostic = " ".join(str(value) for message in log_messages for value in message)
    assert "returncode= 7" in diagnostic
    assert "output_exists= False" in diagnostic
    assert "archive error" in diagnostic
    assert errors == [("     ERROR: could not uncompress Global Scenery DSF.",)]
    assert list(tmp_dir.iterdir()) == []
