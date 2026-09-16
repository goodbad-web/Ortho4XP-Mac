import json
import pickle
import sys
import threading
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from O4_ASHelper_Server import ASHelperJSONLServer, ASHelperServerCrashed


def _write_helper(path, body):
    path.write_text(
        f"#!{sys.executable}\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_jsonl_server_reuses_process_and_returns_per_task_results(tmp_path):
    helper = tmp_path / "fake_ashelper"
    _write_helper(
        helper,
        """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'shutdown':
        print(json.dumps({'id': request['id'], 'op': 'shutdown', 'ok': True, 'results': [], 'shutdown': True}), flush=True)
        break
    results = [
        {'id': task['id'], 'ok': True, 'backend': 'fake'}
        for task in request['tasks']
    ]
    print(json.dumps({'id': request['id'], 'op': request['op'], 'ok': True, 'results': results}), flush=True)
""",
    )

    client = ASHelperJSONLServer(str(helper))
    try:
        first = client.convert_batch([{'id': 'a', 'input': 'a', 'output': 'a.dds', 'format': 'BC3'}])
        second = client.convert_batch([{'id': 'b', 'input': 'b', 'output': 'b.dds', 'format': 'BC3'}])
        assert first['ok'] is True
        assert second['results'][0]['id'] == 'b'
        assert client.restart_count == 0
    finally:
        client.close()


def test_jsonl_server_clamps_resident_worker_request(tmp_path):
    helper = tmp_path / "parallelism_ashelper"
    _write_helper(
        helper,
        """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'shutdown':
        print(json.dumps({'id': request['id'], 'op': 'shutdown', 'ok': True, 'results': [], 'shutdown': True}), flush=True)
        break
    results = [
        {'id': task['id'], 'ok': True, 'backend': str(request.get('parallelism'))}
        for task in request['tasks']
    ]
    print(json.dumps({'id': request['id'], 'op': request['op'], 'ok': True, 'results': results}), flush=True)
""",
    )

    client = ASHelperJSONLServer(str(helper))
    try:
        response = client.convert_batch(
            [{'id': 'a', 'input': 'a', 'output': 'a.dds', 'format': 'BC3'}],
            parallelism=99,
        )
        assert response['results'][0]['backend'] == '12'
    finally:
        client.close()


def test_jsonl_server_restarts_once_then_disables_gpu(tmp_path):
    helper = tmp_path / "crashing_ashelper"
    state = tmp_path / "starts"
    _write_helper(
        helper,
        f"""
import json
import pathlib
import sys

state = pathlib.Path({str(state)!r})
starts = int(state.read_text()) if state.exists() else 0
state.write_text(str(starts + 1))
for line in sys.stdin:
    request = json.loads(line)
    if starts < 2:
        raise SystemExit(3)
    print(json.dumps({{'id': request['id'], 'op': request['op'], 'ok': True, 'results': []}}), flush=True)
""",
    )

    client = ASHelperJSONLServer(str(helper), max_restarts=1)
    with pytest.raises(ASHelperServerCrashed) as first_error:
        client.convert_batch([])
    assert first_error.value.restarted is True
    with pytest.raises(ASHelperServerCrashed) as second_error:
        client.convert_batch([])
    assert second_error.value.restarted is False
    assert client.gpu_disabled is True
    client.close()


def test_jsonl_server_timeout_uses_failure_policy(tmp_path):
    helper = tmp_path / "hung_ashelper"
    _write_helper(
        helper,
        """
import sys

for _line in sys.stdin:
    pass
""",
    )

    client = ASHelperJSONLServer(
        str(helper),
        max_restarts=0,
        response_timeout_s=0.05,
    )
    try:
        with pytest.raises(ASHelperServerCrashed) as error:
            client.convert_batch([])
        assert error.value.restarted is False
        assert "timed out" in str(error.value)
        assert client.gpu_disabled is True
    finally:
        client.close()


def test_tile_runtime_handles_are_not_sent_to_spawn_workers():
    from O4_Config_Utils import Tile

    tile = Tile(34, 133, "")
    tile._performance_metrics = threading.RLock()
    tile._ashelper_jsonl_server = ASHelperJSONLServer("/bin/true")

    restored = pickle.loads(pickle.dumps(tile))

    assert not hasattr(restored, "_performance_metrics")
    assert not hasattr(restored, "_ashelper_jsonl_server")
