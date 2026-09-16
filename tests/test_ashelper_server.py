import json
import sys
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
