import json
import sys
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from O4_ASHelper_Server import ASHelperJSONLServer  # noqa: E402
from O4_Shared_Memory import (  # noqa: E402
    SharedMemoryBudget,
    SharedMemoryError,
    SharedMemoryRegion,
    dds_capacity_bytes,
    shared_memory_budget_bytes,
)


def _write_helper(path, body):
    path.write_text("#!{}\n{}".format(sys.executable, body), encoding="utf-8")
    path.chmod(0o755)


def test_shared_memory_budget_is_bounded_and_applies_backpressure():
    assert shared_memory_budget_bytes(0, physical_bytes=128 * 1024**3) == 8 * 1024**3
    assert shared_memory_budget_bytes(0, physical_bytes=16 * 1024**3) == 2 * 1024**3
    assert shared_memory_budget_bytes(4, physical_bytes=128 * 1024**3) == 4 * 1024**3

    budget = SharedMemoryBudget(10)
    assert budget.acquire(6)
    assert not budget.acquire(5, timeout=0.001)
    budget.release(6)
    assert budget.snapshot()["in_use_bytes"] == 0


def test_shared_memory_region_descriptor_and_cleanup():
    try:
        region = SharedMemoryRegion(32, label="test")
    except SharedMemoryError as error:
        pytest.skip("POSIX shared memory is unavailable in this test sandbox: {}".format(error))
    try:
        region.write(b"abc")
        descriptor = region.descriptor(
            width=1,
            height=1,
            stride=3,
            pixel_format="RGB8",
            used_bytes=3,
            read_only=True,
        )
        assert descriptor["capacity"] == 32
        assert descriptor["pixel_format"] == "RGB8"
        assert region.read(3) == b"abc"
    finally:
        region.close()


def test_server_shared_transport_is_versioned_and_path_compatible(tmp_path):
    helper = tmp_path / "echo_transport"
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
        {'id': task['id'], 'ok': True, 'backend': request.get('transport', 'path')}
        for task in request.get('tasks', [])
    ]
    print(json.dumps({'id': request['id'], 'op': request['op'], 'ok': True, 'results': results}), flush=True)
""",
    )
    server = ASHelperJSONLServer(str(helper))
    try:
        shared = server.convert_batch(
            [{"id": "shared", "format": "BC3"}],
            transport="shared_memory",
        )
        path = server.convert_batch(
            [{"id": "path", "format": "BC3"}],
            transport="path",
        )
        assert shared["results"][0]["backend"] == "shared_memory"
        assert path["results"][0]["backend"] == "path"
    finally:
        server.close()


def test_dds_capacity_includes_header_and_full_mip_chain():
    assert dds_capacity_bytes(4, 4, "BC1") == 152
    assert dds_capacity_bytes(4, 4, "BC3") == 176
