import json
import multiprocessing
import os
import sys
import threading
import time
import numpy
import pytest
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Config_Utils as CFG  # noqa: E402
import O4_File_Names as FNAMES  # noqa: E402
import O4_Mesh_Utils as MESH  # noqa: E402
import O4_OSM_Utils as OSM  # noqa: E402
import O4_Tile_Utils as TILE  # noqa: E402


class Response:
    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.text = content.decode("utf-8", errors="replace")


@pytest.fixture(autouse=True)
def isolated_overpass_request_coordinator(monkeypatch, tmp_path):
    clock = [1000.0]
    sleeps = []

    def sleep(seconds):
        seconds = float(seconds)
        sleeps.append(seconds)
        clock[0] += seconds

    coordinator = OSM._OverpassRequestCoordinator(
        lock_path=str(tmp_path / "overpass.lock"),
        state_path=str(tmp_path / "overpass.json"),
        sleep_func=sleep,
        clock_func=lambda: clock[0],
    )
    monkeypatch.setattr(OSM, "_overpass_request_coordinator", coordinator)
    return SimpleNamespace(clock=clock, sleeps=sleeps)


def _osm_with_node():
    return (
        b'<osm version="0.6">\n'
        b'<node id="1" lat="30.5" lon="130.5"/>\n'
        b'<way id="2">\n<nd ref="1"/>\n<nd ref="1"/>\n'
        b'<tag k="natural" v="water"/>\n</way>\n'
        b"</osm>\n"
    )


def _valid_response_info(status=OSM.VALID_DATA, group_index=None):
    response = {
        "status": status,
        "data_status": status,
        "server": "DE",
        "attempt": 1,
        "http_status": 200,
        "counts": {"node": 1, "way": 1, "relation": 0},
        "payload_sha256": "fixture",
        "attempts": [],
    }
    if group_index is not None:
        response["group_index"] = group_index
    return response


def _coordinator_process_worker(lock_path, state_path, events):
    coordinator = OSM._OverpassRequestCoordinator(
        lock_path=lock_path,
        state_path=state_path,
    )
    with coordinator.slot(cancel_check=lambda: False):
        events.put(("enter", os.getpid()))
        time.sleep(0.05)
        events.put(("exit", os.getpid()))


def test_config_defaults_to_abort_and_is_an_application_setting():
    assert CFG.cfg_vars["osm_download_failure_policy"]["default"] == "abort"
    assert CFG.cfg_vars["osm_download_failure_policy"]["values"] == (
        "abort",
        "continue_degraded",
        "prompt",
    )
    assert "osm_download_failure_policy" in CFG.list_app_vars
    assert OSM.normalize_osm_failure_policy("invalid") == "abort"


def test_prompt_policy_stops_in_headless_execution(monkeypatch):
    monkeypatch.setattr(OSM.UI, "gui", None, raising=False)
    assert OSM.prompt_osm_failure(SimpleNamespace(), {}) == "abort"


def test_compact_empty_osm_documents_are_parseable():
    for payload in (b"<osm></osm>", b"<osm />", b'<?xml version="1.0"?><osm></osm>'):
        layer = OSM.OSM_layer()
        assert layer.update_dicosm(payload, None, None) == 1
        assert not layer.dicosmn
        assert not layer.dicosmfirst["w"]


def test_compact_nonempty_osm_document_is_parseable():
    layer = OSM.OSM_layer()
    assert layer.update_dicosm(
        b'<osm><node id="1" lat="30.5" lon="130.5"/></osm>', None, None
    ) == 1
    assert len(layer.dicosmn) == 1


def test_regional_server_is_not_used_for_a_japan_bbox(monkeypatch):
    responses = iter((Response(200, b"<osm></osm>"),))
    urls = []

    class Session:
        def post(self, url, **kwargs):
            urls.append(url)
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "max_osm_tentatives", 1)
    monkeypatch.setattr(OSM, "overpass_server_choice", "CH")

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]',
        (30, 130, 31, 131),
        return_metadata=True,
    )

    assert result == b"<osm></osm>"
    assert metadata["status"] == OSM.VALID_EMPTY
    assert metadata["server"] == "DE"
    assert all("osm.ch" not in url for url in urls)


def test_empty_response_is_valid_only_from_a_covered_endpoint(monkeypatch):
    responses = iter((Response(200, b"<osm></osm>"),))

    class Session:
        def post(self, *args, **kwargs):
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "max_osm_tentatives", 1)
    monkeypatch.setattr(OSM, "overpass_server_choice", "CH")

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]',
        (46, 7, 47, 8),
        return_metadata=True,
    )

    assert result == b"<osm></osm>"
    assert metadata["status"] == OSM.VALID_EMPTY
    assert metadata["server"] == "CH"


def test_self_closing_empty_response_is_valid(monkeypatch):
    class Session:
        def post(self, *args, **kwargs):
            return Response(200, b"<osm />")

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "max_osm_tentatives", 1)
    monkeypatch.setattr(OSM, "overpass_server_choice", "DE")
    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )
    assert result == b"<osm />"
    assert metadata["status"] == OSM.VALID_EMPTY


def test_regional_server_coverage_is_conservative_and_bbox_aware():
    assert "CH" not in OSM._server_order("CH", (30, 130, 31, 131))
    assert "CH" in OSM._server_order("CH", (46, 7, 47, 8))
    assert "CH" not in OSM._server_order("CH", (45, 7, 46, 8))
    assert "FR" in OSM._server_order("FR", (45, 2, 46, 3))
    assert "FR" not in OSM._server_order("FR", (30, 2, 31, 3))


def test_retry_after_is_preferred_for_429(monkeypatch, isolated_overpass_request_coordinator):
    responses = iter((Response(429, b"", {"Retry-After": "7"}), Response(200, b"<osm></osm>")))
    sleeps = isolated_overpass_request_coordinator.sleeps

    class Session:
        def post(self, *args, **kwargs):
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(OSM, "max_osm_tentatives", 3)

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )

    assert result == b"<osm></osm>"
    assert sum(sleeps) == pytest.approx(7.0)
    assert all(seconds <= 0.25 for seconds in sleeps)
    assert metadata["attempt"] == 2
    assert metadata["attempts"][0]["http_status"] == 429


def test_retry_after_is_honored_before_a_fallback_server(
    monkeypatch, isolated_overpass_request_coordinator
):
    responses = iter(
        (
            Response(429, b"", {"Retry-After": "7"}),
            Response(200, b"<osm></osm>"),
        )
    )
    urls = []
    sleeps = isolated_overpass_request_coordinator.sleeps

    class Session:
        def post(self, url, **kwargs):
            urls.append(url)
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(
        OSM, "_server_order", lambda preferred, bbox: ["DE", "LZ"]
    )
    monkeypatch.setattr(OSM, "max_osm_tentatives", 1)
    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]',
        (30, 130, 31, 131),
        return_metadata=True,
    )

    assert result == b"<osm></osm>"
    assert metadata["server"] == "LZ"
    assert urls == [OSM.overpass_servers["DE"], OSM.overpass_servers["LZ"]]
    assert sum(sleeps) == pytest.approx(7.0)


@pytest.mark.parametrize("status_code", [406, 429])
def test_rate_limit_without_retry_after_uses_thirty_seconds(
    monkeypatch, isolated_overpass_request_coordinator, status_code
):
    responses = iter((Response(status_code, b""), Response(200, b"<osm></osm>")))
    sleeps = isolated_overpass_request_coordinator.sleeps

    class Session:
        def post(self, *args, **kwargs):
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(OSM, "max_osm_tentatives", 3)

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )

    assert result == b"<osm></osm>"
    assert metadata["attempts"][0]["http_status"] == status_code
    assert sum(sleeps) == pytest.approx(30.0)
    assert all(seconds <= 0.25 for seconds in sleeps)


def test_504_and_communication_error_use_round_backoff(
    monkeypatch, isolated_overpass_request_coordinator
):
    responses = iter(
        (
            Response(504, b""),
            OSM.requests.RequestException("connection reset"),
            Response(200, b"<osm></osm>"),
        )
    )
    sleeps = isolated_overpass_request_coordinator.sleeps

    class Session:
        def post(self, *args, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(OSM, "max_osm_tentatives", 3)
    monkeypatch.setattr(OSM.random, "uniform", lambda low, high: 1.0)

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )

    assert result == b"<osm></osm>"
    assert sum(sleeps) == pytest.approx(20.0)
    assert all(seconds <= 0.25 for seconds in sleeps)
    assert [attempt["reason"] for attempt in metadata["attempts"][:2]] == [
        "HTTP status 504",
        "request-error",
    ]


def test_malformed_200_response_is_retried_and_not_valid_data(
    monkeypatch, isolated_overpass_request_coordinator
):
    responses = iter(
        (
            Response(200, b"<osm>"),
            Response(200, b"<osm><node></osm>"),
            Response(200, b"<osm></osm>"),
        )
    )
    sleeps = isolated_overpass_request_coordinator.sleeps

    class Session:
        def post(self, *args, **kwargs):
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(OSM, "max_osm_tentatives", 3)
    monkeypatch.setattr(OSM.random, "uniform", lambda low, high: 1.0)

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )

    assert result == b"<osm></osm>"
    assert metadata["status"] == OSM.VALID_EMPTY
    assert metadata["attempt"] == 3
    assert sum(sleeps) == pytest.approx(20.0)
    assert all(seconds <= 0.25 for seconds in sleeps)
    assert metadata["attempts"][0]["reason"] == "missing closing </osm> tag"
    assert metadata["attempts"][1]["reason"] == "malformed XML response"


def test_verified_cache_is_written_and_reused_without_network(monkeypatch, tmp_path):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest))
    monkeypatch.setattr(OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy.osm"))
    calls = []

    def network(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            _osm_with_node(),
            {
                "status": OSM.VALID_DATA,
                "data_status": OSM.VALID_DATA,
                "server": "DE",
                "attempt": 1,
                "http_status": 200,
                "counts": {"node": 1, "way": 1, "relation": 0},
                "payload_sha256": "fixture",
                "attempts": [],
            },
        )

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    queries = ['way["natural"="water"]']
    first_layer = OSM.OSM_layer()

    assert OSM.OSM_queries_to_OSM_layer(
        queries, first_layer, 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert cache.exists()
    assert manifest.exists()
    assert len(calls) == 1

    def fail_network(*args, **kwargs):
        raise AssertionError("verified cache should avoid network")

    monkeypatch.setattr(OSM, "get_overpass_data", fail_network)
    second_layer = OSM.OSM_layer()
    assert OSM.OSM_queries_to_OSM_layer(
        queries, second_layer, 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert len(second_layer.dicosmfirst["w"]) == 1


def test_unverified_cache_is_preserved_when_new_generation_is_published(
    monkeypatch, tmp_path
):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    old_payload = b"<osm><node id=\"old\" lat=\"30\" lon=\"130\"/></osm>"
    cache.write_bytes(old_payload)
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest))
    monkeypatch.setattr(OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy.osm"))
    monkeypatch.setattr(
        OSM,
        "get_overpass_data",
        lambda *args, **kwargs: (
            _osm_with_node(),
            {
                "status": OSM.VALID_DATA,
                "data_status": OSM.VALID_DATA,
                "server": "DE",
                "attempt": 1,
                "http_status": 200,
                "counts": {"node": 1, "way": 1, "relation": 0},
                "payload_sha256": "fixture",
                "attempts": [],
            },
        ),
    )

    layer = OSM.OSM_layer()
    assert OSM.OSM_queries_to_OSM_layer(
        ['way["natural"="water"]'], layer, 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE

    preserved = Path(str(cache) + ".unverified")
    assert preserved.read_bytes() == old_payload
    assert cache.exists()
    assert manifest.exists()


def test_manifest_bbox_and_hash_mismatch_are_not_reused(monkeypatch, tmp_path):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest))
    monkeypatch.setattr(OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy.osm"))
    network_calls = []

    def network(*args, **kwargs):
        network_calls.append(1)
        return (
            _osm_with_node(),
            {
                "status": OSM.VALID_DATA,
                "data_status": OSM.VALID_DATA,
                "server": "DE",
                "attempt": 1,
                "http_status": 200,
                "counts": {"node": 1, "way": 1, "relation": 0},
                "payload_sha256": "fixture",
                "attempts": [],
            },
        )

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    queries = ['way["natural"="water"]']
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert len(network_calls) == 1

    changed = json.loads(manifest.read_text(encoding="utf-8"))
    changed["bbox"] = [31.0, 130.0, 32.0, 131.0]
    manifest.write_text(json.dumps(changed), encoding="utf-8")
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert len(network_calls) == 2

    cache.write_bytes(cache.read_bytes() + b"tampered")
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert len(network_calls) == 3


def test_unverified_cache_is_not_reused_and_partial_query_is_discarded(
    monkeypatch, tmp_path
):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    cache.write_bytes(b"<osm></osm>")
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest))
    monkeypatch.setattr(OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy.osm"))
    responses = iter(
        (
            (None, {
                "status": OSM.FAILED,
                "reason": "fixture failure",
                "attempts": [{"http_status": 500}],
            }),
        )
    )
    monkeypatch.setattr(OSM, "get_overpass_data", lambda *args, **kwargs: next(responses))
    queries = ['way["natural"="water"]', 'way["waterway"="dock"]']
    layer = OSM.OSM_layer()

    assert OSM.OSM_queries_to_OSM_layer(
        queries, layer, 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_FAILED
    assert not layer.dicosmn
    assert not layer.dicosmfirst["w"]
    assert not manifest.exists()
    assert cache.read_bytes() == b"<osm></osm>"


def test_non_object_manifest_is_ignored_without_raising(monkeypatch, tmp_path):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    cache.write_bytes(b"<osm></osm>")
    manifest.write_text("null", encoding="utf-8")
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest))
    monkeypatch.setattr(OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy.osm"))
    calls = []

    def network(*args, **kwargs):
        calls.append(1)
        return (
            b"<osm></osm>",
            {
                "status": OSM.VALID_EMPTY,
                "data_status": OSM.VALID_EMPTY,
                "server": "DE",
                "attempt": 1,
                "http_status": 200,
                "counts": {"node": 0, "way": 0, "relation": 0},
                "attempts": [],
            },
        )

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    assert OSM.OSM_queries_to_OSM_layer(
        ['way["natural"="water"]'], OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert calls == [1]
    assert Path(str(cache) + ".unverified").exists()
    assert cache.exists()
    assert manifest.exists()


def test_ask_policy_can_load_matching_verified_cache(monkeypatch, tmp_path):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest))
    monkeypatch.setattr(OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy.osm"))
    query = 'way["natural"="water"]'
    monkeypatch.setattr(
        OSM,
        "get_overpass_data",
        lambda *args, **kwargs: (
            _osm_with_node(),
            {
                "status": OSM.VALID_DATA,
                "data_status": OSM.VALID_DATA,
                "server": "DE",
                "attempt": 1,
                "http_status": 200,
                "counts": {"node": 1, "way": 1, "relation": 0},
                "attempts": [],
            },
        ),
    )
    assert OSM.OSM_queries_to_OSM_layer(
        [query], OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE

    prompt_calls = []

    def fail_network(layer_queries, layer, *args, **kwargs):
        layer.last_failure = {"reason": "network failure"}
        layer.last_cache_info = None
        return OSM.OSM_FAILED

    def choose_cache(tile, failure, cache_available=False):
        prompt_calls.append(cache_available)
        return "use_cache"

    monkeypatch.setattr(OSM, "OSM_queries_to_OSM_layer", fail_network)
    monkeypatch.setattr(OSM, "prompt_osm_failure", choose_cache)
    monkeypatch.setattr(OSM, "osm_download_failure_policy", "prompt")
    tile = SimpleNamespace(lat=30, lon=130)
    layer = OSM.OSM_layer()
    assert OSM.run_osm_layer_with_policy(
        tile, "water", [query], layer, tags_of_interest=[], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert prompt_calls == [True]
    assert len(layer.dicosmfirst["w"]) == 1
    assert tile.osm_failures[0]["cache"]["used"] is True


def test_mesh_coastline_uses_common_osm_failure_policy(monkeypatch, tmp_path):
    tile = SimpleNamespace(
        lat=30,
        lon=130,
        apt_curv_tol=3.0,
        curvature_tol=3.0,
        coast_curv_tol=2.0,
    )
    monkeypatch.setattr(MESH.FNAMES, "custom_coastline", lambda *args: "")
    monkeypatch.setattr(MESH.FNAMES, "custom_coastline_dir", lambda *args: "")
    calls = []

    def common_policy(*args, **kwargs):
        calls.append((args, kwargs))
        return OSM.OSM_DEGRADED

    monkeypatch.setattr(MESH.OSM, "run_osm_layer_with_policy", common_policy)
    result = MESH.build_curv_tol_weight_map(
        tile, numpy.ones((1001, 1001), dtype=numpy.float32)
    )
    assert result == OSM.OSM_DEGRADED
    assert calls
    assert calls[0][0][1] == "coastline"


def test_common_osm_policy_marks_failure_as_degraded_only_when_configured(
    monkeypatch
):
    def failed_query(queries, layer, *args, **kwargs):
        layer.last_failure = {"reason": "fixture failure"}
        layer.last_cache_info = None
        return OSM.OSM_FAILED

    monkeypatch.setattr(OSM, "OSM_queries_to_OSM_layer", failed_query)
    tile = SimpleNamespace(lat=30, lon=130)
    layer = OSM.OSM_layer()

    monkeypatch.setattr(OSM, "osm_download_failure_policy", "abort")
    assert OSM.run_osm_layer_with_policy(tile, "water", ["way[\"natural\"]"], layer) == OSM.OSM_FAILED
    assert not hasattr(tile, "osm_degraded_layers")

    tile = SimpleNamespace(lat=30, lon=130)
    layer = OSM.OSM_layer()
    monkeypatch.setattr(OSM, "osm_download_failure_policy", "continue_degraded")
    assert OSM.run_osm_layer_with_policy(tile, "water", ["way[\"natural\"]"], layer) == OSM.OSM_DEGRADED
    assert tile.osm_degraded_layers == {"water"}


def test_standalone_cache_requires_manifest_and_preserves_legacy_data(
    monkeypatch, tmp_path
):
    cache = tmp_path / "extent.osm.bz2"
    cache.write_bytes(_osm_with_node())
    calls = []

    def network(*args, **kwargs):
        calls.append(1)
        return (
            b"<osm></osm>",
            {
                "status": OSM.VALID_EMPTY,
                "data_status": OSM.VALID_EMPTY,
                "server": "DE",
                "http_status": 200,
                "attempt": 1,
                "counts": {"node": 0, "way": 0, "relation": 0},
                "attempts": [],
            },
        )

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    query = 'way["natural"="water"]'
    layer = OSM.OSM_layer()
    assert OSM.OSM_query_to_OSM_layer(
        query, "", layer, "all", cached_file_name=str(cache)
    ) == 1
    assert calls == [1]
    assert Path(str(cache) + ".unverified").exists()
    assert Path(str(cache) + ".manifest.json").exists()

    def fail_network(*args, **kwargs):
        raise AssertionError("verified standalone cache should avoid network")

    monkeypatch.setattr(OSM, "get_overpass_data", fail_network)
    recycled = OSM.OSM_layer()
    assert OSM.OSM_query_to_OSM_layer(
        query, "", recycled, "all", cached_file_name=str(cache)
    ) == 1
    cache_only = OSM.OSM_layer()
    assert OSM.OSM_query_to_OSM_layer(
        None, "", cache_only, "all", cached_file_name=str(cache)
    ) == 1

    missing_cache_only = OSM.OSM_layer()
    assert OSM.OSM_query_to_OSM_layer(
        None,
        "",
        missing_cache_only,
        "all",
        cached_file_name=str(tmp_path / "missing.osm.bz2"),
    ) == 0


def test_pipeline_stops_after_failed_vector_stage(monkeypatch):
    called = []
    monkeypatch.setattr(TILE.VMAP, "build_poly_file", lambda tile: OSM.OSM_FAILED)
    monkeypatch.setattr(TILE.MESH, "build_mesh", lambda tile: called.append("mesh") or 1)
    monkeypatch.setattr(TILE.MASK, "build_masks", lambda tile: called.append("mask") or 1)
    monkeypatch.setattr(TILE, "build_tile", lambda tile, persist_config=False: called.append("dsf") or 1)
    monkeypatch.setattr(TILE, "_report_pipeline_failure", lambda *args, **kwargs: None)

    result = TILE._run_pipeline_once(SimpleNamespace())

    assert result == OSM.OSM_FAILED
    assert called == []


def test_pipeline_propagates_degraded_vector_result(monkeypatch):
    called = []
    monkeypatch.setattr(TILE.VMAP, "build_poly_file", lambda tile: OSM.OSM_DEGRADED)
    monkeypatch.setattr(TILE.MESH, "build_mesh", lambda tile: called.append("mesh") or 1)
    monkeypatch.setattr(TILE.MASK, "build_masks", lambda tile: called.append("mask") or 1)
    monkeypatch.setattr(TILE, "build_tile", lambda tile, persist_config=False: called.append("dsf") or 1)

    result = TILE._run_pipeline_once(SimpleNamespace())

    assert result == OSM.OSM_DEGRADED
    assert called == ["mesh", "mask", "dsf"]


def test_pipeline_propagates_degraded_result_from_later_osm_dependent_stage(
    monkeypatch,
):
    called = []
    monkeypatch.setattr(TILE.VMAP, "build_poly_file", lambda tile: OSM.OSM_COMPLETE)
    monkeypatch.setattr(TILE.MESH, "build_mesh", lambda tile: OSM.OSM_DEGRADED)
    monkeypatch.setattr(TILE.MASK, "build_masks", lambda tile: called.append("mask") or 1)
    monkeypatch.setattr(TILE, "build_tile", lambda tile, persist_config=False: called.append("dsf") or 1)

    result = TILE._run_pipeline_once(SimpleNamespace())

    assert result == OSM.OSM_DEGRADED
    assert called == ["mask", "dsf"]


def test_degraded_snapshot_can_be_exported_and_old_output_restored(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(FNAMES, "Mask_dir", str(tmp_path / "Masks"))
    tile_dir = tmp_path / "tile"
    tile_dir.mkdir()
    output = tile_dir / ("Data" + FNAMES.short_latlon(1, 2) + ".node")
    output.write_text("old\n", encoding="utf-8")
    tile = type("Tile", (), {"build_dir": str(tile_dir), "lat": 1, "lon": 2, "grouped": False})()

    transaction = TILE._BuildTransaction(tile, preserve_inputs=True)
    output.write_text("degraded\n", encoding="utf-8")
    snapshot = transaction.capture_candidate("degraded")
    staging = transaction.export_snapshot(snapshot, {"status": "DEGRADED"})
    transaction.restore_snapshot_files("initial")
    transaction.cleanup()

    assert output.read_text(encoding="utf-8") == "old\n"
    assert (Path(staging) / "status.json").exists()
    assert (Path(staging) / "tile" / output.name).read_text(encoding="utf-8") == "degraded\n"


def test_degraded_full_pipeline_stages_candidate_and_skips_auto_reduce(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(TILE.FNAMES, "Mask_dir", str(tmp_path / "Masks"))
    build_dir = tmp_path / "tile"
    build_dir.mkdir()
    output = build_dir / ("Data" + FNAMES.short_latlon(1, 2) + ".node")
    output.write_text("old\n", encoding="utf-8")
    tile = SimpleNamespace(
        lat=1,
        lon=2,
        custom_build_dir=str(tmp_path),
        grouped=False,
        build_dir=str(build_dir),
        max_levelled_segs=100,
        water_simplification=1.0,
        cover_zl=18,
        curvature_tol=3.0,
        limit_tris=0.8,
        dsf_node_budget=100,
    )
    attempts = []

    def degraded_pipeline(current_tile):
        attempts.append(1)
        assert current_tile._allow_degraded_intermediate is True
        output.write_text("degraded\n", encoding="utf-8")
        current_tile.osm_degraded_layers = {"roads"}
        current_tile.osm_failures = [{"layer": "roads", "cache": {"used": False}}]
        return OSM.OSM_DEGRADED

    monkeypatch.setattr(TILE, "_run_pipeline_once", degraded_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    result = TILE._build_all(tile, include_overlays=False)

    assert result == OSM.OSM_DEGRADED
    assert attempts == [1]
    assert not hasattr(tile, "_allow_degraded_intermediate")
    assert output.read_text(encoding="utf-8") == "old\n"
    staging_dirs = list(build_dir.parent.glob(".o4xp-degraded-*-*"))
    assert len(staging_dirs) == 1
    assert json.loads((staging_dirs[0] / "status.json").read_text(encoding="utf-8"))["missing_layers"] == [
        "roads"
    ]
    assert (staging_dirs[0] / "tile" / output.name).read_text(encoding="utf-8") == "degraded\n"


def test_osm_failure_on_reduced_attempt_is_not_reclassified_as_budget_success(
    monkeypatch, tmp_path
):
    tile = SimpleNamespace(
        lat=1,
        lon=2,
        custom_build_dir=str(tmp_path),
        grouped=False,
        build_dir=str(tmp_path / "tile"),
        mesh_zl=16,
        max_levelled_segs=100,
        water_simplification=1.0,
        cover_zl=18,
        curvature_tol=3.0,
        limit_tris=0.8,
        dsf_node_budget=100,
    )
    output = Path(tile.build_dir) / "candidate.mesh"
    attempts = []

    def pipeline(current_tile):
        attempt = len(attempts)
        attempts.append(attempt)
        Path(current_tile.build_dir).mkdir(parents=True, exist_ok=True)
        output.write_text("baseline" if attempt == 0 else "failed", encoding="utf-8")
        if attempt == 0:
            current_tile.last_dsf_metrics = {
                "point_count": 200,
                "budget_exceeded": True,
                "structurally_valid": True,
            }
            return OSM.OSM_COMPLETE
        current_tile.osm_failures = [{"layer": "roads"}]
        return OSM.OSM_FAILED

    tile.write_to_config = lambda: 1
    monkeypatch.setattr(TILE, "_run_pipeline_once", pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == OSM.OSM_FAILED
    assert attempts == [0, 1]


def test_same_layer_queries_are_combined_into_one_request(monkeypatch):
    queries = [
        'way["highway"="motorway"]',
        'way["highway"="trunk"]',
        'way["railway"="rail"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        result = (_osm_with_node(), _valid_response_info())
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    layer = OSM.OSM_layer()

    assert OSM.OSM_queries_to_OSM_layer(
        queries, layer, 30, 130, [], cached_suffix=""
    ) == OSM.OSM_COMPLETE
    assert calls == [tuple(queries)]


def test_combined_request_is_not_split_when_it_succeeds(monkeypatch):
    queries = [
        'way["highway"="motorway"]',
        'way["highway"="trunk"]',
        'way["highway"="primary"]',
        'way["highway"="secondary"]',
        'way["railway"="rail"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        result = (_osm_with_node(), _valid_response_info())
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, []
    ) == OSM.OSM_COMPLETE
    assert calls == [tuple(queries)]


def test_final_504_failure_splits_queries_into_two_sequential_groups(monkeypatch):
    queries = [
        'way["highway"="motorway"]',
        'way["highway"="trunk"]',
        'way["highway"="primary"]',
        'way["highway"="secondary"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        if len(calls) == 1:
            result = (
                None,
                {
                    "status": OSM.FAILED,
                    "reason": "all-covered-servers-failed",
                    "attempts": [{"http_status": 504}],
                },
            )
        else:
            result = (_osm_with_node(), _valid_response_info())
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, []
    ) == OSM.OSM_COMPLETE
    assert calls == [
        tuple(queries),
        tuple(queries[:2]),
        tuple(queries[2:]),
    ]


def test_five_queries_use_three_groups_after_final_504(monkeypatch):
    queries = [
        'way["highway"="motorway"]',
        'way["highway"="trunk"]',
        'way["highway"="primary"]',
        'way["highway"="secondary"]',
        'way["railway"="rail"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        if len(calls) == 1:
            result = (
                None,
                {
                    "status": OSM.FAILED,
                    "reason": "all-covered-servers-failed",
                    "attempts": [{"http_status": 504}],
                },
            )
        else:
            result = (b"<osm></osm>", _valid_response_info(OSM.VALID_EMPTY))
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, []
    ) == OSM.OSM_COMPLETE
    assert calls == [
        tuple(queries),
        tuple(queries[:2]),
        tuple(queries[2:4]),
        queries[4],
    ]


def test_non_504_failure_does_not_trigger_query_split(monkeypatch):
    queries = [
        'way["highway"="motorway"]',
        'way["highway"="trunk"]',
        'way["highway"="primary"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        result = (
            None,
            {
                "status": OSM.FAILED,
                "reason": "all-covered-servers-failed",
                "attempts": [{"http_status": 429}],
            },
        )
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, []
    ) == OSM.OSM_FAILED
    assert calls == [tuple(queries)]


def test_split_failure_discards_partial_layer_and_does_not_publish_cache(
    monkeypatch, tmp_path
):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    old_payload = b"<osm><node id=\"old\" lat=\"30\" lon=\"130\"/></osm>"
    cache.write_bytes(old_payload)
    queries = [
        'way["natural"="water"]',
        'way["waterway"="dock"]',
        'way["waterway"="riverbank"]',
        'way["natural"="wetland"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        if len(calls) == 1:
            result = (
                None,
                {
                    "status": OSM.FAILED,
                    "reason": "all-covered-servers-failed",
                    "attempts": [{"http_status": 504}],
                },
            )
        elif len(calls) == 2:
            result = (_osm_with_node(), _valid_response_info())
        else:
            result = (
                None,
                {
                    "status": OSM.FAILED,
                    "reason": "all-covered-servers-failed",
                    "attempts": [{"http_status": 500}],
                },
            )
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(
        OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest)
    )
    monkeypatch.setattr(
        OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy")
    )
    layer = OSM.OSM_layer()

    assert OSM.OSM_queries_to_OSM_layer(
        queries, layer, 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_FAILED
    assert len(calls) == 3
    assert not layer.dicosmn
    assert not manifest.exists()
    assert cache.read_bytes() == old_payload


def test_split_success_publishes_one_grouped_manifest_and_reuses_it(
    monkeypatch, tmp_path
):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    queries = [
        'way["natural"="water"]',
        'way["waterway"="dock"]',
        'way["waterway"="riverbank"]',
        'way["natural"="wetland"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        if len(calls) == 1:
            result = (
                None,
                {
                    "status": OSM.FAILED,
                    "reason": "all-covered-servers-failed",
                    "attempts": [{"http_status": 504}],
                },
            )
        else:
            result = (_osm_with_node(), _valid_response_info())
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(
        OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest)
    )
    monkeypatch.setattr(
        OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy")
    )
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["request_plan"]["group_count"] == 2
    assert saved["request_plan"]["split_reason"] == "http_504"
    assert [response["group_index"] for response in saved["responses"]] == [0, 1]

    def fail_network(*args, **kwargs):
        raise AssertionError("grouped verified cache should avoid network")

    monkeypatch.setattr(OSM, "get_overpass_data", fail_network)
    assert OSM.OSM_queries_to_OSM_layer(
        queries, OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert len(calls) == 3


def test_schema_v1_verified_manifest_remains_compatible(monkeypatch, tmp_path):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    query = 'way["natural"="water"]'
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(
        OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest)
    )
    monkeypatch.setattr(
        OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy")
    )
    monkeypatch.setattr(
        OSM,
        "get_overpass_data",
        lambda *args, **kwargs: (
            _osm_with_node(),
            _valid_response_info(),
        ),
    )
    assert OSM.OSM_queries_to_OSM_layer(
        [query], OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    saved["schema_version"] = 1
    saved.pop("request_plan")
    for response in saved["responses"]:
        response.pop("group_index", None)
        response.pop("group_queries", None)
    manifest.write_text(json.dumps(saved), encoding="utf-8")

    monkeypatch.setattr(
        OSM,
        "get_overpass_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v1 verified cache should avoid network")
        ),
    )
    assert OSM.OSM_queries_to_OSM_layer(
        [query], OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE


def test_grouped_manifest_partition_mismatch_is_not_reused(monkeypatch, tmp_path):
    cache = tmp_path / "tile_water.osm.bz2"
    manifest = tmp_path / "tile_water.osm.bz2.manifest.json"
    query = 'way["natural"="water"]'
    monkeypatch.setattr(OSM.FNAMES, "osm_cached", lambda *args: str(cache))
    monkeypatch.setattr(
        OSM.FNAMES, "osm_cache_manifest", lambda *args: str(manifest)
    )
    monkeypatch.setattr(
        OSM.FNAMES, "osm_old_cached", lambda *args: str(tmp_path / "legacy")
    )
    calls = []

    def network(*args, **kwargs):
        calls.append(1)
        return _osm_with_node(), _valid_response_info()

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    assert OSM.OSM_queries_to_OSM_layer(
        [query], OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    saved["request_plan"]["groups"][0]["queries"] = [
        'way["waterway"="dock"]'
    ]
    manifest.write_text(json.dumps(saved), encoding="utf-8")

    assert OSM.OSM_queries_to_OSM_layer(
        [query], OSM.OSM_layer(), 30, 130, [], cached_suffix="water"
    ) == OSM.OSM_COMPLETE
    assert len(calls) == 2


def test_overpass_request_coordinator_serializes_separate_threads(tmp_path):
    lock_path = str(tmp_path / "overpass.lock")
    state_path = str(tmp_path / "overpass.json")
    coordinators = [
        OSM._OverpassRequestCoordinator(lock_path, state_path),
        OSM._OverpassRequestCoordinator(lock_path, state_path),
    ]
    counters = {"active": 0, "maximum": 0}
    counters_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(coordinator):
        barrier.wait()
        with coordinator.slot(cancel_check=lambda: False):
            with counters_lock:
                counters["active"] += 1
                counters["maximum"] = max(
                    counters["maximum"], counters["active"]
                )
            time.sleep(0.03)
            with counters_lock:
                counters["active"] -= 1

    threads = [
        threading.Thread(target=worker, args=(coordinator,))
        for coordinator in coordinators
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counters["maximum"] == 1


def test_overpass_request_coordinator_serializes_spawn_workers(tmp_path):
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    lock_path = str(tmp_path / "overpass.lock")
    state_path = str(tmp_path / "overpass.json")
    processes = [
        context.Process(
            target=_coordinator_process_worker,
            args=(lock_path, state_path, events),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    event_list = [events.get(timeout=10) for _ in range(4)]
    for process in processes:
        process.join(timeout=10)
    assert all(process.exitcode == 0 for process in processes)
    assert [event[0] for event in event_list] == [
        "enter",
        "exit",
        "enter",
        "exit",
    ]


def test_retry_after_is_shared_between_coordinator_instances(tmp_path):
    clock = [1000.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(float(seconds))
        clock[0] += float(seconds)

    first = OSM._OverpassRequestCoordinator(
        str(tmp_path / "overpass.lock"),
        str(tmp_path / "overpass.json"),
        sleep_func=sleep,
        clock_func=lambda: clock[0],
    )
    second = OSM._OverpassRequestCoordinator(
        str(tmp_path / "overpass.lock"),
        str(tmp_path / "overpass.json"),
        sleep_func=sleep,
        clock_func=lambda: clock[0],
    )
    first.set_cooldown(7.0, "429")
    with second.slot(cancel_check=lambda: False):
        pass
    assert sum(sleeps) == pytest.approx(7.0)
    assert all(seconds <= 0.25 for seconds in sleeps)


def test_invalid_shared_state_waits_safely_before_recreating_it(
    isolated_overpass_request_coordinator
):
    coordinator = OSM._overpass_request_coordinator
    Path(coordinator.state_path).write_text("[]", encoding="utf-8")

    with coordinator.slot(cancel_check=lambda: False):
        pass

    state = json.loads(Path(coordinator.state_path).read_text(encoding="utf-8"))
    assert sum(isolated_overpass_request_coordinator.sleeps) == pytest.approx(30.0)
    assert set(state) >= {"cooldown_until", "reason", "updated_at"}
    assert state["reason"] == "invalid-state"


def test_lock_acquisition_failure_never_starts_network_request(
    monkeypatch, isolated_overpass_request_coordinator
):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append(1)
            raise AssertionError("network must not start without the gate")

    def unavailable(_handle):
        raise OSM._OverpassRequestUnavailable("fixture lock failure")

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(
        OSM._overpass_request_coordinator,
        "_try_lock",
        unavailable,
    )

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )

    assert result is None
    assert metadata["status"] == OSM.FAILED
    assert metadata["reason"] == "request-coordinator-unavailable"
    assert calls == []


def test_cancelled_request_never_starts_network_request(monkeypatch):
    calls = []

    class Session:
        def post(self, *args, **kwargs):
            calls.append(1)
            raise AssertionError("cancelled request must not start network")

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(OSM.UI, "red_flag", True)

    result, metadata = OSM.get_overpass_data(
        'way["natural"="water"]', (30, 130, 31, 131), return_metadata=True
    )

    assert result is None
    assert metadata["status"] == OSM.FAILED
    assert metadata["reason"] == "cancelled"
    assert calls == []


def test_real_fetch_splits_only_after_three_504_rounds_and_shares_backoff(
    monkeypatch, isolated_overpass_request_coordinator
):
    queries = [
        'way["natural"="water"]',
        'way["waterway"="dock"]',
        'way["waterway"="riverbank"]',
        'way["natural"="wetland"]',
    ]
    responses = iter(
        [Response(504, b"")] * 3 + [Response(200, b"<osm></osm>")] * 2
    )
    posted_queries = []

    class Session:
        def post(self, url, **kwargs):
            posted_queries.append(kwargs["data"]["data"])
            return next(responses)

    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "_server_order", lambda preferred, bbox: ["DE"])
    monkeypatch.setattr(OSM.random, "uniform", lambda low, high: 1.0)
    layer = OSM.OSM_layer()

    assert OSM.OSM_queries_to_OSM_layer(
        queries, layer, 30, 130, []
    ) == OSM.OSM_COMPLETE
    assert len(posted_queries) == 5
    assert posted_queries[0].count("way[") == 4
    assert posted_queries[3].count("way[") == 2
    assert posted_queries[4].count("way[") == 2
    assert sum(isolated_overpass_request_coordinator.sleeps) == pytest.approx(
        50.0
    )


def test_split_responses_deduplicate_repeated_osm_element_ids(monkeypatch):
    queries = [
        'way["natural"="water"]',
        'way["waterway"="dock"]',
        'way["waterway"="riverbank"]',
        'way["natural"="wetland"]',
    ]
    calls = []

    def network(query, bbox, server_code=None, return_metadata=False):
        calls.append(query)
        if len(calls) == 1:
            result = (
                None,
                {
                    "status": OSM.FAILED,
                    "attempts": [{"http_status": 504}],
                },
            )
        else:
            result = (_osm_with_node(), _valid_response_info())
        return result if return_metadata else result[0]

    monkeypatch.setattr(OSM, "get_overpass_data", network)
    layer = OSM.OSM_layer()
    assert OSM.OSM_queries_to_OSM_layer(
        queries, layer, 30, 130, []
    ) == OSM.OSM_COMPLETE
    assert len(layer.dicosmn) == 1
    assert len(layer.dicosmfirst["w"]) == 1
