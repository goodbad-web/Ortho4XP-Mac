import sys
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Tile_Utils as TILE  # noqa: E402


def test_tensorops_intermediate_is_not_reused_by_cpu_fallback(tmp_path, monkeypatch):
    prepared = tmp_path / "tile_tensorops_upscaled.png"
    prepared.write_bytes(b"tensorops intermediate")
    tile = SimpleNamespace(lat=1, lon=2, upscale_backend="tensorops")
    item = (tile, 100, 200, 16, "provider")
    monkeypatch.setattr(TILE.IMG, "providers_dict", {})

    result = TILE._cpu_fallback_convert_args([item], [str(prepared)])

    assert result == [item]


def test_legacy_tensorops_intermediate_is_not_reused_by_cpu_fallback(
    tmp_path, monkeypatch
):
    prepared = tmp_path / "tile_fp8_tensorops_upscaled.png"
    prepared.write_bytes(b"legacy tensorops intermediate")
    tile = SimpleNamespace(lat=1, lon=2, upscale_backend="fp8_tensorops")
    item = (tile, 100, 200, 16, "provider")
    monkeypatch.setattr(TILE.IMG, "providers_dict", {})

    result = TILE._cpu_fallback_convert_args([item], [str(prepared)])

    assert result == [item]


def test_non_tensorops_prepared_input_remains_reusable(tmp_path, monkeypatch):
    prepared = tmp_path / "tile_ci_lanczos_upscaled.png"
    prepared.write_bytes(b"prepared input")
    tile = SimpleNamespace(lat=1, lon=2, upscale_backend="ci_lanczos")
    item = (tile, 100, 200, 16, "provider")
    monkeypatch.setattr(TILE.IMG, "providers_dict", {})

    result = TILE._cpu_fallback_convert_args([item], [str(prepared)])

    assert result == [(*item, "dds", str(prepared))]


def test_cpu_fallback_forces_lanczos_backend(monkeypatch):
    captured = {}

    def fake_pool(*args, **kwargs):
        captured.update(kwargs["init_args"])
        return True

    monkeypatch.setattr(TILE, "multiprocessing_pool", fake_pool)

    assert TILE._run_cpu_fallback(
        [],
        {
            "upscale_backend": "tensorops",
            "use_gpu_acceleration": True,
        },
        1,
        {},
    )
    assert captured["upscale_backend"] == "ci_lanczos"
    assert captured["use_gpu_acceleration"] is False


def _memory_spec(width, height):
    return {"input_size": (width, height)}


def test_tensorops_memory_plan_scales_workers_by_working_set(monkeypatch):
    monkeypatch.setattr(TILE.PERF, "physical_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(TILE.PERF, "peak_rss_bytes", lambda: 1 * 1024**3)

    small = TILE._tensorops_memory_plan(
        [_memory_spec(512, 512)] * 8,
        max_workers=4,
        chunk_size=1,
    )
    large = TILE._tensorops_memory_plan(
        [_memory_spec(2048, 2048)] * 8,
        max_workers=4,
        chunk_size=1,
    )

    assert small["can_run"] is True
    assert small["workers"] == 4
    assert large["can_run"] is True
    assert large["workers"] == 1
    assert large["estimated_worker_mb"] > small["estimated_worker_mb"]


def test_tensorops_memory_plan_skips_when_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(TILE.PERF, "physical_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(TILE.PERF, "peak_rss_bytes", lambda: 2 * 1024**3)

    plan = TILE._tensorops_memory_plan(
        [_memory_spec(2048, 2048)],
        max_workers=4,
        chunk_size=1,
    )

    assert plan["can_run"] is False
    assert plan["workers"] == 0
    assert plan["reason"] == "memory_budget_exceeded"


def test_tensorops_memory_plan_is_conservative_without_physical_memory(monkeypatch):
    monkeypatch.setattr(TILE.PERF, "physical_memory_bytes", lambda: 0)
    monkeypatch.setattr(TILE.PERF, "peak_rss_bytes", lambda: 0)

    plan = TILE._tensorops_memory_plan([_memory_spec(512, 512)], max_workers=4)

    assert plan["can_run"] is True
    assert plan["workers"] == 1
    assert plan["reason"] == "physical_memory_unavailable"
