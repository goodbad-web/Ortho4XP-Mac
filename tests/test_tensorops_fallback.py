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
