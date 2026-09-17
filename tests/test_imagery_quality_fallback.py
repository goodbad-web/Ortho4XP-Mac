import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Imagery_Utils as IMG  # noqa: E402


def _load_repair_cli():
    path = Path(__file__).parents[1] / "tools" / "repair_imagery_fallback.py"
    spec = importlib.util.spec_from_file_location("repair_imagery_fallback", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parent_crop_uses_child_area_and_stays_in_bounds():
    # Cache filenames are y_top_x_left, so this is x=465760, y=206720.
    assert IMG._parent_crop_box(465760, 206720, 19, 18) == (
        232880,
        103360,
        0,
        0,
        2048,
    )
    parent = IMG._parent_crop_box(465760, 206720, 19, 16)
    assert parent == (58208, 25840, 3072, 0, 512)
    assert parent[2] + parent[4] <= 4096
    assert parent[3] + parent[4] <= 4096


def test_wmts_404_then_lower_zoom_is_recorded(monkeypatch):
    state = IMG._TextureFetchState()
    responses = [(0, "[404]"), (1, Image.new("RGB", (256, 256), "blue"))]

    def fake_request(*_args):
        return responses.pop(0)

    monkeypatch.setattr(IMG, "http_request_to_image", fake_request)
    provider = {
        "code": "TEST",
        "request_type": "tms",
        "grid_type": "webmercator",
        "tile_size": 256,
        "url_template": "https://example.invalid/{zoom}/{x}/{y}",
        "resolutions": {18: 1, 19: 1},
        "top_left_corner": {18: (0, 0), 19: (0, 0)},
    }
    success, image = IMG.get_wmts_image(19, 10, 20, provider, object(), state)

    assert success == 1
    assert image.size == (256, 256)
    assert state.degraded
    assert "404" in state.reasons
    assert "unexpected_lower_zoom" in state.reasons


def test_corrupt_http_response_is_recorded(monkeypatch):
    class Response:
        headers = {"Content-Type": "image/jpeg", "Content-Length": "3"}
        content = b"bad"

        def __str__(self):
            return "[200]"

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    state = IMG._TextureFetchState()
    monkeypatch.setattr(IMG, "max_baddata_retries", 1)
    success, _data = IMG.http_request_to_image(
        256, 256, "https://example.invalid/image.jpg", {}, Session(), state
    )

    assert success == 0
    assert "corrupt_image" in state.reasons


def test_wrong_sized_http_response_is_retried_and_recorded(monkeypatch):
    payload = io.BytesIO()
    Image.new("RGB", (1, 1), "blue").save(payload, format="JPEG")

    class Response:
        headers = {"Content-Type": "image/jpeg"}
        content = payload.getvalue()

        def __str__(self):
            return "[200]"

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    state = IMG._TextureFetchState()
    monkeypatch.setattr(IMG, "max_baddata_retries", 1)
    success, _data = IMG.http_request_to_image(
        256, 256, "https://example.invalid/image.jpg", {}, Session(), state
    )

    assert success == 0
    assert "unexpected_image_dimensions" in state.reasons


def test_transient_corrupt_http_response_does_not_degrade_successful_retry(
    monkeypatch,
):
    payload = io.BytesIO()
    Image.new("RGB", (256, 256), "blue").save(payload, format="JPEG")
    responses = []

    class Response:
        def __init__(self, content):
            self.headers = {"Content-Type": "image/jpeg"}
            self.content = content

        def __str__(self):
            return "[200]"

    responses.extend((Response(b"bad"), Response(payload.getvalue())))

    class Session:
        def get(self, *_args, **_kwargs):
            return responses.pop(0)

    state = IMG._TextureFetchState()
    monkeypatch.setattr(IMG, "max_baddata_retries", 2)
    success, image = IMG.http_request_to_image(
        256, 256, "https://example.invalid/image.jpg", {}, Session(), state
    )

    assert success == 1
    assert image.size == (256, 256)
    assert not state.degraded


def test_provider_max_zl_is_not_degraded(monkeypatch):
    provider = {
        "code": "TEST",
        "grid_type": "webmercator",
        "tile_size": 256,
        "max_zl": 18,
    }
    state = IMG._TextureFetchState()
    monkeypatch.setitem(IMG.providers_dict, "TEST", provider)
    monkeypatch.setattr(
        IMG,
        "build_texture_from_tilbox",
        lambda *_args, **_kwargs: (1, Image.new("RGB", (2048, 2048), "black")),
    )

    success, image, provider_limited = IMG._fetch_orthophoto_image(
        465760, 206720, 19, "TEST", quality_state=state
    )

    assert success == 1
    assert image.size == (4096, 4096)
    assert provider_limited is True
    assert not state.degraded


def test_dark_valid_direct_image_is_not_replaced_by_parent(monkeypatch, tmp_path):
    provider = {"code": "TEST"}
    monkeypatch.setitem(IMG.providers_dict, "TEST", provider)
    monkeypatch.setattr(
        IMG,
        "_fetch_orthophoto_image",
        lambda *_args, **_kwargs: (1, Image.new("RGB", (4096, 4096), "black"), False),
    )

    def unexpected_parent(*_args, **_kwargs):
        pytest.fail("a valid dark image must not trigger parent fallback")

    monkeypatch.setattr(IMG, "_rebuild_from_parent", unexpected_parent)
    target = tmp_path / "206720_465760_TEST19.jpg"

    assert IMG.download_jpeg_ortho(
        str(tmp_path), target.name, 465760, 206720, 19, "TEST"
    ) == 1
    assert target.is_file()
    with Image.open(target) as image:
        assert image.size == (4096, 4096)


def test_direct_failure_uses_parent_and_exhaustion_writes_nothing(monkeypatch, tmp_path):
    provider = {"code": "TEST"}
    monkeypatch.setitem(IMG.providers_dict, "TEST", provider)
    monkeypatch.setattr(
        IMG,
        "_fetch_orthophoto_image",
        lambda *_args, **_kwargs: (0, None, False),
    )
    parent_path = tmp_path / "parent.jpg"
    parent_path.write_bytes(b"parent")
    monkeypatch.setattr(
        IMG,
        "_rebuild_from_parent",
        lambda *_args, **_kwargs: (
            1,
            Image.new("RGB", (4096, 4096), "green"),
            18,
            str(parent_path),
        ),
    )
    target = tmp_path / "206720_465760_TEST19.jpg"
    assert IMG.download_jpeg_ortho(
        str(tmp_path), target.name, 465760, 206720, 19, "TEST"
    ) == 1
    assert target.is_file()

    target.unlink()
    monkeypatch.setattr(IMG, "_rebuild_from_parent", lambda *_args, **_kwargs: (0, None, None, None))
    assert IMG.download_jpeg_ortho(
        str(tmp_path), target.name, 465760, 206720, 19, "TEST"
    ) == 0
    assert not target.exists()


def test_parent_crop_fixture_has_no_black_padding(monkeypatch, tmp_path):
    provider = {"code": "BI", "imagery_dir": "grouped"}
    monkeypatch.setitem(IMG.providers_dict, "BI", provider)
    target_dir = tmp_path / "+30+130" / "+35+139" / "BI_19"
    parent_dir = target_dir.parent / "BI_16"
    parent_dir.mkdir(parents=True)
    parent_path = parent_dir / "25840_58208_BI16.jpg"
    Image.new("RGB", (4096, 4096), (37, 101, 183)).save(
        parent_path, format="JPEG", quality=95
    )

    rebuilt, image, parent_zl, selected = IMG._rebuild_from_parent(
        str(target_dir), 465760, 206720, 19, "BI"
    )

    assert rebuilt == 1
    assert parent_zl == 16
    assert selected == str(parent_path)
    assert image.getbbox() is not None
    assert image.getpixel((0, 0)) != (0, 0, 0)


def test_parent_cache_requires_jpeg_or_webp_and_exact_dimensions(tmp_path):
    parent_dir = tmp_path / "BI_16"
    parent_dir.mkdir()
    wrong_format = parent_dir / "25840_58208_BI16.jpg"
    Image.new("RGB", (4096, 4096), "green").save(wrong_format, format="PNG")
    assert IMG._valid_parent_cache_path(
        str(parent_dir), 58208, 25840, 16, "BI"
    ) is None

    wrong_size = parent_dir / "25840_58208_BI16.webp"
    Image.new("RGB", (2048, 2048), "green").save(wrong_size, format="WEBP")
    assert IMG._valid_parent_cache_path(
        str(parent_dir), 58208, 25840, 16, "BI"
    ) is None


def test_repair_cli_extracts_55_unique_targets(tmp_path):
    repair = _load_repair_cli()
    tile_dir = tmp_path / "zOrtho4XP_+35+139"
    tile_dir.mkdir()
    log = tile_dir / "Ortho4XP_build.log"
    lines = [
        f"[Quality Check] Rebuilt {1000 + i}_{2000 + i}_BI19.jpg from 125_{250}_BI16.jpg."
        for i in range(55)
    ]
    lines.append(lines[0])
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    targets = repair.extract_targets([log])

    assert len(targets) == 55
    assert targets[0]["provider_code"] == "BI"
    assert targets[0]["tile_lat"] == 35
    assert targets[0]["tile_lon"] == 139


def test_repair_cli_extracts_japanese_targets(tmp_path):
    repair = _load_repair_cli()
    tile_dir = tmp_path / "zOrtho4XP_+35+139"
    tile_dir.mkdir()
    log = tile_dir / "Ortho4XP_build.log"
    log.write_text(
        "[品質確認] 206720_465760_BI19.jpg を 25840_58208_BI16.jpg から再構成しました。\n",
        encoding="utf-8",
    )

    targets = repair.extract_targets([log])

    assert len(targets) == 1
    assert targets[0]["provider_code"] == "BI"


def test_repair_cli_extracts_custom_build_dir_from_tile_config(tmp_path):
    repair = _load_repair_cli()
    build_dir = tmp_path / "custom-build"
    build_dir.mkdir()
    (build_dir / "Ortho4XP_+35+139.cfg").write_text("", encoding="utf-8")
    log = build_dir / "Ortho4XP_build.log"
    log.write_text(
        "[Quality Check] Rebuilt 206720_465760_BI19.jpg from 25840_58208_BI16.jpg.\n",
        encoding="utf-8",
    )

    targets = repair.extract_targets([log])

    assert len(targets) == 1
    assert targets[0]["tile_lat"] == 35
    assert targets[0]["tile_lon"] == 139
    assert targets[0]["tile_build_dir"] == str(build_dir.resolve())


def test_repair_cli_loads_tile_from_log_build_dir(monkeypatch, tmp_path):
    repair = _load_repair_cli()
    captured = {}

    class FakeTile:
        def __init__(self, lat, lon, custom_build_dir):
            captured.update(
                lat=lat, lon=lon, custom_build_dir=custom_build_dir
            )

        def read_from_config(self):
            return True

    monkeypatch.setattr(repair.CFG, "Tile", FakeTile)
    build_dir = tmp_path / "custom-build"
    target = {
        "tile_lat": 35,
        "tile_lon": 139,
        "tile_build_dir": str(build_dir.resolve()),
    }

    assert isinstance(repair._load_tile(target), FakeTile)
    assert captured == {
        "lat": 35,
        "lon": 139,
        "custom_build_dir": str(build_dir.resolve()),
    }


def test_repair_cli_uses_tile_upscale_settings_and_restores_globals(monkeypatch):
    repair = _load_repair_cli()
    previous_values = {
        name: getattr(repair.IMG, name, None)
        for name in ("upscale_backend", "upscale_scope", "fp8_model_path")
    }
    tile = SimpleNamespace(
        upscale_backend="tensorops",
        upscale_scope="airport",
        fp8_model_path="/tmp/fp8sr",
        dds_converter="nvcompress",
        dds_format="BC3",
        use_gpu_acceleration=False,
        use_gpu_for_color_filters=False,
    )

    previous = repair._configure_direct_conversion(tile)
    try:
        assert repair.IMG.upscale_backend == "tensorops"
        assert repair.IMG.upscale_scope == "airport"
        assert repair.IMG.fp8_model_path == "/tmp/fp8sr"
    finally:
        repair._restore_direct_conversion(previous)

    for name, value in previous_values.items():
        assert getattr(repair.IMG, name, None) == value


def test_repair_cli_validates_upscaled_dds_dimensions(monkeypatch, tmp_path):
    repair = _load_repair_cli()
    target_path = tmp_path / "target.jpg"
    dds_path = tmp_path / "target.dds"
    target = {
        "name": target_path.name,
        "til_x_left": 465760,
        "til_y_top": 206720,
        "zoomlevel": 19,
        "provider_code": "BI",
    }
    tile = SimpleNamespace(
        upscale_backend="metalfx_spatial",
        upscale_scope="all",
        build_dir=str(tmp_path),
    )
    backup_entries = [
        {"path": str(target_path), "exists": False, "backup_path": None},
        {"path": str(dds_path), "exists": False, "backup_path": None},
    ]
    expected = {}

    monkeypatch.setattr(
        repair,
        "_target_context",
        lambda *_args: (str(tmp_path), str(target_path), str(dds_path)),
    )
    monkeypatch.setattr(
        repair.IMG,
        "_rebuild_from_parent",
        lambda *_args, **_kwargs: (
            1,
            Image.new("RGB", (4096, 4096), "blue"),
            18,
            "parent.jpg",
        ),
    )
    monkeypatch.setattr(repair.IMG, "save_imagery_cache_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(repair, "_image_is_4096", lambda _path: True)
    monkeypatch.setattr(repair.IMG, "convert_texture", lambda *_args, **_kwargs: 1)

    def validate(_path, **kwargs):
        expected.update(kwargs)
        return True, None

    monkeypatch.setattr(repair.IMG, "validate_dds_file", validate)

    result = repair._repair_one(target, tile, backup_entries)

    assert result["status"] == "repaired"
    assert expected["expected_dimensions"] == (8192, 8192)


def test_repair_cli_refreshes_existing_sibling_cache(monkeypatch, tmp_path):
    repair = _load_repair_cli()
    target_name = "206720_465760_BI19.jpg"
    target_path = tmp_path / target_name
    sibling_path = target_path.with_suffix(".webp")
    dds_path = tmp_path / "target.dds"
    sibling_path.write_bytes(b"stale-webp")
    target = {
        "name": target_name,
        "til_x_left": 465760,
        "til_y_top": 206720,
        "zoomlevel": 19,
        "provider_code": "BI",
    }
    tile = SimpleNamespace(
        upscale_backend="none",
        upscale_scope="none",
        build_dir=str(tmp_path),
    )
    backup_entries = [
        {"path": str(target_path), "exists": False, "backup_path": None},
        {"path": str(sibling_path), "exists": True, "backup_path": None},
        {"path": str(dds_path), "exists": False, "backup_path": None},
    ]

    monkeypatch.setattr(
        repair,
        "_target_context",
        lambda *_args: (str(tmp_path), str(target_path), str(dds_path)),
    )
    monkeypatch.setattr(
        repair.IMG,
        "_rebuild_from_parent",
        lambda *_args, **_kwargs: (
            1,
            Image.new("RGB", (1, 1), "blue"),
            18,
            "parent.jpg",
        ),
    )
    monkeypatch.setattr(repair, "_image_is_4096", lambda _path: True)
    monkeypatch.setattr(repair.IMG, "convert_texture", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        repair.IMG,
        "validate_dds_file",
        lambda *_args, **_kwargs: (True, None),
    )

    result = repair._repair_one(target, tile, backup_entries)

    assert result["status"] == "repaired"
    with Image.open(sibling_path) as sibling:
        assert sibling.format == "WEBP"
        assert sibling.size == (1, 1)


def test_repair_cli_apply_rolls_back_one_target_and_records_sha256(
    monkeypatch, tmp_path
):
    repair = _load_repair_cli()
    tile_dir = tmp_path / "zOrtho4XP_+35+139"
    tile_dir.mkdir()
    log = tile_dir / "Ortho4XP_build.log"
    log.write_text(
        "[Quality Check] Rebuilt 206720_465760_BI19.jpg from 25840_58208_BI16.jpg.\n",
        encoding="utf-8",
    )
    target_path = tmp_path / "target.jpg"
    dds_path = tmp_path / "target.dds"
    Image.new("RGB", (4096, 4096), "red").save(target_path, format="JPEG")
    dds_path.write_bytes(b"old-dds")
    old_target = target_path.read_bytes()
    old_dds = dds_path.read_bytes()
    tile = SimpleNamespace(
        dds_converter="nvcompress",
        dds_format="BC3",
        use_gpu_acceleration=False,
        use_gpu_for_color_filters=False,
        upscale_backend="none",
        upscale_scope="none",
        fp8_model_path="",
    )

    monkeypatch.setattr(repair, "_initialize_imagery", lambda: None)
    monkeypatch.setattr(repair, "_load_tile", lambda _target: tile)
    monkeypatch.setattr(
        repair,
        "_target_context",
        lambda _target, _tile: (str(tmp_path), str(target_path), str(dds_path)),
    )
    monkeypatch.setattr(repair, "_parent_candidates", lambda *_args: [])
    monkeypatch.setattr(
        repair.IMG,
        "_rebuild_from_parent",
        lambda *_args, **_kwargs: (
            1,
            Image.new("RGB", (4096, 4096), "blue"),
            18,
            "parent.jpg",
        ),
    )
    convert_kwargs = {}

    def fake_convert(*_args, **kwargs):
        convert_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(repair.IMG, "convert_texture", fake_convert)

    backup_dir = tmp_path / "backup"
    report_path = tmp_path / "report.json"
    result = repair.main(
        [
            "--log",
            str(log),
            "--apply",
            "--yes",
            "--backup-dir",
            str(backup_dir),
            "--report",
            str(report_path),
        ]
    )

    assert result == 1
    assert target_path.read_bytes() == old_target
    assert dds_path.read_bytes() == old_dds
    assert convert_kwargs["source_file"] == str(target_path)
    assert convert_kwargs["cleanup_preserved_inputs"] is True
    manifest = (backup_dir / "manifest.json").read_text(encoding="utf-8")
    assert "sha256" in manifest
    assert report_path.is_file()
