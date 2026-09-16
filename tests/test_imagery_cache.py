import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
from PIL import Image


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Config_Utils as CFG  # noqa: E402
import O4_File_Names as FNAMES  # noqa: E402
import O4_Imagery_Cache as CACHE  # noqa: E402
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Tile_Utils as TILE  # noqa: E402


def _cache_fixture(tmp_path, image_array=None):
    tile_dir = tmp_path / "+20+120" / "+34+132" / "BI_16"
    tile_dir.mkdir(parents=True)
    if image_array is None:
        image_array = numpy.full((512, 512, 3), [120, 180, 220], dtype=numpy.uint8)
    jpeg_path = tile_dir / "0_0_BI16.jpg"
    Image.fromarray(image_array).save(jpeg_path, format="JPEG", quality=100)
    return tile_dir, jpeg_path


def test_default_settings_remain_jpeg_and_webp_quality_is_explicit():
    assert CFG.cfg_vars["imagery_cache_format"]["default"] == "jpg"
    assert CFG.cfg_vars["imagery_cache_quality"]["default"] == ""
    assert CACHE.validate_cache_settings("jpg", "") == ("jpg", None)
    with pytest.raises(ValueError):
        CACHE.validate_cache_settings("webp", "")
    assert CACHE.validate_cache_settings("webp", "95") == ("webp", 95)


def test_invalid_webp_settings_stop_before_tile_build(monkeypatch):
    monkeypatch.setattr(IMG, "imagery_cache_format", "webp")
    monkeypatch.setattr(IMG, "imagery_cache_quality", "")
    assert TILE.build_tile(SimpleNamespace()) == 0


def test_webp_priority_jpeg_fallback_and_corrupt_webp(tmp_path):
    directory, jpeg_path = _cache_fixture(tmp_path)
    webp_path = jpeg_path.with_suffix(".webp")
    assert CACHE.find_cache_path(directory, 0, 0, 16, "BI") == str(jpeg_path)

    Image.open(jpeg_path).save(webp_path, format="WEBP", quality=95)
    assert CACHE.find_cache_path(directory, 0, 0, 16, "BI") == str(webp_path)

    webp_path.write_bytes(b"broken")
    assert CACHE.find_cache_path(directory, 0, 0, 16, "BI") == str(jpeg_path)
    assert not webp_path.exists()


def test_migration_scope_filters_provider_and_zoomlevel(tmp_path):
    _directory, jpeg_path = _cache_fixture(tmp_path)
    assert CACHE.iter_jpeg_cache_files(
        tmp_path, tiles=["+34+132"], providers=["BI"], zoomlevels=[16]
    ) == [jpeg_path]
    assert CACHE.iter_jpeg_cache_files(
        tmp_path, tiles=["+34+132"], providers=["OTHER"]
    ) == []
    assert CACHE.iter_jpeg_cache_files(
        tmp_path, tiles=["+34+132"], zoomlevels=[15]
    ) == []


def test_migration_gates_atomic_result_and_rerun_skip(tmp_path):
    directory, jpeg_path = _cache_fixture(tmp_path)
    report_path = tmp_path / "report.json"
    report = CACHE.migrate_cache(
        mode="convert",
        root=tmp_path,
        tiles=["+34+132"],
        quality=95,
        workers=2,
        report_path=report_path,
    )
    assert report["files_considered"] == 1
    assert report["files"][0]["status"] == "converted"
    assert report["files"][0]["psnr_db"] is not None
    assert report["files"][0]["mae_8bit"] is not None
    webp_path = jpeg_path.with_suffix(".webp")
    assert webp_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["mode"] == "convert"

    rerun = CACHE.migrate_cache(
        mode="convert",
        root=tmp_path,
        tiles=["+34+132"],
        quality=95,
        report_path=tmp_path / "rerun.json",
    )
    assert rerun["files"][0]["status"] == "skipped_existing"

    dry_directory, dry_jpeg = _cache_fixture(tmp_path / "dry-run")
    dry_report = CACHE.migrate_cache(
        mode="convert",
        root=tmp_path / "dry-run",
        tiles=["+34+132"],
        quality=95,
        force=True,
        dry_run=True,
        report_path=tmp_path / "dry-run-report.json",
    )
    assert dry_report["files"][0]["status"] == "would_convert"
    assert not dry_jpeg.with_suffix(".webp").exists()


def test_quality_or_size_rejection_keeps_jpeg(tmp_path):
    noisy = numpy.random.default_rng(7).integers(
        0, 256, (512, 512, 3), dtype=numpy.uint8
    )
    _, jpeg_path = _cache_fixture(tmp_path, noisy)
    result = CACHE._convert_one(jpeg_path, quality=80, force=True, dry_run=False)
    assert result["status"] == "rejected"
    assert jpeg_path.exists()
    assert not jpeg_path.with_suffix(".webp").exists()


def test_atomic_write_failure_does_not_remove_jpeg(tmp_path, monkeypatch):
    _, jpeg_path = _cache_fixture(tmp_path)

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(CACHE.os, "replace", fail_replace)
    result = CACHE._convert_one(jpeg_path, quality=95, force=True, dry_run=False)
    assert result["status"] == "error"
    assert jpeg_path.exists()
    assert not jpeg_path.with_suffix(".webp").exists()


def test_cleanup_requires_apply_and_revalidation(tmp_path):
    _, jpeg_path = _cache_fixture(tmp_path)
    webp_path = jpeg_path.with_suffix(".webp")
    Image.open(jpeg_path).save(webp_path, format="WEBP", quality=95)

    preview = CACHE.migrate_cache(
        mode="cleanup",
        root=tmp_path,
        tiles=["+34+132"],
        workers=1,
        report_path=tmp_path / "cleanup-preview.json",
    )
    assert preview["files"][0]["status"] == "would_cleanup"
    assert jpeg_path.exists()

    applied = CACHE.migrate_cache(
        mode="cleanup",
        root=tmp_path,
        tiles=["+34+132"],
        workers=1,
        apply=True,
        confirmed=True,
        report_path=tmp_path / "cleanup-apply.json",
    )
    assert applied["files"][0]["status"] == "cleaned"
    assert not jpeg_path.exists()
    assert webp_path.exists()


def test_webp_is_normalized_to_temporary_png(tmp_path):
    _, jpeg_path = _cache_fixture(tmp_path)
    webp_path = jpeg_path.with_suffix(".webp")
    Image.open(jpeg_path).save(webp_path, format="WEBP", quality=95)
    output, created = CACHE.prepare_external_image_input(str(webp_path), str(tmp_path))
    try:
        assert created
        assert output.endswith(".png")
        with Image.open(output) as image:
            assert image.size == (512, 512)
    finally:
        Path(output).unlink(missing_ok=True)


def test_convert_texture_passes_png_to_dds_converter_for_webp(
    tmp_path, monkeypatch
):
    orthophotos = tmp_path / "Orthophotos"
    directory = orthophotos / "+30+130" / "+34+132" / "BI_16"
    directory.mkdir(parents=True)
    image = Image.new("RGB", (512, 512), (120, 180, 220))
    webp_path = directory / "0_0_BI16.webp"
    image.save(webp_path, format="WEBP", quality=95)

    tile = SimpleNamespace(
        build_dir=str(tmp_path / "tile"),
        imprint_masks_to_dds=False,
        mask_zl=14,
        lat=34,
        lon=132,
        upscale_backend="none",
        upscale_scope="none",
        airport_highres_texture_keys=(),
    )
    monkeypatch.setattr(FNAMES, "Imagery_dir", str(orthophotos))
    monkeypatch.setattr(IMG, "providers_dict", {
        "BI": {"imagery_dir": "grouped", "code": "BI", "color_filters": "none"}
    })
    monkeypatch.setattr(IMG, "local_combined_providers_dict", {})
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path))
    monkeypatch.setattr(IMG.UI, "dds_converter", "nvcompress")
    monkeypatch.setattr(IMG.UI, "dds_format", "BC1")
    monkeypatch.setattr(IMG, "dds_convert_cmd", "fake-nvcompress")

    def write_dxt1(path):
        width = height = 512
        mipmaps = max(width, height).bit_length()
        header = bytearray(128)
        header[:4] = b"DDS "
        struct.pack_into("<I", header, 4, 124)
        struct.pack_into("<I", header, 12, height)
        struct.pack_into("<I", header, 16, width)
        struct.pack_into("<I", header, 28, mipmaps)
        struct.pack_into("<I", header, 76, 32)
        header[84:88] = b"DXT1"
        payload = 0
        level_width = width
        level_height = height
        for _ in range(mipmaps):
            payload += max(1, (level_width + 3) // 4) * max(
                1, (level_height + 3) // 4
            ) * 8
            level_width = max(1, level_width // 2)
            level_height = max(1, level_height // 2)
        Path(path).write_bytes(bytes(header) + bytes(payload))

    converter_inputs = []

    def fake_call(command, **_kwargs):
        converter_inputs.append(command[-2])
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        write_dxt1(command[-1])
        return 0

    monkeypatch.setattr(IMG.subprocess, "call", fake_call)
    assert IMG.convert_texture(tile, 0, 0, 16, "BI") == 1
    assert converter_inputs
    assert converter_inputs[0].endswith(".png")
    assert not converter_inputs[0].endswith(".webp")
    assert Path(tile.build_dir, "textures", "0_0_BI16.dds").exists()
    assert not list((tmp_path / "tmp").glob("ortho4xp-webp-*.png"))


def test_imagery_filename_wrappers_keep_legacy_contract():
    assert FNAMES.jpeg_file_name_from_attributes(0, 0, 16, "BI").endswith(
        ".jpg"
    )
    names = FNAMES.imagery_file_names_from_attributes(0, 0, 16, "BI")
    assert [Path(name).suffix for name in names] == [".webp", ".jpg"]
    assert FNAMES.jpeg_file_dir_from_attributes(
        34, 132, 16, {"imagery_dir": "grouped", "code": "BI"}
    ) == FNAMES.imagery_file_dir_from_attributes(
        34, 132, 16, {"imagery_dir": "grouped", "code": "BI"}
    )
