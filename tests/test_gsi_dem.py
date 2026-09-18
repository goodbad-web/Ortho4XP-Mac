import json
import importlib
import shutil
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_GSI_DEM_Utils as GSI  # noqa: E402
import O4_DEM_Utils as DEM  # noqa: E402


def _gsi_xml(
    product,
    mesh_code="52326600",
    date="20250101",
    south=35.166666667,
    west=132.75,
    rows=2,
    cols=2,
    values=None,
):
    step = 0.04 / 3600.0 if product == "DEM1A" else 0.2 / 3600.0
    north = south + rows * step
    east = west + cols * step
    values = values or [1.0] * (rows * cols)
    tuples = "\n".join(f"地表面,{value}" for value in values)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Dataset xmlns:gml="http://www.opengis.net/gml/3.2">
  <gml:description>test</gml:description>
  <DEM>
    <type>{product}</type>
    <mesh>{mesh_code}</mesh>
    <lfSpanFr><gml:timePosition>{date[:4]}-{date[4:6]}-{date[6:]}</gml:timePosition></lfSpanFr>
    <coverage>
      <gml:boundedBy>
        <gml:Envelope srsName="fguuid:jgd2024.bl">
          <gml:lowerCorner>{south} {west}</gml:lowerCorner>
          <gml:upperCorner>{north} {east}</gml:upperCorner>
        </gml:Envelope>
      </gml:boundedBy>
      <gml:gridDomain><gml:Grid><gml:limits><gml:GridEnvelope>
        <gml:low>0 0</gml:low><gml:high>{cols - 1} {rows - 1}</gml:high>
      </gml:GridEnvelope></gml:limits></gml:Grid></gml:gridDomain>
      <gml:rangeSet><gml:DataBlock><gml:tupleList>{tuples}</gml:tupleList></gml:DataBlock></gml:rangeSet>
    </coverage>
  </DEM>
</Dataset>
""".encode("utf-8")


def _zip(path, product, values=None):
    xml_name = f"FG-GML-5232-66-00-{product}-20250101.xml"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(xml_name, _gsi_xml(product, values=values))


def test_import_classifies_deduplicates_and_quarantines(tmp_path):
    source = tmp_path / "downloads"
    input_dir = tmp_path / "Elevation_data" / "GSI" / "input"
    source.mkdir()
    valid = source / "FG-GML-523266-DEM5A-20250101.zip"
    _zip(valid, "DEM5A")
    shutil.copy2(valid, source / "same-content.zip")
    (source / "broken.zip").write_bytes(b"not a zip")

    result = GSI.import_gsi_archives(source, input_dir)

    assert result.imported == 1
    assert result.skipped_duplicates == 1
    assert result.quarantined == 1
    assert valid.read_bytes() == (source / "FG-GML-523266-DEM5A-20250101.zip").read_bytes()
    assert list((input_dir / "DEM5A" / "20250101").glob("*.zip"))
    assert list((input_dir / "_quarantine").glob("*.zip"))

    catalog = json.loads((input_dir / "catalog.json").read_text(encoding="utf-8"))
    statuses = {entry["status"] for entry in catalog["entries"]}
    assert "ready" in statuses
    assert "quarantined" in statuses
    ready_entry = next(entry for entry in catalog["entries"] if entry["status"] == "ready")
    quarantine_entry = next(entry for entry in catalog["entries"] if entry["status"] == "quarantined")
    assert ready_entry["source_path"] == str(valid)
    assert "ZIP" in quarantine_entry["reason"] or "zip" in quarantine_entry["reason"]


def test_import_quarantines_archive_with_malformed_xml(tmp_path):
    source = tmp_path / "downloads"
    input_dir = tmp_path / "input"
    source.mkdir()
    archive = source / "FG-GML-523266-DEM5A-20250101.zip"
    _zip(archive, "DEM5A")
    with zipfile.ZipFile(archive, "a") as managed_archive:
        managed_archive.writestr("malformed.xml", b"<broken")

    result = GSI.import_gsi_archives(source, input_dir)

    assert result.imported == 0
    assert result.quarantined == 1
    catalog = json.loads((input_dir / "catalog.json").read_text(encoding="utf-8"))
    assert [entry["status"] for entry in catalog["entries"]] == ["quarantined"]


def test_import_and_build_ignore_macos_appledouble_xml(tmp_path):
    source = tmp_path / "downloads"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source.mkdir()
    archive = source / "FG-GML-523266-DEM5A-20250101.zip"
    _zip(archive, "DEM5A")
    with zipfile.ZipFile(archive, "a") as managed_archive:
        managed_archive.writestr(
            "__MACOSX/FG-GML-523266-DEM5A-20250101/._data.xml",
            b"AppleDouble metadata, not GSI XML",
        )

    imported = GSI.import_gsi_archives(source, input_dir)

    assert imported.imported == 1
    assert imported.quarantined == 0
    bbox = (
        35.166666667,
        132.75,
        35.166666667 + 2 * 0.2 / 3600.0,
        132.75 + 2 * 0.2 / 3600.0,
    )
    built = GSI.build_gsi_dem(
        GSI.GSIOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            bbox=bbox,
            resolution="auto",
        )
    )

    assert built.outputs
    assert not built.failures


def test_build_one_meter_output_uses_lower_resolution_only_for_nodata(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _zip(input_dir / "FG-GML-523266-DEM10B-20240101.zip", "DEM10B", [1, 1, 1, 1])
    _zip(input_dir / "FG-GML-523266-DEM1A-20250101.zip", "DEM1A", [10, np.nan, np.nan, 40])

    region_bbox = (35.166666667, 132.75, 35.166666667 + 2 * 0.04 / 3600.0, 132.75 + 2 * 0.04 / 3600.0)
    result = GSI.build_gsi_dem(
        GSI.GSIOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            bbox=region_bbox,
            resolution="auto",
            make_vrt=True,
        )
    )

    assert result.outputs
    assert result.vrt
    assert result.manifest
    assert not result.failures

    from osgeo import gdal

    dataset = gdal.Open(result.outputs[0])
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray()
    scale = band.GetScale() or 1.0
    offset = band.GetOffset() or 0.0
    values = values.astype(np.float32) * scale + offset
    dataset = None
    assert values.shape == (2, 2)
    assert np.allclose(values, [[10, 1], [1, 40]])

    manifest = json.loads(Path(result.manifest).read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert manifest["options"]["storage_format"] == "compact_int16"
    assert manifest["results"][0]["resolution"] == "1m"
    assert manifest["results"][0]["crs"] == "EPSG:4326"
    assert manifest["results"][0]["data_type"] == "Int16"
    assert manifest["results"][0]["scale"] == 0.25
    assert manifest["results"][0]["offset"] == 0.0
    assert manifest["results"][0]["nodata"] == -32768
    assert manifest["results"][0]["compression"] in ("ZSTD", "DEFLATE")
    assert manifest["results"][0]["quantization_max_error"] == 0.125
    assert manifest["vrt_contract"] == {
        "data_type": "Int16",
        "scale": 0.25,
        "offset": 0.0,
        "nodata": -32768.0,
    }


def _write_test_geotiff(path, region, values, storage_format="compact_int16"):
    return GSI.write_geotiff(
        path,
        np.asarray(values, dtype=np.float32),
        region,
        0.0001,
        storage_format=storage_format,
    )


def test_compact_geotiff_quantization_nodata_and_reader_roundtrip(tmp_path):
    from osgeo import gdal

    path = tmp_path / "compact.tif"
    region = GSI.GSIRegion("N34E132", 34.0, 132.0, 34.0002, 132.0002)
    original = np.asarray([[10.12, 20.125], [np.nan, -3.14]], dtype=np.float32)

    metadata = _write_test_geotiff(path, region, original)
    dataset = gdal.Open(str(path))
    band = dataset.GetRasterBand(1)
    raw = band.ReadAsArray()
    reconstructed = raw.astype(np.float32) * (band.GetScale() or 1.0) + (
        band.GetOffset() or 0.0
    )

    assert metadata["data_type"] == "Int16"
    assert metadata["scale"] == 0.25
    assert metadata["offset"] == 0.0
    assert metadata["nodata"] == -32768
    assert metadata["predictor"] == 2
    assert metadata["compression"] == "ZSTD"
    assert gdal.GetDataTypeName(band.DataType) == "Int16"
    assert band.GetScale() == pytest.approx(0.25)
    assert band.GetOffset() == pytest.approx(0.0)
    assert band.GetNoDataValue() == pytest.approx(-32768)
    assert raw[1, 0] == -32768
    valid = np.isfinite(original)
    assert np.max(np.abs(reconstructed[valid] - original[valid])) <= 0.125
    dataset = None

    read_result = DEM.read_elevation_from_file(str(path), 34.0, 132.0)
    assert read_result[5] == -32768
    assert np.allclose(read_result[8][valid], reconstructed[valid])
    assert read_result[8][1, 0] == -32768


def test_float32_legacy_geotiff_remains_readable(tmp_path):
    from osgeo import gdal

    path = tmp_path / "legacy.tif"
    region = GSI.GSIRegion("N34E132", 34.0, 132.0, 34.0002, 132.0002)
    original = np.asarray([[10.12, 20.125], [np.nan, -3.14]], dtype=np.float32)

    metadata = _write_test_geotiff(path, region, original, "float32_legacy")
    dataset = gdal.Open(str(path))
    band = dataset.GetRasterBand(1)
    assert metadata["data_type"] == "Float32"
    assert gdal.GetDataTypeName(band.DataType) == "Float32"
    assert band.GetScale() is None
    assert band.GetOffset() is None
    assert band.GetNoDataValue() == pytest.approx(-9999)
    raw = band.ReadAsArray()
    assert raw[0, 0] == pytest.approx(original[0, 0])
    assert raw[1, 0] == pytest.approx(-9999)
    dataset = None

    read_result = DEM.read_elevation_from_file(str(path), 34.0, 132.0)
    assert np.allclose(read_result[8][0, 0], original[0, 0])
    assert read_result[8][1, 0] == -32768


def test_compact_range_failure_is_explicit_and_does_not_clip(tmp_path):
    path = tmp_path / "out-of-range.tif"
    region = GSI.GSIRegion("N34E132", 34.0, 132.0, 34.0001, 132.0001)

    with pytest.raises(GSI.GSIError, match="compact_int16 range"):
        _write_test_geotiff(path, region, [[8192.0]])
    assert not path.exists()


def test_vrt_preserves_common_scale_offset_contract_and_rejects_mixed_input(tmp_path):
    region_a = GSI.GSIRegion("a", 35.0, 132.0, 35.0002, 132.0002)
    region_b = GSI.GSIRegion("b", 35.0, 132.0002, 35.0002, 132.0004)
    compact_a = tmp_path / "a.tif"
    compact_b = tmp_path / "b.tif"
    legacy = tmp_path / "legacy.tif"
    _write_test_geotiff(compact_a, region_a, [[1, 2], [3, 4]])
    _write_test_geotiff(compact_b, region_b, [[5, 6], [7, 8]])
    _write_test_geotiff(legacy, region_b, [[5, 6], [7, 8]], "float32_legacy")

    contract = GSI._build_vrt(tmp_path / "compact.vrt", [compact_a, compact_b], False)
    assert contract == {
        "data_type": "Int16",
        "scale": 0.25,
        "offset": 0.0,
        "nodata": -32768.0,
    }
    read_result = DEM.read_elevation_from_file(
        str(tmp_path / "compact.vrt"), 35.0, 132.0
    )
    assert read_result[8][0, 0] == pytest.approx(1.0)
    assert read_result[8][0, 3] == pytest.approx(6.0)

    with pytest.raises(GSI.GSIError, match="different data type"):
        GSI._build_vrt(tmp_path / "mixed.vrt", [compact_a, legacy], False)


def test_compact_writer_falls_back_to_deflate_with_reason(tmp_path, monkeypatch):
    from osgeo import gdal

    real_driver = gdal.GetDriverByName("GTiff")

    class DriverProxy:
        def Create(self, *args, **kwargs):
            options = list(kwargs.get("options", []))
            kwargs["options"] = [
                "COMPRESS=DEFLATE" if option == "COMPRESS=ZSTD" else option
                for option in options
            ]
            return real_driver.Create(*args, **kwargs)

    monkeypatch.setattr(GSI.gdal, "GetDriverByName", lambda name: DriverProxy())
    path = tmp_path / "fallback.tif"
    region = GSI.GSIRegion("N34E132", 34.0, 132.0, 34.0001, 132.0001)
    metadata = _write_test_geotiff(path, region, [[1.0]])

    assert metadata["compression"] == "DEFLATE"
    assert metadata["compression_fallback_reason"]


def test_cli_config_and_gui_storage_format_contract():
    make_gsi_dem = importlib.import_module("make_gsi_dem")
    args = make_gsi_dem.build_parser().parse_args(
        ["build", "--bbox", "35", "132", "36", "133", "--storage-format", "float32_legacy"]
    )
    assert args.storage_format == "float32_legacy"
    default_args = make_gsi_dem.build_parser().parse_args(
        ["build", "--bbox", "35", "132", "36", "133"]
    )
    assert default_args.storage_format == "compact_int16"

    import O4_Config_Utils as CFG
    import O4_GUI_Utils as GUI

    assert CFG.cfg_vars["gsi_dem_storage_format"]["default"] == "compact_int16"
    assert "gsi_dem_storage_format" in CFG.list_other_vars

    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class EmptySelection:
        def curselection(self):
            return ()

        def get(self, index):
            del index
            raise AssertionError("selection should be empty")

    dialog = GUI.Ortho4XP_GSI_DEM.__new__(GUI.Ortho4XP_GSI_DEM)
    dialog.mesh_list = EmptySelection()
    dialog.bbox_vars = [Value("") for _ in range(4)]
    dialog.hgt_tiles = Value("")
    dialog.storage_format_labels = {
        "compact_int16": "Compact Int16 (0.25m)",
        "float32_legacy": "Float32 legacy",
    }
    dialog.storage_format = Value("Float32 legacy")
    dialog.input_dir = Value(str(Path("input")))
    dialog.output_dir = Value(str(Path("output")))
    dialog.resolution = Value("auto")
    dialog.make_vrt = Value(True)
    dialog.overwrite = Value(False)
    options = dialog._build_options()
    assert options.storage_format == "float32_legacy"


def test_scan_reports_duplicate_and_invalid_managed_archives(tmp_path):
    input_dir = tmp_path / "input"
    ready_dir = input_dir / "DEM5A" / "20250101"
    ready_dir.mkdir(parents=True)
    archive = ready_dir / "FG-GML-523266-DEM5A-20250101.zip"
    _zip(archive, "DEM5A")
    shutil.copy2(archive, ready_dir / "duplicate.zip")
    (ready_dir / "invalid.zip").write_bytes(b"invalid")

    result = GSI.scan_gsi_input(input_dir)

    assert result.ready == 1
    assert result.duplicates == 1
    assert result.invalid == 1
    catalog = json.loads((input_dir / "catalog.json").read_text(encoding="utf-8"))
    assert len(catalog["entries"]) == 3


def test_scan_and_build_detect_catalog_file_changes_and_missing_inputs(tmp_path):
    input_dir = tmp_path / "input"
    ready_dir = input_dir / "DEM5A" / "20250101"
    ready_dir.mkdir(parents=True)
    archive = ready_dir / "FG-GML-523266-DEM5A-20250101.zip"
    _zip(archive, "DEM5A")

    initial = GSI.scan_gsi_input(input_dir)
    assert initial.ready == 1
    archive.write_bytes(archive.read_bytes() + b"changed")

    modified = GSI.scan_gsi_input(input_dir)
    assert modified.modified == 1
    assert modified.entries[0]["status"] == "modified"
    persisted = GSI.scan_gsi_input(input_dir)
    assert persisted.modified == 1
    assert persisted.entries[0]["status"] == "modified"
    archive.unlink()

    missing = GSI.scan_gsi_input(input_dir)
    assert missing.missing == 1
    assert missing.entries[0]["status"] == "missing"


def test_import_restores_catalog_missing_archive(tmp_path):
    source = tmp_path / "downloads"
    input_dir = tmp_path / "input"
    source.mkdir()
    source_archive = source / "FG-GML-523266-DEM5A-20250101.zip"
    _zip(source_archive, "DEM5A")

    GSI.import_gsi_archives(source, input_dir)
    managed_archive = next((input_dir / "DEM5A" / "20250101").glob("*.zip"))
    managed_archive.unlink()
    assert GSI.scan_gsi_input(input_dir).missing == 1

    restored = GSI.import_gsi_archives(source, input_dir)

    assert restored.imported == 1
    assert restored.skipped_duplicates == 0
    assert managed_archive.is_file()
    assert GSI.scan_gsi_input(input_dir).missing == 0

    managed_archive.write_bytes(b"changed")
    assert GSI.scan_gsi_input(input_dir).modified == 1
    managed_archive.unlink()
    assert GSI.scan_gsi_input(input_dir).missing == 1
    restored_after_change = GSI.import_gsi_archives(source, input_dir)
    final_scan = GSI.scan_gsi_input(input_dir)

    assert restored_after_change.imported == 1
    assert final_scan.ready == 1
    assert final_scan.modified == 0
    assert final_scan.missing == 0


def test_candidate_filter_uses_catalog_mesh_bounds_for_bbox_and_hgt(tmp_path):
    del tmp_path
    region = GSI.GSIRegion("N35E139", 35.0, 139.0, 36.0, 140.0)
    scan = GSI.GSIScanResult(
        entries=[
            {"status": "ready", "path": "near.zip", "mesh_codes": ["533900"]},
            {"status": "ready", "path": "far.zip", "mesh_codes": ["523266"]},
        ],
        ready=2,
        duplicates=0,
        quarantined=0,
        invalid=0,
    )

    candidates = GSI._candidate_entries(scan, region)

    assert [entry["path"] for entry in candidates] == ["near.zip"]


def test_write_hgt_keeps_big_endian_nodata(tmp_path):
    path = tmp_path / "N34E132.hgt"
    GSI.write_hgt(path, np.asarray([[1.4, np.nan], [-2.6, 32768.0]], dtype=np.float32))

    assert path.read_bytes() == b"\x00\x01\x80\x00\xff\xfd\x7f\xff"


def test_mesh_code_bounds_match_gsi_third_mesh_geometry():
    south, west, north, east = GSI.mesh_code_bounds("52326600")

    assert abs(south - 35.1666666667) < 1e-9
    assert abs(west - 132.75) < 1e-9
    assert abs((north - south) - 30.0 / 3600.0) < 1e-12
    assert abs((east - west) - 45.0 / 3600.0) < 1e-12


def test_gsi_dialog_cancel_is_cooperative():
    import O4_GUI_Utils as GUI

    class Status:
        def __init__(self):
            self.values = []

        def set(self, value):
            self.values.append(value)

    dialog = GUI.Ortho4XP_GSI_DEM.__new__(GUI.Ortho4XP_GSI_DEM)
    dialog.cancel_event = threading.Event()
    dialog.status_var = Status()

    dialog.request_cancel()

    assert dialog.cancel_event.is_set()
    assert dialog.status_var.values
