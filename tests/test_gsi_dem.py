import json
import shutil
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_GSI_DEM_Utils as GSI  # noqa: E402


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
    values = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    assert values.shape == (2, 2)
    assert np.allclose(values, [[10, 1], [1, 40]])

    manifest = json.loads(Path(result.manifest).read_text(encoding="utf-8"))
    assert manifest["results"][0]["resolution"] == "1m"
    assert manifest["results"][0]["crs"] == "EPSG:4326"


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
    archive.unlink()

    missing = GSI.scan_gsi_input(input_dir)
    assert missing.missing == 1
    assert missing.entries[0]["status"] == "missing"


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
