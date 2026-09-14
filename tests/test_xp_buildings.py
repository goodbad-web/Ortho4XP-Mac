import importlib.util
import json
import math
import sys
from pathlib import Path
import pytest
from shapely.geometry import Point


MODULE_PATH = Path(__file__).parents[1] / "tools" / "xp_buildings.py"
SPEC = importlib.util.spec_from_file_location("xp_buildings", MODULE_PATH)
xp = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = xp
SPEC.loader.exec_module(xp)


def _base_asset_root(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    for relative_path in xp.DEFAULT_ASSETS.values():
        (root / Path(relative_path).name).write_text("A\n800\nOBJ\n", encoding="utf-8")
    return root


def test_height_policy_prefers_height_then_levels_then_category():
    assert xp.height_from_tags({"height": "10 m", "building:levels": "5"}) == 10
    assert xp.height_from_tags({"building:levels": "3"}) == pytest.approx(9.4)
    assert xp.height_from_tags({"building": "industrial"}) == 7.0


def test_load_geojson_and_write_package(tmp_path):
    source = tmp_path / "buildings.geojson"
    source.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature", "id": "house-1",
            "properties": {"building": "house", "building:levels": 2},
            "geometry": {"type": "Polygon", "coordinates": [[
                [133.1, 34.1], [133.1002, 34.1], [133.1002, 34.1002],
                [133.1, 34.1002], [133.1, 34.1]
            ]]}
        }]
    }), encoding="utf-8")
    buildings = xp.load_geojson(source, 34, 133)
    assert len(buildings) == 1
    assert buildings[0].height_m == 6.6
    output = tmp_path / "package"
    xp.build_package(
        output, 34, 133, buildings, dict(xp.DEFAULT_ASSETS), "replace", None, None,
        _base_asset_root(tmp_path),
    )
    library = (output / "library.txt").read_text(encoding="utf-8")
    dsf = next((output / "Earth nav data").rglob("*.txt")).read_text(encoding="utf-8")
    assert "REGION_RECT 133 34 133 34" in library
    assert "EXPORT simheaven/houses/house_15x20x2.obj" in library
    assert "OBJECT 0" in dsf
    report = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))
    assert report["building_count"] == 1


def test_exclusion_targets_objects_and_facades(tmp_path):
    output = tmp_path / "package"
    xp.build_package(
        output,
        34,
        133,
        [],
        dict(xp.DEFAULT_ASSETS),
        "replace",
        None,
        (133.935, 34.620, 133.937, 34.622),
        _base_asset_root(tmp_path),
    )
    dsf = next((output / "Earth nav data").rglob("*.txt")).read_text(encoding="utf-8")
    assert "PROPERTY sim/exclude_obj 133.935/34.62/133.937/34.622" in dsf
    assert "PROPERTY sim/exclude_fac 133.935/34.62/133.937/34.622" in dsf


def test_asset_variant_tracks_building_size_and_height():
    polygon = xp.Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
    building = xp.Building(
        polygon=polygon,
        lon=133.5,
        lat=34.5,
        width_m=30,
        depth_m=20,
        heading=0,
        height_m=18,
        category="house",
        source_id="test",
    )
    assert xp.asset_for_building(building, xp.DEFAULT_ASSETS).endswith("jp_house_large_high.obj")


def test_cli_accepts_split_overpass_json_and_deduplicates_way_ids(tmp_path, monkeypatch):
    element = {
        "type": "way",
        "id": 42,
        "tags": {"building": "house"},
        "geometry": [
            {"lon": 133.1, "lat": 34.1},
            {"lon": 133.1002, "lat": 34.1},
            {"lon": 133.1002, "lat": 34.1002},
            {"lon": 133.1, "lat": 34.1002},
            {"lon": 133.1, "lat": 34.1},
        ],
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path in (first, second):
        path.write_text(json.dumps({"elements": [element]}), encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    for relative_path in xp.DEFAULT_ASSETS.values():
        (asset_root / Path(relative_path).name).write_text("I 800 OBJ", encoding="utf-8")
    output = tmp_path / "package"
    monkeypatch.setattr(sys, "argv", [
        "xp_buildings.py",
        "--lat", "34", "--lon", "133",
        "--overpass-json", str(first),
        "--overpass-json", str(second),
        "--asset-root", str(asset_root),
        "--output", str(output),
    ])
    assert xp.main() == 0
    report = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))
    assert report["building_count"] == 1


def test_heading_aligns_asset_width_with_long_footprint_edge():
    building = xp._building_from_polygon(
        xp.Polygon([(0, 0), (30, 0), (30, 10), (0, 10)]),
        {"building": "house"},
        "east-west",
        34,
        133,
    )
    assert building is not None
    assert building.heading % 180 == pytest.approx(0)


def test_concave_building_anchor_is_inside_footprint():
    polygon = xp.Polygon([
        (0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10),
    ])
    building = xp._building_from_polygon(polygon, {"building": "house"}, "concave", 34, 133)
    assert building is not None
    scale_x = 111320.0 * math.cos(math.radians(34.5))
    scale_y = 110540.0
    anchor = Point((building.lon - 133) * scale_x, (building.lat - 34) * scale_y)
    assert polygon.covers(anchor)


def test_build_package_requires_self_owned_assets(tmp_path):
    with pytest.raises(ValueError, match="asset-root is required"):
        xp.build_package(
            tmp_path / "package",
            34,
            133,
            [],
            dict(xp.DEFAULT_ASSETS),
            "replace",
            None,
            None,
            None,
        )
