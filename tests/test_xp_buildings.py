import importlib.util
import json
import sys
from pathlib import Path
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "xp_buildings.py"
SPEC = importlib.util.spec_from_file_location("xp_buildings", MODULE_PATH)
xp = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = xp
SPEC.loader.exec_module(xp)


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
    xp.build_package(output, 34, 133, buildings, dict(xp.DEFAULT_ASSETS), "replace", None, None, None)
    library = (output / "library.txt").read_text(encoding="utf-8")
    dsf = next((output / "Earth nav data").rglob("*.txt")).read_text(encoding="utf-8")
    assert "REGION_RECT 133 34 133 34" in library
    assert "EXPORT simheaven/houses/house_15x20x2.obj" in library
    assert "OBJECT 0" in dsf
    report = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))
    assert report["building_count"] == 1
