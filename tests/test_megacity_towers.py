import importlib.util
import json
from pathlib import Path
import sys


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
mega = _load("create_megacity_towers", ROOT / "tools" / "create_megacity_towers.py")


def test_tower_near_mesh_reaches_public_height_without_interior_geometry():
    mesh = mega.build_tower_mesh("east", "near")

    assert max(vertex[1] for vertex in mesh.vertices) == mega.TOWER_HEIGHT_M
    assert min(vertex[1] for vertex in mesh.vertices) == mega.PODIUM_HEIGHT_M - 0.5
    assert len(mesh.faces) > 200


def test_obj8_lod_sections_have_disjoint_draw_ranges():
    sections = [
        mega.LODSection(0, 500, mega.build_tower_mesh("west", "near")),
        mega.LODSection(500, 2500, mega.build_tower_mesh("west", "mid")),
        mega.LODSection(2500, 15000, mega.build_tower_mesh("west", "far")),
    ]
    text = mega.obj8_lod_text(sections, "mega_city_towers_facade.dds")

    assert text.count("ATTR_LOD ") == 3
    assert text.count("TRIS ") == 3
    assert "TEXTURE mega_city_towers_facade.dds" in text


def test_package_writes_twin_tower_dsf_and_report(tmp_path):
    config = ROOT / "tools" / "megacity_towers_config.json"
    output = tmp_path / "zz_MegaCityTowers_+34+135"

    report = mega.generate_package(output, config)
    dsf_text = output / "Earth nav data" / "+30+130" / "+34+135.txt"
    dsf = dsf_text.read_text(encoding="utf-8")
    saved_report = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))

    assert report["tile"] == {"lat": 34, "lon": 135}
    assert "OBJECT_DEF objects/mega_city_towers_west.obj" in dsf
    assert "OBJECT_DEF objects/mega_city_towers_east.obj" in dsf
    assert dsf.count("OBJECT ") == 3
    assert saved_report["texture_source"].startswith("self-owned")
    assert (output / "objects" / "mega_city_towers_east.obj").is_file()
