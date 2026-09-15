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
johoku = _load("create_scene_johoku", ROOT / "tools" / "create_scene_johoku.py")


def test_astro_tower_has_public_height_and_elliptical_detail():
    mesh = johoku.build_astro_tower_mesh("near")

    assert max(vertex[1] for vertex in mesh.vertices) == 160.0
    assert len(mesh.faces) > 1_000
    assert len(mesh.vertices) < 20_000
    # The box core is intentionally emitted first as a robust fallback; the
    # first elliptical side follows its six faces.
    first_face = mesh.faces[12]
    normal = johoku._normal(*(mesh.vertices[index] for index in first_face[:3]))
    assert normal[0] > 0.0


def test_ellipse_caps_face_away_from_the_mesh():
    mesh = johoku.MeshData()
    johoku.add_ellipse_shell(mesh, 10.0, 6.0, 0.0, 4.0, 12)

    bottom = johoku._normal(*(mesh.vertices[index] for index in mesh.faces[-2][:3]))
    top = johoku._normal(*(mesh.vertices[index] for index in mesh.faces[-1][:3]))

    assert bottom[1] < 0.0
    assert top[1] > 0.0


def test_obj8_lod_sections_have_disjoint_draw_ranges():
    sections = [
        johoku.LODSection(0, 600, johoku.build_astro_tower_mesh("near")),
        johoku.LODSection(600, 3000, johoku.build_astro_tower_mesh("mid")),
        johoku.LODSection(3000, 18000, johoku.build_astro_tower_mesh("far")),
    ]
    text = johoku.obj8_lod_text(sections, "scene_johoku_facade.png")

    assert text.count("ATTR_LOD ") == 3
    assert text.count("ATTR_shadow") == 1
    assert "ATTR_no_cull" not in text
    assert text.count("TRIS ") == 3
    assert "TEXTURE scene_johoku_facade.png" in text
    commands = text[text.index("IDX "):]
    assert commands.index("ATTR_LOD 0.0 600.0") < commands.index("TRIS ")
    assert "ATTR_shadow" not in commands


def test_package_writes_three_buildings_and_report(tmp_path):
    config = ROOT / "tools" / "scene_johoku_config.json"
    output = tmp_path / "zz_TheSceneJohoku_+35+136"

    report = johoku.generate_package(output, config)
    dsf_text = output / "Earth nav data" / "+30+130" / "+35+136.txt"
    dsf = dsf_text.read_text(encoding="utf-8")
    saved_report = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))

    assert report["tile"] == {"lat": 35, "lon": 136}
    assert dsf.count("OBJECT ") == 4
    assert "OBJECT_DEF objects/scene_johoku_astro_tower.obj" in dsf
    assert "OBJECT_DEF objects/scene_johoku_west_star.obj" in dsf
    assert "OBJECT_DEF objects/scene_johoku_east_star.obj" in dsf
    assert "OBJECT_DEF objects/scene_johoku_podium.obj" in dsf
    assert saved_report["landmark_spec"]["astro_tower_height_m"] == 160.0
    assert (output / "objects" / "scene_johoku_astro_tower.obj").is_file()
    assert (output / "objects" / "scene_johoku_podium.obj").is_file()
