import importlib.util
import json
from pathlib import Path

import pytest


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[1]
blender_assets = _load("blender_generate_assets", ROOT / "tools" / "blender_generate_assets.py")
comfy_batch = _load("comfyui_texture_batch", ROOT / "tools" / "comfyui_texture_batch.py")
controlnet_guides = _load("render_controlnet_guides", ROOT / "tools" / "render_controlnet_guides.py")


def test_blender_asset_mesh_has_roof_and_valid_obj8():
    mesh = blender_assets.build_mesh("house", 16.0, 6.0, 10.0, "low")
    text = blender_assets.obj8_text(mesh, "jp_house.png")

    assert len(mesh.faces) == 9
    assert "A\n800\nOBJ" in text
    assert "TEXTURE jp_house.png" in text
    assert "POINT_COUNTS " in text
    assert "TRIS 0 " in text


def test_box_uses_face_specific_uv_regions_for_facade_texture():
    mesh = blender_assets.MeshData()
    blender_assets._box(mesh, 10.0, 6.0, 8.0)

    assert mesh.face_uvs[1] == blender_assets._LEFT_SIDE_UV
    assert mesh.face_uvs[2] == blender_assets._RIGHT_SIDE_UV
    assert mesh.face_uvs[3] == blender_assets._FRONT_UV
    assert mesh.face_uvs[4] == blender_assets._BACK_UV
    assert mesh.face_uvs[5] == blender_assets._ROOF_UV
    assert mesh.face_uvs[3] != mesh.face_uvs[1]


def test_blender_variant_manifest_covers_categories_and_size_height_buckets():
    records = blender_assets._variant_records()

    assert len(records) == 40
    assert {record["category"] for record in records} == {
        "house", "apartments", "commercial", "industrial"
    }
    assert sum(record["size"] == "large" and record["height"] == "high" for record in records) == 4


def test_obj8_only_pack_copies_category_textures_to_all_variants(tmp_path):
    texture_root = tmp_path / "textures"
    texture_root.mkdir()
    for stem in ("house", "apartment", "commercial", "industrial"):
        (texture_root / f"jp_{stem}.png").write_bytes(b"texture")

    output = tmp_path / "assets"
    manifest = blender_assets.generate_pack(output, texture_root, False, None)

    assert len(manifest["assets"]) == 40
    assert (output / "jp_house_small_low.obj").is_file()
    assert (output / "jp_house_small_low.png").read_bytes() == b"texture"
    assert "TEXTURE jp_house_small_low.png" in (output / "jp_house_small_low.obj").read_text()


def test_comfy_job_changes_seed_prompt_and_filename_without_mutating_base():
    workflow = {
        "6": {"inputs": {"seed": 1}},
        "8": {"inputs": {"filename_prefix": "base"}},
        "3": {"inputs": {"text": "base"}},
    }
    job = {
        "name": "jp_house",
        "seed": 2401,
        "overrides": {"3.text": "house facade"},
    }

    result = comfy_batch.apply_job(workflow, job)

    assert result["6"]["inputs"]["seed"] == 2401
    assert result["8"]["inputs"]["filename_prefix"] == "jp_house"
    assert result["3"]["inputs"]["text"] == "house facade"
    assert workflow["6"]["inputs"]["seed"] == 1
    assert workflow["8"]["inputs"]["filename_prefix"] == "base"


def test_comfy_batch_stages_loadimage_files_into_input_directory(tmp_path):
    guide_dir = tmp_path / "guides"
    input_dir = tmp_path / "comfy-input"
    guide_dir.mkdir()
    guide = guide_dir / "building_facade_house.png"
    guide.write_bytes(b"guide")
    workflow = {
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": guide.name},
        }
    }

    staged = comfy_batch.stage_input_images([workflow], guide_dir, input_dir)

    assert staged == [input_dir / guide.name]
    assert staged[0].read_bytes() == b"guide"


def test_controlnet_depth_contract_uses_camera_axis_and_near_white_polarity():
    assert controlnet_guides.DEPTH_CAMERA_OUTPUT == "View Z Depth"
    assert controlnet_guides.DEPTH_NEAR_VALUE == 1.0
    assert controlnet_guides.DEPTH_FAR_VALUE == 0.0


def test_comfy_jobs_reject_path_traversal(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps([{"name": "../escape"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid job name"):
        comfy_batch.load_jobs(path)


def test_obj8_only_rejects_placeholder_without_blender(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(blender_assets, "bpy", None)

    with pytest.raises(SystemExit) as error:
        blender_assets.main(
            [
                "--output",
                str(tmp_path / "assets"),
                "--obj8-only",
                "--allow-placeholder",
            ]
        )

    assert error.value.code == 2
    assert "requires Blender" in capsys.readouterr().err
