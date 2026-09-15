import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "tools" / name).read_text(encoding="utf-8"))


def test_controlnet_workflow_contains_both_sdxl_controlnets_and_chain():
    workflow = _load("comfyui_building_controlnet_api.json")

    assert workflow["2"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"
    assert workflow["7"]["inputs"]["control_net_name"] == (
        "controlnet-canny-sdxl-1.0-fp16.safetensors"
    )
    assert workflow["9"]["inputs"]["control_net_name"] == (
        "controlnet-depth-sdxl-1.0-fp16.safetensors"
    )
    assert workflow["12"]["inputs"]["positive"] == ["10", 0]
    assert workflow["13"]["inputs"]["positive"] == ["12", 0]
    assert workflow["6"]["inputs"]["positive"] == ["13", 0]


def test_controlnet_jobs_keep_facade_and_depth_guides_paired():
    jobs = _load("comfyui_building_controlnet_jobs.json")

    assert len(jobs) == 4
    for job in jobs:
        overrides = job["overrides"]
        facade = overrides["3.image"]
        depth = overrides["5.image"]
        assert facade.startswith("building_facade_")
        assert depth == facade.replace("building_facade_", "building_depth_")
        assert "2.ckpt_name" not in overrides
