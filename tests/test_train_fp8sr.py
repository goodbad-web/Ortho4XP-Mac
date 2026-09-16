import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_fp8sr  # noqa: E402
from fp8sr_pack import EXPECTED_LAYERS, validate_pack, write_pack  # noqa: E402


def make_pair(root: Path, split: str = "val", size: tuple[int, int] = (8, 6)) -> None:
    lr_dir = root / split / "lr"
    hr_dir = root / split / "hr"
    lr_dir.mkdir(parents=True)
    hr_dir.mkdir(parents=True)
    lr = Image.new("RGB", size, (20, 40, 60))
    hr = lr.resize((size[0] * 2, size[1] * 2), Image.Resampling.NEAREST)
    lr.save(lr_dir / "tile.png")
    hr.save(hr_dir / "tile.png")


def test_list_pairs_requires_exact_two_x_matching(tmp_path):
    make_pair(tmp_path)
    pairs = train_fp8sr.list_pairs(tmp_path, "val")
    assert len(pairs) == 1
    assert pairs[0].lr.name == "tile.png"

    with Image.new("RGB", (15, 12), (0, 0, 0)) as wrong:
        wrong.save(tmp_path / "val" / "hr" / "tile.png")
    with pytest.raises(train_fp8sr.TrainingError, match="exactly 2x"):
        train_fp8sr.list_pairs(tmp_path, "val")


def test_fp16_and_fp8_quality_gates_have_fixed_boundaries():
    lanczos = {"psnr_db": 30.0, "mae": 2.0, "rmse": 3.0}
    assert train_fp8sr._gate_fp16_against_lanczos(lanczos, lanczos)["status"] == "PASS"
    assert train_fp8sr._gate_fp16_against_lanczos(
        {"psnr_db": 29.9, "mae": 2.0, "rmse": 3.0}, lanczos
    )["status"] == "FAIL"

    baseline = {"psnr_db": 30.0, "mae": 2.0, "rmse": 3.0}
    candidate = {"psnr_db": 29.75, "mae": 2.1, "rmse": 3.15}
    assert train_fp8sr._fp8_gate(candidate, baseline)["status"] == "PASS"
    candidate["psnr_db"] = 29.74
    assert train_fp8sr._fp8_gate(candidate, baseline)["status"] == "FAIL"


def test_writer_emits_fp16_v2_and_fp8_v1_from_logical_weights(tmp_path):
    layers = []
    for name, kernel, in_channels, out_channels in EXPECTED_LAYERS:
        weight = np.zeros((out_channels, in_channels, kernel, kernel), dtype=np.float32)
        bias = np.zeros((out_channels,), dtype=np.float32)
        weight[:, :, kernel // 2, kernel // 2] = 0.25
        layers.append({"name": name, "weight": weight, "bias": bias, "scale": 1.0})

    fp16_pack = write_pack(tmp_path / "fp16", layers, "Float16")
    fp8_pack = write_pack(tmp_path / "fp8", layers, "MetalFloat8E4M3")
    assert validate_pack(fp16_pack)["version"] == 2
    assert validate_pack(fp8_pack)["version"] == 1
    manifest = json.loads((fp8_pack / "manifest.json").read_text())
    assert manifest["weight_dtype"] == "MetalFloat8E4M3"
    assert manifest["layers"][0]["weights"] == "conv0.fp8"


def test_cli_exposes_all_pipeline_stages():
    parser = train_fp8sr.build_parser()
    arguments = {
        "train": ["--dataset", "data", "--output-dir", "out"],
        "export-fp16": ["--checkpoint", "x", "--output-pack", "y"],
        "quantize-fp8": ["--checkpoint", "x", "--output-pack", "y"],
        "verify": ["--dataset", "data", "--fp16-pack", "fp16", "--output-dir", "out"],
        "run": ["--dataset", "data", "--output-dir", "out"],
    }
    for command, command_args in arguments.items():
        parsed = parser.parse_args([command, *command_args])
        assert parsed.command == command
