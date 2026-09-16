import argparse
import json
import math
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
from fp8sr_pack import (  # noqa: E402
    EXPECTED_LAYERS,
    FP8SRPackError,
    _round_fp16,
    fp8sr_fp16_reference,
    validate_pack,
    write_pack,
)


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
    train_args = parser.parse_args(["train", "--dataset", "data", "--output-dir", "out"])
    assert train_args.adam_eps == pytest.approx(1e-4)


def test_fp16_reference_reports_nonfinite_instead_of_overflowing(tmp_path):
    layers = []
    for name, kernel, in_channels, out_channels in EXPECTED_LAYERS:
        layers.append(
            {
                "name": name,
                "weight": np.full(
                    (out_channels, in_channels, kernel, kernel),
                    65504.0,
                    dtype=np.float32,
                ),
                "bias": np.zeros((out_channels,), dtype=np.float32),
                "scale": 1.0,
            }
        )
    pack = write_pack(tmp_path / "overflow", layers, "Float16")
    source = tmp_path / "input.png"
    Image.new("RGB", (1, 1), (255, 255, 255)).save(source)

    with pytest.raises(FP8SRPackError, match="nonfinite"):
        fp8sr_fp16_reference(
            pack,
            source,
            tmp_path / "output.png",
            reject_nonfinite=True,
        )


def test_fp16_rounding_uses_the_overflow_midpoint():
    assert _round_fp16(65505.0) == 65504.0
    assert _round_fp16(65519.0) == 65504.0
    assert _round_fp16(-65519.0) == -65504.0
    assert math.isinf(_round_fp16(65520.0))
    assert math.isinf(_round_fp16(-65520.0))


def test_record_contains_timing_and_gpu_evidence_fields():
    record = train_fp8sr._record(
        "tensorops",
        "candidate",
        "MetalFloat8E4M3",
        "PASS",
        None,
        None,
        timing_ms={"median": 1.0, "p95": 2.0, "samples": [1.0, 2.0]},
        gpu_tools={
            "status": "PASS",
            "capture": {"path": "/tmp/test.gputrace"},
            "debug": {"path": "/tmp/gpudebug.json"},
            "metalperftrace": {
                "traces": ["/tmp/perf.atrc"],
                "overviews": ["/tmp/perf.overview.json"],
            },
            "tensorops_dispatch_observed": True,
        },
    )
    assert record["timing_ms"]["median"] == 1.0
    assert record["timing_ms"]["p95"] == 2.0
    assert record["tensorops_dispatch_observed"] is True
    assert len(record["gpu_evidence_paths"]) == 4
    assert record["neural_accelerator_confirmed"] is False


def test_fp16_gate_blocks_fp8_reference_execution(tmp_path, monkeypatch):
    make_pair(tmp_path, size=(8, 6))
    layers = []
    for name, kernel, in_channels, out_channels in EXPECTED_LAYERS:
        layers.append(
            {
                "name": name,
                "weight": np.zeros(
                    (out_channels, in_channels, kernel, kernel), dtype=np.float32
                ),
                "bias": np.zeros((out_channels,), dtype=np.float32),
                "scale": 1.0,
            }
        )
    fp16_pack = write_pack(tmp_path / "fp16", layers, "Float16")
    fp8_pack = write_pack(tmp_path / "fp8", layers, "MetalFloat8E4M3")
    calls = []
    original = train_fp8sr.fp8sr_fp16_reference

    def counted_reference(pack, *args, **kwargs):
        calls.append(Path(pack).name)
        return original(pack, *args, **kwargs)

    monkeypatch.setattr(train_fp8sr, "fp8sr_fp16_reference", counted_reference)
    args = argparse.Namespace(
        dataset=tmp_path,
        fp16_pack=fp16_pack,
        fp8_pack=fp8_pack,
        output_dir=tmp_path / "verify",
        record_jsonl=tmp_path / "verify" / "execution.jsonl",
        helper=tmp_path / "missing-helper",
        compare_runs=1,
        representative_count=1,
        gpu_tools=False,
    )

    assert train_fp8sr.verify(args) == 1
    assert calls == ["fp16"]
    assert not (tmp_path / "verify/representatives/0000_tile_fp8.png").exists()


def test_gpu_failure_changes_verification_exit_status(tmp_path, monkeypatch):
    make_pair(tmp_path, size=(8, 6))
    layers = []
    for name, kernel, in_channels, out_channels in EXPECTED_LAYERS:
        weight = np.zeros((out_channels, in_channels, kernel, kernel), dtype=np.float32)
        bias = np.zeros((out_channels,), dtype=np.float32)
        weight[:, :, kernel // 2, kernel // 2] = 0.25
        layers.append({"name": name, "weight": weight, "bias": bias, "scale": 1.0})
    fp16_pack = write_pack(tmp_path / "fp16", layers, "Float16")
    fp8_pack = write_pack(tmp_path / "fp8", layers, "MetalFloat8E4M3")
    fp8sr_fp16_reference(
        fp16_pack,
        tmp_path / "val/lr/tile.png",
        tmp_path / "val/hr/tile.png",
    )
    helper = tmp_path / "ASHelper"
    helper.write_text("placeholder")

    import verify_metal

    monkeypatch.setattr(train_fp8sr.sys, "platform", "darwin")
    monkeypatch.setattr(
        verify_metal,
        "compare_tensorops_backend",
        lambda *args, **kwargs: {
            "status": "FAIL",
            "run_exit": 7,
            "diagnostic": "shader_failed",
        },
    )
    args = argparse.Namespace(
        dataset=tmp_path,
        fp16_pack=fp16_pack,
        fp8_pack=fp8_pack,
        output_dir=tmp_path / "verify-gpu",
        record_jsonl=tmp_path / "verify-gpu" / "execution.jsonl",
        helper=helper,
        compare_runs=1,
        representative_count=1,
        gpu_tools=False,
    )

    assert train_fp8sr.verify(args) == 1
    report = json.loads((tmp_path / "verify-gpu/verification.json").read_text())
    assert report["status"] == "FAIL(gpu_execution)"
    assert report["gpu"]["status"] == "FAIL"
