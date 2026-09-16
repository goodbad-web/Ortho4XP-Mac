import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
TOOLS_ROOT = ROOT / "Utils" / "run"
PACK_ROOT = ROOT / "tools"
for path in (TOOLS_ROOT, PACK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import verify_metal  # noqa: E402
from fp8sr_pack import create_fixture  # noqa: E402


def make_args(tmp_path, **overrides):
    values = {
        "precision_reference": None,
        "fp16_pack": create_fixture(tmp_path / "fp16", "Float16"),
        "fp8_pack": create_fixture(tmp_path / "fp8", "MetalFloat8E4M3"),
        "fp4_pack": create_fixture(tmp_path / "fp4", "MetalFloat4E2M1"),
        "int2_pack": create_fixture(tmp_path / "int2", "Int2"),
        "compare_runs": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def fake_result(pack, status="PASS"):
    dtype = json.loads((pack / "manifest.json").read_text())["weight_dtype"]
    return {
        "status": status,
        "quality": {
            "mae": 0.0 if status == "PASS" else 10.0,
            "rmse": 0.0 if status == "PASS" else 10.0,
            "psnr_db": float("inf") if status == "PASS" else 20.0,
        },
        "timing_ms": {"median": 1.0, "p95": 1.0, "samples": [1.0]},
        "dtype": dtype,
    }


def test_quality_gate_requires_exact_match_when_fp16_error_is_zero():
    baseline = {"mae": 0.0, "rmse": 0.0, "psnr_db": float("inf")}
    candidate = {"mae": 0.0, "rmse": 0.0, "psnr_db": float("inf")}
    assert verify_metal.precision_quality_gate(candidate, baseline)["status"] == "PASS"
    candidate["rmse"] = 0.01
    assert verify_metal.precision_quality_gate(candidate, baseline)["status"] == "FAIL"


def test_precision_ladder_runs_all_stages_in_order(monkeypatch, tmp_path):
    calls = []

    def fake_compare(helper, pack, source, reference, output, runs):
        calls.append(json.loads((pack / "manifest.json").read_text())["weight_dtype"])
        return fake_result(pack)

    monkeypatch.setattr(verify_metal, "compare_tensorops_backend", fake_compare)
    report, pack, ok = verify_metal.run_precision_ladder(
        make_args(tmp_path),
        Path("ASHelper"),
        tmp_path / "artifacts",
        True,
        [],
    )

    assert ok is True
    assert calls == ["Float16", "MetalFloat8E4M3", "MetalFloat4E2M1", "Int2"]
    assert report["status"] == "PASS"
    assert pack is not None


def test_precision_ladder_blocks_fp4_and_int2_after_fp8_failure(monkeypatch, tmp_path):
    calls = []

    def fake_compare(helper, pack, source, reference, output, runs):
        dtype = json.loads((pack / "manifest.json").read_text())["weight_dtype"]
        calls.append(dtype)
        return fake_result(pack, "FAIL" if dtype == "MetalFloat8E4M3" else "PASS")

    monkeypatch.setattr(verify_metal, "compare_tensorops_backend", fake_compare)
    report, _, ok = verify_metal.run_precision_ladder(
        make_args(tmp_path),
        Path("ASHelper"),
        tmp_path / "artifacts",
        True,
        [],
    )

    assert ok is False
    assert calls == ["Float16", "MetalFloat8E4M3"]
    assert report["stages"]["MetalFloat4E2M1"]["status"].startswith("BLOCKED")
    assert report["stages"]["Int2"]["status"].startswith("BLOCKED")


def test_execution_record_keeps_canonical_backend_and_gpu_evidence(tmp_path):
    record_path = tmp_path / "records.jsonl"
    record = verify_metal.execution_record(
        backend="coreml",
        role="reference_only",
        status="PASS",
        gpu_tools={"tensorops_dispatch_observed": False},
    )
    verify_metal.write_execution_records(record_path, [record])
    saved = json.loads(record_path.read_text().splitlines()[0])
    assert saved["backend"] == "coreml"
    assert saved["role"] == "reference_only"
    assert saved["neural_accelerator_confirmed"] is False


def test_tensorops_dispatch_metadata_records_tiled_execution():
    diagnostics = """\
tensorops_dispatch=completed dtype=Float16 output=8192x8192
tensorops_dispatch=tiled tile=1/4 core=2048x2048 input=2049x2049 halo=1
tensorops_dispatch=tiled tile=2/4 core=2048x2048 input=2049x2049 halo=1
"""
    metadata = verify_metal.tensorops_dispatch_metadata(diagnostics)
    assert metadata["tensorops_dispatch"] == "tiled"
    assert metadata["tile_count"] == 4
    assert metadata["tile_core_size"] == [2048, 2048]
    assert metadata["tile_input_sizes"] == [[2049, 2049], [2049, 2049]]
    assert metadata["tile_halo"] == 1
    assert metadata["tensorops_output_size"] == [8192, 8192]


def test_neural_accelerator_counters_are_parsed_and_recorded(tmp_path):
    output = """{"children":[{"name":"neural_accelerator_utilization","values":[{"type":"string","value":"8.80%"}]},{"name":"neural_accelerator_limiter","values":[{"type":"string","value":"9.19%"}]}]}\n"""
    counters = verify_metal._neural_accelerator_counters(output)
    assert counters == {
        "neural_accelerator_utilization": "8.80%",
        "neural_accelerator_limiter": "9.19%",
    }
    record = verify_metal.execution_record(
        backend="tensorops",
        role="candidate",
        status="PASS",
        gpu_tools={
            "tensorops_dispatch_observed": True,
            "neural_accelerator_confirmed": True,
        },
    )
    assert record["neural_accelerator_confirmed"] is True
