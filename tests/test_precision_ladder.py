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


def test_direct_dds_performance_gate_requires_both_speed_targets():
    def result(median):
        return {"status": "PASS", "timing_ms": {"median": median}}

    passing = verify_metal.direct_dds_performance_gate(
        512,
        result(4.0),
        result(8.0),
        result(3.0),
    )
    assert passing["status"] == "PASS"
    assert passing["candidate_over_baseline_ratio"] == 0.5
    assert passing["candidate_over_metalfx_ratio"] < 2.0

    failing = verify_metal.direct_dds_performance_gate(
        2048,
        result(4.1),
        result(8.0),
        result(1.0),
    )
    assert failing["status"] == "FAIL"
    assert failing["checks"]["candidate_faster_than_old_tensorops"] is False

    nonfinite = verify_metal.direct_dds_performance_gate(
        512,
        result(float("inf")),
        result(8.0),
        result(3.0),
    )
    assert nonfinite["status"] == "FAIL"
    assert nonfinite["reason"] == "median_nonfinite_or_invalid"


def test_quality_gate_compares_candidate_against_pre_change_baseline():
    baseline = {"mae": 10.0, "rmse": 12.0, "psnr_db": 30.0}
    candidate = {"mae": 10.4, "rmse": 12.4, "psnr_db": 29.8}
    assert verify_metal.precision_quality_gate(candidate, baseline)["status"] == "PASS"

    regression = {"mae": 11.0, "rmse": 12.4, "psnr_db": 29.8}
    assert verify_metal.precision_quality_gate(regression, baseline)["status"] == "FAIL"


def test_run_measured_reports_child_scoped_rss(tmp_path):
    helper = tmp_path / "measured-child.py"
    helper.write_text(
        "import sys\n"
        "sys.stdout.write('measured\\n')\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )
    result, timing = verify_metal.run_measured([sys.executable, str(helper)])
    assert result.returncode == 0
    assert result.stdout == "measured\n"
    if hasattr(verify_metal.os, "wait4"):
        assert timing["rss_scope"] == "child_wait4"
        assert timing["peak_rss_mb"] > 0


def test_tensorops_baseline_helper_is_required_and_executable(tmp_path):
    assert (
        verify_metal.validate_tensorops_baseline_helper(None)
        == "--compare-tensorops requires --tensorops-baseline-helper"
    )
    missing = tmp_path / "missing-helper"
    assert "does not exist" in verify_metal.validate_tensorops_baseline_helper(missing)
    helper = tmp_path / "old-ASHelper"
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert "not executable" in verify_metal.validate_tensorops_baseline_helper(helper)
    helper.chmod(0o755)
    assert verify_metal.validate_tensorops_baseline_helper(helper) is None


def test_tile_snapshot_is_copied_per_backend_without_canonical_outputs(tmp_path):
    from PIL import Image

    source = tmp_path / "provider.jpg"
    mask = tmp_path / "mask.png"
    mesh = tmp_path / "mesh.dsf"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cached.jpg").write_bytes(b"cache")
    Image.new("RGB", (4, 3), (20, 40, 60)).save(source, format="JPEG")
    Image.new("L", (2, 2), 255).save(mask, format="PNG")
    mesh.write_bytes(b"mesh")
    manifest = tmp_path / "tile.json"
    manifest.write_text(
        json.dumps(
            {
                "tile": "+35+135",
                "items": [{"input": str(source), "mask": str(mask), "format": "BC3"}],
                "mesh": [str(mesh)],
                "cache": [str(cache)],
            }
        ),
        encoding="utf-8",
    )

    snapshot = verify_metal.prepare_tile_snapshot(manifest, tmp_path / "artifacts")

    assert snapshot["canonical_outputs_written"] is False
    assert set(snapshot["roles"]) == {"baseline", "candidate", "metalfx_spatial"}
    copied_inputs = [
        Path(snapshot["roles"][role]["items"][0]["input"])
        for role in snapshot["roles"]
    ]
    assert all(path.is_file() for path in copied_inputs)
    assert len({path.read_bytes() for path in copied_inputs}) == 1
    assert all(
        str(path).startswith(str(tmp_path / "artifacts")) for path in copied_inputs
    )
    assert not (tmp_path / "provider.gpu.tmp.dds").exists()


def test_tile_snapshot_requires_canonical_tile_name(tmp_path):
    manifest = tmp_path / "tile.json"
    manifest.write_text(
        json.dumps({"tile": "+34+132", "items": [], "mesh": [], "cache": []}),
        encoding="utf-8",
    )
    try:
        verify_metal.prepare_tile_snapshot(manifest, tmp_path / "artifacts")
    except ValueError as error:
        assert "+35+135" in str(error)
    else:
        raise AssertionError("non-canonical tile must be rejected")


def test_tile_snapshot_performance_gate_requires_both_speed_targets():
    def result(median):
        return {"status": "PASS", "timing_ms": {"median": median}}

    assert verify_metal.tile_snapshot_performance_gate(
        result(4), result(8), result(2)
    )["status"] == "PASS"
    assert verify_metal.tile_snapshot_performance_gate(
        result(4.1), result(8), result(1)
    )["status"] == "FAIL"
    nonfinite = verify_metal.tile_snapshot_performance_gate(
        result(4), result(float("inf")), result(2)
    )
    assert nonfinite["status"] == "FAIL"
    assert nonfinite["reason"] == "median_nonfinite_or_invalid"


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


def test_direct_dds_comparison_validates_backend_and_mipmaps(tmp_path):
    from PIL import Image

    source = tmp_path / "source.jpg"
    Image.new("RGB", (2, 2), (30, 60, 90)).save(source, format="JPEG")
    request_log = tmp_path / "requests.jsonl"
    helper = tmp_path / "fake_direct_dds_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import struct\n"
        "import sys\n"
        f"with open({str(request_log)!r}, 'a', encoding='utf-8') as log:\n"
        "    request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "    log.write(json.dumps(request) + '\\n')\n"
        "header = bytearray(124)\n"
        "struct.pack_into('<I', header, 0, 124)\n"
        "struct.pack_into('<I', header, 8, 4)\n"
        "struct.pack_into('<I', header, 12, 4)\n"
        "struct.pack_into('<I', header, 24, 2)\n"
        "struct.pack_into('<I', header, 72, 32)\n"
        "header[80:84] = b'DXT5'\n"
        "for item in request['items']:\n"
        "    with open(item['output'], 'wb') as output:\n"
        "        output.write(b'DDS ' + header + bytes(32))\n"
        "    prefix = 'tensorops_dds_item' if sys.argv[1].startswith('--tensorops') else 'metalfx_dds_item'\n"
        "    backend = 'tensorops' if prefix.startswith('tensorops') else 'metalfx_spatial'\n"
        "    extra = ' tensorops_dispatch_observed=true' if backend == 'tensorops' else ''\n"
        "    print(f'{prefix}=1/1 backend={backend} effective_backend={backend} dispatch=direct_dds' + extra + ' tensorops_ms=3 metalfx_ms=4 readback_ms=2 dds_ms=1 total_ms=6 rss_mb=42')\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    tensorops_result = verify_metal.compare_direct_dds_backend(
        helper,
        "tensorops",
        tmp_path / "pack",
        source,
        None,
        tmp_path / "tensorops.dds",
        1,
    )
    metalfx_result = verify_metal.compare_direct_dds_backend(
        helper,
        "metalfx_spatial",
        None,
        source,
        None,
        tmp_path / "metalfx.dds",
        1,
    )

    assert tensorops_result["status"] == "PASS"
    assert tensorops_result["effective_backend"] == "tensorops"
    assert tensorops_result["dds"]["mipmaps"] == 2
    assert metalfx_result["status"] == "PASS"
    assert metalfx_result["effective_backend"] == "metalfx_spatial"
    requests = [json.loads(line) for line in request_log.read_text().splitlines()]
    assert requests[0]["fallback_to_ci"] is False
    assert "pack" in requests[0]
    assert "pack" not in requests[-1]
