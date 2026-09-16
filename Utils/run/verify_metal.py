#!/usr/bin/env python3
"""Run deterministic ASHelper/Metal checks without changing the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TOOLS))
os.chdir(ROOT)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def build_coreml_reference_helper(artifact_dir: Path) -> tuple[Path | None, str]:
    helper = artifact_dir / "CoreMLReference"
    module_cache = artifact_dir / "coreml-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-O",
            "-module-cache-path",
            str(module_cache),
            str(ROOT / "Utils/run/CoreMLReference.swift"),
            "-o",
            str(helper),
            "-framework",
            "Foundation",
            "-framework",
            "CoreGraphics",
            "-framework",
            "CoreImage",
            "-framework",
            "CoreML",
            "-framework",
            "CoreVideo",
            "-framework",
            "ImageIO",
            "-framework",
            "UniformTypeIdentifiers",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0 or not helper.is_file():
        return None, (result.stdout or "").strip()
    return helper, (result.stdout or "").strip()


def run_coreml_reference(
    model: Path,
    source: Path,
    output: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Run the optional Core ML reference model without entering normal runtime."""
    report: dict[str, Any] = {"model": str(model), "output": str(output)}
    if not model.exists():
        report["status"] = "FAIL(model_missing)"
        report["diagnostic"] = f"model does not exist: {model}"
        return report
    helper, compile_detail = build_coreml_reference_helper(artifact_dir)
    if helper is None:
        report["status"] = "FAIL(compile)"
        report["diagnostic"] = compile_detail
        return report
    result = subprocess.run(
        [str(helper), str(model), str(source), str(output)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    report["exit"] = result.returncode
    report["diagnostic"] = (result.stdout or "").strip()
    report["status"] = "PASS" if result.returncode == 0 and output.is_file() else "FAIL(runtime)"
    return report


def parse_probe(probe: Path) -> tuple[bool, str, bool]:
    result = subprocess.run(
        [str(probe)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("--- Metal probe ---")
    print(result.stdout.rstrip())
    if result.returncode != 0:
        fail(f"Metal probe failed with exit {result.returncode}")
    available = "metal_available=true" in result.stdout.splitlines()
    host_metal_supported = False
    if not available and sys.platform == "darwin":
        host_result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        host_metal_supported = "Metal: Supported" in host_result.stdout
        if host_metal_supported:
            print("metal_host_supported=true")
            print("metal_process_access=unavailable")
            print("hint=run this script outside the host sandbox to exercise Metal")
    return available, result.stdout, host_metal_supported


def make_fixtures(directory: Path) -> tuple[Path, Path, Path, Path, Path]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        fail(f"Pillow is required to generate fixtures: {error}")

    width = height = 512
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 255) // (width - 1),
                (y * 255) // (height - 1),
                ((x + y) * 255) // (width + height - 2),
            )
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 180, 180), fill=(225, 35, 45))
    draw.ellipse((210, 40, 470, 300), fill=(35, 185, 80))
    draw.line((0, 390, 512, 390), fill=(250, 240, 30), width=7)
    draw.rectangle((64, 420, 448, 480), fill=(40, 70, 220))

    source = directory / "source.jpg"
    image.save(source, format="JPEG", quality=97, subsampling=0)

    resampling = getattr(Image, "Resampling", Image).BICUBIC
    source_4096 = directory / "source_4096.png"
    image.resize((4096, 4096), resampling).save(source_4096, format="PNG")
    upscale_seed_2048 = directory / "upscale_seed_2048.png"
    image.resize((2048, 2048), resampling).save(upscale_seed_2048, format="PNG")

    # Deliberately lower resolution than the source. ASHelper must scale it
    # before applying it to the high-resolution image.
    mask = Image.new("L", (128, 128), color=0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((64, 0, 127, 127), fill=255)
    mask_draw.rectangle((24, 24, 40, 104), fill=128)
    mask_draw.line((0, 0, 127, 127), fill=64, width=3)
    mask_path = directory / "low_resolution_mask.png"
    mask.save(mask_path, format="PNG")

    # Keep an alpha-channel version to distinguish a low-resolution scaling
    # problem from CIBlendWithAlphaMask's channel semantics. The production
    # mask files are L-mode grayscale images, so this is a control case.
    alpha_mask = Image.new("RGBA", (128, 128), (255, 255, 255, 0))
    alpha_mask_draw = ImageDraw.Draw(alpha_mask)
    alpha_mask_draw.rectangle((64, 0, 127, 127), fill=(255, 255, 255, 255))
    alpha_mask_draw.rectangle((24, 24, 40, 104), fill=(255, 255, 255, 128))
    alpha_mask_draw.line((0, 0, 127, 127), fill=(255, 255, 255, 64), width=3)
    alpha_mask_path = directory / "low_resolution_alpha_mask.png"
    alpha_mask.save(alpha_mask_path, format="PNG")
    return source, mask_path, alpha_mask_path, source_4096, upscale_seed_2048


def make_upscale_comparison_fixture(directory: Path, target_size: int) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    master = Image.new("RGB", (target_size, target_size))
    pixels = master.load()
    for y in range(target_size):
        for x in range(target_size):
            pixels[x, y] = (
                (x * 255) // max(1, target_size - 1),
                (y * 255) // max(1, target_size - 1),
                ((x * 3 + y * 5) * 255) // max(1, target_size * 8 - 8),
            )
    draw = ImageDraw.Draw(master)
    line_width = max(1, target_size // 512)
    draw.rectangle(
        (target_size // 16, target_size // 16, target_size * 7 // 16, target_size * 7 // 16),
        fill=(225, 35, 45),
        outline=(255, 255, 255),
        width=line_width,
    )
    draw.ellipse(
        (target_size * 5 // 16, target_size // 8, target_size * 15 // 16, target_size * 5 // 8),
        fill=(35, 185, 80),
        outline=(10, 10, 10),
        width=line_width,
    )
    draw.line(
        (0, target_size * 3 // 4, target_size, target_size * 3 // 4),
        fill=(250, 240, 30),
        width=max(2, target_size // 128),
    )
    for x in range(target_size // 2, target_size, max(2, target_size // 64)):
        draw.line((x, target_size * 5 // 8, x, target_size), fill=(30, 30, 30), width=line_width)

    resampling = getattr(Image, "Resampling", Image)
    source = master.resize((target_size // 2, target_size // 2), resampling.BOX)
    source_path = directory / f"comparison_source_{target_size // 2}.png"
    reference_path = directory / f"comparison_reference_{target_size}.png"
    source.save(source_path, format="PNG")
    master.save(reference_path, format="PNG")
    return source_path, reference_path


def image_quality_metrics(output_path: Path, reference_path: Path) -> dict[str, Any]:
    from PIL import Image, ImageChops, ImageStat

    with Image.open(output_path).convert("RGB") as output, Image.open(reference_path).convert("RGB") as reference:
        if output.size != reference.size:
            raise ValueError(f"size {output.size} != reference {reference.size}")
        difference = ImageChops.difference(output, reference)
        stats = ImageStat.Stat(difference)
        mae_channels = [float(value) for value in stats.mean]
        rms_channels = [float(value) for value in stats.rms]
        mae = sum(mae_channels) / len(mae_channels)
        rmse = math.sqrt(sum(value * value for value in rms_channels) / len(rms_channels))
        psnr = float("inf") if rmse == 0 else 20.0 * math.log10(255.0 / rmse)
        return {
            "mae": mae,
            "mae_rgb": mae_channels,
            "rmse": rmse,
            "rmse_rgb": rms_channels,
            "psnr_db": psnr,
        }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no timing samples")
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def tensorops_dispatch_metadata(diagnostics: str) -> dict[str, Any]:
    """Extract tiled TensorOps execution details from ASHelper diagnostics."""
    metadata: dict[str, Any] = {
        "tensorops_dispatch": "single",
        "tile_count": None,
        "tile_core_size": None,
        "tile_input_sizes": [],
        "tile_halo": None,
    }
    tile_pattern = re.compile(
        r"tensorops_dispatch=tiled\s+tile=(\d+)/(\d+)\s+"
        r"core=(\d+)x(\d+)\s+input=(\d+)x(\d+)\s+halo=(\d+)"
    )
    for match in tile_pattern.finditer(diagnostics):
        metadata["tensorops_dispatch"] = "tiled"
        metadata["tile_count"] = int(match.group(2))
        metadata["tile_core_size"] = [int(match.group(3)), int(match.group(4))]
        metadata["tile_input_sizes"].append([int(match.group(5)), int(match.group(6))])
        metadata["tile_halo"] = int(match.group(7))
    output_match = re.search(
        r"tensorops_dispatch=completed.*?output=(\d+)x(\d+)", diagnostics
    )
    if output_match:
        metadata["tensorops_output_size"] = [
            int(output_match.group(1)),
            int(output_match.group(2)),
        ]
    return metadata


def diagnostic_fallback_reason(diagnostics: str) -> str | None:
    match = re.search(r"fallback reason=([^\s]+)", diagnostics)
    return match.group(1) if match else None


QUALITY_PSNR_DROP_DB = 0.25
QUALITY_ERROR_INCREASE_RATIO = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_record_metadata(pack: Path) -> dict[str, Any]:
    manifest = pack / "manifest.json"
    metadata: dict[str, Any] = {"pack": str(pack), "manifest_sha256": None}
    if manifest.is_file():
        metadata["manifest_sha256"] = sha256_file(manifest)
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_data = {}
        metadata["dtype"] = manifest_data.get("weight_dtype")
        metadata["pack_version"] = manifest_data.get("version")
    return metadata


def precision_quality_gate(
    candidate_quality: dict[str, Any], baseline_quality: dict[str, Any]
) -> dict[str, Any]:
    candidate_psnr = float(candidate_quality["psnr_db"])
    baseline_psnr = float(baseline_quality["psnr_db"])
    candidate_mae = float(candidate_quality["mae"])
    baseline_mae = float(baseline_quality["mae"])
    candidate_rmse = float(candidate_quality["rmse"])
    baseline_rmse = float(baseline_quality["rmse"])
    psnr_limit = baseline_psnr - QUALITY_PSNR_DROP_DB
    if math.isinf(baseline_psnr):
        psnr_pass = math.isinf(candidate_psnr)
    else:
        psnr_pass = candidate_psnr >= psnr_limit
    mae_limit = baseline_mae * (1.0 + QUALITY_ERROR_INCREASE_RATIO)
    rmse_limit = baseline_rmse * (1.0 + QUALITY_ERROR_INCREASE_RATIO)
    mae_pass = candidate_mae <= mae_limit if baseline_mae else candidate_mae == 0.0
    rmse_pass = candidate_rmse <= rmse_limit if baseline_rmse else candidate_rmse == 0.0
    return {
        "status": "PASS" if psnr_pass and mae_pass and rmse_pass else "FAIL",
        "psnr_drop_limit_db": QUALITY_PSNR_DROP_DB,
        "error_increase_limit_ratio": QUALITY_ERROR_INCREASE_RATIO,
        "baseline": baseline_quality,
        "candidate": candidate_quality,
        "psnr_limit_db": psnr_limit,
        "mae_limit": mae_limit,
        "rmse_limit": rmse_limit,
        "checks": {
            "psnr": psnr_pass,
            "mae": mae_pass,
            "rmse": rmse_pass,
        },
    }


def execution_record(
    *,
    backend: str,
    role: str,
    dtype: str | None = None,
    status: str,
    requested_backend: str | None = None,
    effective_backend: str | None = None,
    source: Path | None = None,
    output: Path | None = None,
    result: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
    gpu_tools: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": backend,
        "role": role,
        "dtype": dtype,
        "requested_backend": requested_backend or backend,
        "effective_backend": effective_backend or backend,
        "status": status,
        "fallback_reason": fallback_reason,
        "host": platform.machine(),
        "os": platform.platform(),
    }
    if source is not None:
        record["input"] = str(source)
        try:
            from PIL import Image

            with Image.open(source) as image:
                record["input_size"] = list(image.size)
        except Exception:
            pass
    if output is not None:
        record["output"] = str(output)
        try:
            from PIL import Image

            with Image.open(output) as image:
                record["output_size"] = list(image.size)
        except Exception:
            pass
    if result:
        for key in ("pack", "manifest_sha256", "pack_version"):
            if key in result:
                record[key] = result[key]
        for key in (
            "requested_backend",
            "effective_backend",
            "dispatch",
            "alpha_mode",
            "batch_tasks",
            "batch_success",
            "batch_fallback",
            "model_manifest_hash",
            "tensorops_dispatch",
            "tile_count",
            "tile_core_size",
            "tile_input_sizes",
            "tile_halo",
            "tensorops_output_size",
            "diagnostic",
        ):
            if key in result:
                record[key] = result[key]
        if record["fallback_reason"] is None and result.get("fallback_reason"):
            record["fallback_reason"] = result["fallback_reason"]
        if "timing_ms" in result:
            record["timing_ms"] = result["timing_ms"]
        if "quality" in result:
            record["quality"] = result["quality"]
        if "gate" in result:
            record["gate"] = result["gate"]
        if "run_exit" in result:
            record["exit_code"] = result["run_exit"]
        elif "warmup_exit" in result:
            record["exit_code"] = result["warmup_exit"]
        for key in ("tensorops_dispatch_observed", "neural_accelerator_confirmed"):
            if key in result:
                record[key] = bool(result[key])
    if gpu_tools is not None:
        record["gpu_tools"] = gpu_tools
        evidence_paths: list[str] = []
        capture = gpu_tools.get("capture", {})
        if capture.get("path"):
            evidence_paths.append(str(capture["path"]))
        debug = gpu_tools.get("debug", {})
        if debug.get("path"):
            evidence_paths.append(str(debug["path"]))
        profile = gpu_tools.get("profile", {})
        if profile.get("path"):
            evidence_paths.append(str(profile["path"]))
        for key in ("traces", "overviews"):
            evidence_paths.extend(
                str(path)
                for path in gpu_tools.get("metalperftrace", {}).get(key, [])
            )
        record["gpu_evidence_paths"] = evidence_paths
        record["tensorops_dispatch_observed"] = bool(
            gpu_tools.get("tensorops_dispatch_observed", False)
        )
        record["neural_accelerator_confirmed"] = bool(
            gpu_tools.get("neural_accelerator_confirmed", False)
        )
    if "exit_code" not in record:
        record["exit_code"] = 0 if status.startswith("PASS") else None
    return record


def write_execution_records(path: Path | None, records: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def compare_upscale_backend(
    helper: Path,
    backend: str,
    source: Path,
    reference: Path,
    output: Path,
    runs: int,
) -> dict[str, Any]:
    command_name = (
        "--ci-lanczos-upscale" if backend == "ci_lanczos" else "--metalfx-spatial-upscale"
    )
    warmup_output = output.with_name(output.stem + "_warmup.png")
    warmup = subprocess.run(
        [str(helper), command_name, str(source), str(warmup_output)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if warmup.returncode != 0 or not warmup_output.is_file():
        return {
            "status": "FAIL",
            "backend": backend,
            "warmup_exit": warmup.returncode,
            "output": str(output),
            "diagnostic": (warmup.stdout or "").strip(),
        }

    samples_ms: list[float] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for _ in range(runs):
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        started = time.perf_counter()
        last_result = subprocess.run(
            [str(helper), command_name, str(source), str(output)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        if last_result.returncode != 0 or not output.is_file():
            return {
                "status": "FAIL",
                "backend": backend,
                "warmup_exit": warmup.returncode,
                "run_exit": last_result.returncode,
                "output": str(output),
                "diagnostic": (last_result.stdout or "").strip(),
            }

    try:
        quality = image_quality_metrics(output, reference)
    except ValueError as error:
        return {
            "status": "FAIL",
            "backend": backend,
            "output": str(output),
            "error": str(error),
        }
    return {
        "status": "PASS",
        "backend": backend,
        "output": str(output),
        "runs": runs,
        "timing_ms": {
            "samples": samples_ms,
            "median": percentile(samples_ms, 0.5),
            "p95": percentile(samples_ms, 0.95),
            "first_measured": samples_ms[0],
            "scope": "ASHelper process plus image decode, upscale, readback, and PNG encode",
        },
        "quality": quality,
    }


def compare_metalfx_batch(
    helper: Path,
    pairs: list[tuple[Path, Path, Path]],
    runs: int,
) -> dict[str, Any]:
    """Measure one MetalFX batch and validate every output independently."""
    from PIL import Image

    if not pairs:
        return {"status": "FAIL", "backend": "metalfx_spatial", "error": "empty_batch"}

    def command(outputs: list[Path]) -> list[str]:
        arguments = [str(helper), "--metalfx-spatial-upscale-batch"]
        for (source, _, _), output in zip(pairs, outputs):
            arguments.extend([str(source), str(output)])
        return arguments

    outputs = [output for _, output, _ in pairs]
    warmup_outputs = [
        output.with_name(output.stem + "_warmup.png") for output in outputs
    ]
    warmup = subprocess.run(
        command(warmup_outputs),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if warmup.returncode != 0 or not all(path.is_file() for path in warmup_outputs):
        return {
            "status": "FAIL",
            "backend": "metalfx_spatial",
            "dispatch": "batch",
            "batch_tasks": len(pairs),
            "warmup_exit": warmup.returncode,
            "diagnostic": (warmup.stdout or "").strip(),
        }

    samples_ms: list[float] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for _ in range(runs):
        for output in outputs:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
        started = time.perf_counter()
        last_result = subprocess.run(
            command(outputs),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        if last_result.returncode != 0 or not all(path.is_file() for path in outputs):
            return {
                "status": "FAIL",
                "backend": "metalfx_spatial",
                "dispatch": "batch",
                "batch_tasks": len(pairs),
                "run_exit": last_result.returncode,
                "diagnostic": (last_result.stdout or "").strip(),
            }

    quality: list[dict[str, Any]] = []
    try:
        for (_, output, reference) in pairs:
            with Image.open(output) as image, Image.open(reference) as ref:
                if image.size != (ref.width, ref.height):
                    raise ValueError(
                        f"batch output size {image.size} != reference {ref.size}"
                    )
            quality.append(image_quality_metrics(output, reference))
    except (OSError, ValueError) as error:
        return {
            "status": "FAIL",
            "backend": "metalfx_spatial",
            "dispatch": "batch",
            "batch_tasks": len(pairs),
            "error": str(error),
        }
    diagnostics = "\n".join(
        value
        for value in (
            warmup.stdout or "",
            last_result.stdout if last_result is not None else "",
        )
        if value
    ).strip()
    return {
        "status": "PASS",
        "backend": "metalfx_spatial",
        "effective_backend": "metalfx_spatial",
        "requested_backend": "metalfx_spatial",
        "dispatch": "batch",
        "alpha_mode": "opaque",
        "batch_tasks": len(pairs),
        "batch_success": len(pairs),
        "batch_fallback": diagnostics.count("effective_backend=ci_lanczos"),
        "runs": runs,
        "timing_ms": {
            "samples": samples_ms,
            "median": percentile(samples_ms, 0.5),
            "p95": percentile(samples_ms, 0.95),
            "first_measured": samples_ms[0],
            "scope": "ASHelper batch process plus image decode, MetalFX, readback, and PNG encode",
        },
        "quality": quality,
        "diagnostic": diagnostics,
    }


def compare_tensorops_backend(
    helper: Path,
    pack: Path,
    source: Path,
    reference: Path,
    output: Path,
    runs: int,
) -> dict[str, Any]:
    command_prefix = [
        str(helper),
        "--tensorops-upscale",
        str(pack),
        str(source),
    ]
    warmup_output = output.with_name(output.stem + "_warmup.png")
    warmup = subprocess.run(
        command_prefix + [str(warmup_output)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if warmup.returncode != 0 or not warmup_output.is_file():
        diagnostic = (warmup.stdout or "").strip()
        return {
            "status": "FAIL",
            "backend": "tensorops",
            "warmup_exit": warmup.returncode,
            "output": str(output),
            "diagnostic": diagnostic,
            "fallback_reason": diagnostic_fallback_reason(diagnostic),
        }

    samples_ms: list[float] = []
    last_result: subprocess.CompletedProcess[str] | None = None
    for _ in range(runs):
        try:
            output.unlink()
        except FileNotFoundError:
            pass
        started = time.perf_counter()
        last_result = subprocess.run(
            command_prefix + [str(output)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        if last_result.returncode != 0 or not output.is_file():
            diagnostic = (last_result.stdout or "").strip()
            return {
                "status": "FAIL",
                "backend": "tensorops",
                "warmup_exit": warmup.returncode,
                "run_exit": last_result.returncode,
                "output": str(output),
                "diagnostic": diagnostic,
                "fallback_reason": diagnostic_fallback_reason(diagnostic),
            }

    try:
        quality = image_quality_metrics(output, reference)
    except ValueError as error:
        return {
            "status": "FAIL",
            "backend": "tensorops",
            "output": str(output),
            "error": str(error),
        }
    diagnostics = "\n".join(
        value
        for value in (
            warmup.stdout or "",
            last_result.stdout if last_result is not None else "",
        )
        if value
    ).strip()
    dispatch_metadata = tensorops_dispatch_metadata(diagnostics)
    return {
        "status": "PASS",
        "backend": "tensorops",
        "pack": str(pack),
        **pack_record_metadata(pack),
        "output": str(output),
        "runs": runs,
        "timing_ms": {
            "samples": samples_ms,
            "median": percentile(samples_ms, 0.5),
            "p95": percentile(samples_ms, 0.95),
            "first_measured": samples_ms[0],
            "scope": "ASHelper process plus image decode, TensorOps dispatch, readback, and PNG encode",
        },
        "quality": quality,
        "diagnostic": diagnostics,
        **dispatch_metadata,
        "tensorops_dispatch_observed": "tensorops_dispatch=completed" in diagnostics,
    }


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def _neural_accelerator_counters(output: str) -> dict[str, str]:
    """Extract Neural Accelerator counters from gpudebug JSON-lines output."""
    wanted = {
        "neural_accelerator_limiter",
        "neural_accelerator_utilization",
    }
    counters: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get("name")
            if name in wanted:
                for item in value.get("values", []):
                    if isinstance(item, dict) and item.get("type") == "string":
                        counters[name] = str(item.get("value", ""))
                        break
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in output.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    return counters


def _percentage_value(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip().rstrip("%"))
    except ValueError:
        return None


def run_gpu_tool_verification(
    helper: Path,
    pack: Path,
    source: Path,
    output: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Capture one TensorOps process without making capture a normal runtime dependency."""
    required = ("gpucapture", "gpudebug", "metalperftrace")
    missing = [name for name in required if not _tool_available(name)]
    if missing:
        return {"status": "SKIP(tool_missing)", "missing": missing}
    if not helper.is_file() or not os.access(helper, os.X_OK):
        return {
            "status": "SKIP(helper_unavailable)",
            "diagnostic": f"ASHelper is missing or not executable: {helper}",
        }

    capture_path = artifact_dir / "tensorops.gputrace"
    debug_dir = artifact_dir / "gpudebug"
    profile_dir = artifact_dir / "gpudebug-profile"
    perf_dir = artifact_dir / "metalperftrace"
    debug_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MTL_CAPTURE_ENABLED"] = "1"
    env["MTL_CAPTURE_WAIT_FOR_SIGNAL"] = "1"
    env["ORTHO4XP_GPU_CAPTURE_WAIT_SECONDS"] = "10"
    capture_help = subprocess.run(
        ["gpucapture", "start", "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout or ""
    # macOS 27's gpucapture uses a numeric boundary ID even though the
    # documented contract names the device boundary. Keep the symbolic probe
    # for newer tools and use the local numeric spelling when advertised.
    boundaries = ("0",) if "ID of the boundary object" in capture_help else ("Device", "0")
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(helper), "--tensorops-upscale", str(pack), str(source), str(output)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    capture: dict[str, Any] = {"status": "SKIP(capture_unavailable)"}
    debug: dict[str, Any] = {"status": "SKIP(capture_unavailable)"}
    profile: dict[str, Any] = {"status": "SKIP(profile_not_run)"}
    try:
        # ASHelper creates the Metal device immediately and then waits in its
        # capture-only path. Give gpucapture a process that already owns that
        # device before attaching.
        time.sleep(1.0)
        attempts: list[dict[str, Any]] = []
        for boundary in boundaries:
            capture_start = subprocess.run(
                [
                    "gpucapture",
                    "start",
                    "--pid",
                    str(process.pid),
                    "--boundary",
                    boundary,
                    "--count",
                    "1",
                    "--output",
                    str(capture_path),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
            attempt = {
                "boundary": boundary,
                "exit_code": capture_start.returncode,
                "diagnostic": (capture_start.stdout or "").strip(),
            }
            attempts.append(attempt)
            if capture_start.returncode == 0 and capture_path.exists():
                break
            if process.poll() is not None:
                break
        capture["boundary_attempts"] = attempts
        last_attempt = attempts[-1]
        capture["exit_code"] = last_attempt["exit_code"]
        capture["diagnostic"] = last_attempt["diagnostic"]
        process.wait(timeout=60)
        process_output = process.stdout.read() if process.stdout else ""
        capture["process_output"] = process_output.strip()
        if capture_path.exists() and last_attempt["exit_code"] == 0:
            capture["status"] = "PASS"
            capture["path"] = str(capture_path)
            debug_result = subprocess.run(
                [
                    "gpudebug",
                    "--oneshot",
                    "--quiet",
                    "--json",
                    "--gputrace",
                    str(capture_path),
                    "--output",
                    str(debug_dir),
                    "-c",
                    "status",
                    "-c",
                    "go commands",
                    "-c",
                    "go cb0",
                    "-c",
                    "go ce0",
                    "-c",
                    "list",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            debug_text = debug_result.stdout or ""
            debug_path = debug_dir / "gpudebug.json"
            debug_path.write_text(debug_text, encoding="utf-8")
            debug = {
                "status": "PASS" if debug_result.returncode == 0 else "SKIP(debug_failed)",
                "exit_code": debug_result.returncode,
                "path": str(debug_path),
                "diagnostic": debug_text.strip(),
                "compute_dispatch_observed": "dispatches" in debug_text.lower()
                and "mtl4computecommandencoder" in debug_text.lower(),
                "tensorops_shader_observed": "fp8sr" in debug_text.lower()
                or "tensorops" in debug_text.lower(),
            }
        else:
            debug["diagnostic"] = "GPU trace was not produced"
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as error:
        capture["diagnostic"] = str(error)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    if capture.get("status") == "PASS" and debug.get("status") == "PASS":
        profile_result = subprocess.run(
            [
                "gpudebug",
                "--oneshot",
                "--quiet",
                "--json",
                "--timeout",
                "180",
                "--gputrace",
                str(capture_path),
                "--output",
                str(profile_dir),
                "-c",
                "profile run --gpu-state high --exec serial --embed",
                "-c",
                "profile load 0",
                "-c",
                "go performance/timeline/counters/neural_accelerator",
                "-c",
                "list",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=240,
        )
        profile_text = profile_result.stdout or ""
        profile_path = profile_dir / "neural_accelerator_profile.jsonl"
        profile_path.write_text(profile_text, encoding="utf-8")
        counters = _neural_accelerator_counters(profile_text)
        utilization = _percentage_value(
            counters.get("neural_accelerator_utilization")
        )
        limiter = _percentage_value(counters.get("neural_accelerator_limiter"))
        confirmed = utilization is not None and utilization > 0.0
        profile = {
            "status": "PASS" if confirmed else "SKIP(neural_counter_unavailable)",
            "exit_code": profile_result.returncode,
            "path": str(profile_path),
            "counters": counters,
            "neural_accelerator_utilization_percent": utilization,
            "neural_accelerator_limiter_percent": limiter,
            "neural_accelerator_confirmed": confirmed,
            "diagnostic": profile_text.strip(),
        }

    elapsed_seconds = max(5, int(math.ceil(time.perf_counter() - started)) + 2)
    perf_collect = subprocess.run(
        [
            "metalperftrace",
            "collect",
            "--last",
            f"{elapsed_seconds}s",
            "--prefix",
            "Ortho4XP",
            "--json",
            str(perf_dir),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    perf: dict[str, Any] = {
        "status": "SKIP(no_metal_layer_data)",
        "collect_exit_code": perf_collect.returncode,
        "collect_output": (perf_collect.stdout or "").strip(),
    }
    overview_paths: list[str] = []
    for trace in sorted(perf_dir.glob("*.atrc")):
        overview = subprocess.run(
            ["metalperftrace", "overview", "--json", str(trace)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        overview_path = perf_dir / f"{trace.stem}.overview.json"
        overview_path.write_text(overview.stdout or "", encoding="utf-8")
        overview_paths.append(str(overview_path))
        if overview.returncode == 0 and (overview.stdout or "").strip():
            perf["status"] = "PASS"
    perf["traces"] = [str(path) for path in sorted(perf_dir.glob("*.atrc"))]
    perf["overviews"] = overview_paths
    tensorops_observed = bool(
        capture.get("status") == "PASS"
        and debug.get("compute_dispatch_observed")
        and debug.get("tensorops_shader_observed")
    )
    return {
        "status": "PASS" if tensorops_observed else "SKIP(gpu_evidence_incomplete)",
        "capture": capture,
        "debug": debug,
        "profile": profile,
        "metalperftrace": perf,
        "tensorops_dispatch_observed": tensorops_observed,
        "neural_accelerator_confirmed": bool(
            profile.get("neural_accelerator_confirmed", False)
        ),
    }


def dds_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("file is missing")
    data = path.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError("DDS header is missing")

    header_size = struct.unpack_from("<I", data, 4)[0]
    width = struct.unpack_from("<I", data, 16)[0]
    height = struct.unpack_from("<I", data, 12)[0]
    mipmaps = struct.unpack_from("<I", data, 28)[0] or 1
    pixel_format_size = struct.unpack_from("<I", data, 76)[0]
    fourcc_value = struct.unpack_from("<I", data, 84)[0]
    fourcc = struct.pack("<I", fourcc_value).decode("ascii", errors="replace")
    block_size = {"DXT1": 8, "DXT5": 16}.get(fourcc)
    if header_size != 124:
        raise ValueError(f"unexpected header size {header_size}")
    if pixel_format_size != 32:
        raise ValueError(f"unexpected pixel format size {pixel_format_size}")
    if block_size is None:
        raise ValueError(f"unsupported FourCC {fourcc!r}")

    expected_payload = 0
    level_width = width
    level_height = height
    for _ in range(mipmaps):
        expected_payload += (
            max(1, (level_width + 3) // 4)
            * max(1, (level_height + 3) // 4)
            * block_size
        )
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)
    expected_size = 128 + expected_payload
    if len(data) != expected_size:
        raise ValueError(
            f"payload size {len(data)} does not match header expectation {expected_size}"
        )
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "mipmaps": mipmaps,
        "fourcc": fourcc,
        "size": len(data),
        "expected_size": expected_size,
        "data": data,
    }


def unpack_565(value: int) -> tuple[int, int, int]:
    return (
        (((value >> 11) & 0x1F) * 255) // 31,
        (((value >> 5) & 0x3F) * 255) // 63,
        ((value & 0x1F) * 255) // 31,
    )


def decode_bc3_base(info: dict[str, Any]) -> list[list[tuple[int, int, int, int]]]:
    if info["fourcc"] != "DXT5":
        raise ValueError("alpha inspection requires BC3/DXT5")
    width = int(info["width"])
    height = int(info["height"])
    data = info["data"]
    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    rgba = [[(0, 0, 0, 0) for _ in range(width)] for _ in range(height)]

    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            offset = 128 + (block_y * blocks_x + block_x) * 16
            alpha = data[offset : offset + 8]
            color = data[offset + 8 : offset + 16]
            a0, a1 = alpha[0], alpha[1]
            alpha_indices = int.from_bytes(alpha[2:8], "little")
            alpha_palette = [a0, a1]
            if a0 > a1:
                alpha_palette.extend(
                    ((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)
                )
            else:
                alpha_palette.extend(
                    ((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)
                )
                alpha_palette.extend((0, 255))

            c0 = int.from_bytes(color[0:2], "little")
            c1 = int.from_bytes(color[2:4], "little")
            rgb0 = unpack_565(c0)
            rgb1 = unpack_565(c1)
            color_palette = [
                rgb0,
                rgb1,
                tuple((2 * rgb0[i] + rgb1[i]) // 3 for i in range(3)),
                tuple((rgb0[i] + 2 * rgb1[i]) // 3 for i in range(3)),
            ]
            color_indices = int.from_bytes(color[4:8], "little")

            for local_y in range(4):
                for local_x in range(4):
                    x = block_x * 4 + local_x
                    y = block_y * 4 + local_y
                    if x >= width or y >= height:
                        continue
                    pixel_index = local_y * 4 + local_x
                    alpha_index = (alpha_indices >> (pixel_index * 3)) & 0x7
                    color_index = (color_indices >> (pixel_index * 2)) & 0x3
                    rgb = color_palette[color_index]
                    rgba[y][x] = (*rgb, alpha_palette[alpha_index])
    return rgba


def alpha_profile(info: dict[str, Any]) -> dict[str, float]:
    rgba = decode_bc3_base(info)
    width = int(info["width"])
    height = int(info["height"])
    left = [rgba[y][x][3] for y in range(height) for x in range(width // 8, width // 3)]
    right = [
        rgba[y][x][3]
        for y in range(height)
        for x in range((width * 2) // 3, width - width // 8)
    ]
    return {
        "left_mean": sum(left) / len(left),
        "left_min": float(min(left)),
        "left_max": float(max(left)),
        "right_mean": sum(right) / len(right),
        "right_min": float(min(right)),
        "right_max": float(max(right)),
    }


def rgb_mean(info: dict[str, Any]) -> tuple[float, float, float]:
    rgba = decode_bc3_base(info)
    pixel_count = len(rgba) * len(rgba[0])
    return tuple(
        sum(pixel[channel] for row in rgba for pixel in row)
        / pixel_count
        for channel in range(3)
    )


def run_command(label: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"--- {label} ---")
    rendered_command = shlex.join(command)
    if len(rendered_command) > 1200:
        task_count = (len(command) - 3) // 10
        print(f"{shlex.join(command[:3])} ... ({task_count} batch tasks)")
    else:
        print(rendered_command)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    print(f"exit={result.returncode}")
    return result


def check_outputs(
    paths: list[Path],
    *,
    metal_available: bool,
    require_mipmaps: bool,
    allow_skip_without_metal: bool,
    result: subprocess.CompletedProcess[str],
) -> tuple[str, list[dict[str, Any]]]:
    if result.returncode != 0:
        if allow_skip_without_metal and not metal_available:
            return "SKIP(no Metal/Core Image GPU context)", []
        return f"FAIL(exit {result.returncode})", []

    infos: list[dict[str, Any]] = []
    try:
        for path in paths:
            info = dds_info(path)
            if require_mipmaps and info["mipmaps"] <= 1:
                raise ValueError("GPU run produced only one mip level")
            info.pop("data", None)
            infos.append(info)
    except ValueError as error:
        if allow_skip_without_metal and not metal_available:
            return f"SKIP(no Metal/Core Image GPU context: {error})", []
        return f"FAIL({error})", infos
    return "PASS", infos


def task_args(
    source: Path,
    mask: Path,
    alpha_mask: Path,
    source_4096: Path,
    output: Path,
    index: int,
    high_res_source: Path,
    high_res_upscaled_source: Path,
) -> list[str]:
    # Keep the 64-task contract while exercising only a few large inputs; the
    # remaining tasks keep the fixture run practical on a constrained host.
    if index == 0:
        selected_source = high_res_source
        selected_mask = str(mask)
        contrast, brightness, saturation = "1.0", "0.0", "1.0"
        target_format = "BC3"
    elif index == 1:
        selected_source = high_res_source
        selected_mask = "none"
        contrast, brightness, saturation = "1.15", "0.08", "0.82"
        target_format = "BC3"
    elif index == 2:
        selected_source = high_res_upscaled_source
        selected_mask = str(mask)
        contrast, brightness, saturation = "0.92", "-0.06", "1.18"
        target_format = "BC3"
    elif index == 3:
        selected_source = source_4096
        selected_mask = str(alpha_mask)
        contrast, brightness, saturation = "1.0", "0.0", "1.0"
        target_format = "BC3"
    elif index % 2 == 0:
        selected_source = source
        selected_mask = "none"
        contrast, brightness, saturation = "1.0", "0.0", "1.0"
        target_format = "BC1"
    else:
        selected_source = source
        selected_mask = str(mask)
        contrast, brightness, saturation = "0.92", "-0.06", "1.18"
        target_format = "BC3"
    return [
        str(selected_source),
        selected_mask,
        "1.03",
        "0.97",
        "1.00",
        contrast,
        brightness,
        saturation,
        str(output),
        target_format,
    ]


def make_precision_fixture(directory: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    width, height = 32, 24
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 255) // (width - 1),
                (y * 255) // (height - 1),
                ((x * 3 + y * 5) * 255) // (width * 3 + height * 5 - 8),
            )
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 10, 9), fill=(225, 35, 45))
    draw.ellipse((14, 3, 28, 16), fill=(35, 185, 80))
    draw.line((0, 19, width - 1, 19), fill=(250, 240, 30), width=2)
    source = directory / "precision_source.png"
    reference = directory / "precision_reference.png"
    image.save(source, format="PNG")
    image.resize((width * 2, height * 2), Image.Resampling.NEAREST).save(
        reference, format="PNG"
    )
    return source, reference


def run_precision_ladder(
    args: argparse.Namespace,
    helper: Path,
    artifact_dir: Path,
    tensorops_available: bool,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path | None, bool]:
    from fp8sr_pack import create_fixture, validate_pack

    artifact_dir.mkdir(parents=True, exist_ok=True)
    source, default_reference = make_precision_fixture(artifact_dir)
    reference = args.precision_reference or default_reference
    report: dict[str, Any] = {
        "status": "PASS",
        "source": str(source),
        "reference": str(reference),
        "quality_gate": {
            "psnr_drop_db": QUALITY_PSNR_DROP_DB,
            "error_increase_ratio": QUALITY_ERROR_INCREASE_RATIO,
        },
        "stages": {},
    }
    dtype_args = {
        "Float16": args.fp16_pack,
        "MetalFloat8E4M3": args.fp8_pack,
        "MetalFloat4E2M1": args.fp4_pack,
        "Int2": args.int2_pack,
    }
    packs: dict[str, Path | None] = {}
    for dtype, configured in dtype_args.items():
        if configured is not None:
            packs[dtype] = configured
        elif dtype in ("Float16", "MetalFloat8E4M3"):
            packs[dtype] = create_fixture(artifact_dir / f"pack-{dtype}", dtype)
        else:
            packs[dtype] = None

    coreml_reference_model = getattr(args, "coreml_reference_model", None)
    if coreml_reference_model is not None:
        coreml_output = artifact_dir / "coreml_reference_ladder.png"
        coreml_report = run_coreml_reference(
            coreml_reference_model,
            source,
            coreml_output,
            artifact_dir,
        )
        report["coreml_reference"] = coreml_report
        records.append(
            execution_record(
                backend="coreml",
                role="reference_only",
                status=coreml_report["status"],
                source=source,
                output=coreml_output if coreml_output.is_file() else None,
                result=coreml_report,
            )
        )
        if coreml_report["status"] == "PASS" and args.precision_reference is None:
            reference = coreml_output
            report["reference"] = str(reference)
        elif coreml_report["status"] != "PASS":
            report["status"] = "FAIL(coreml_reference)"
            for dtype in dtype_args:
                report["stages"][dtype] = {"status": "BLOCKED(coreml_reference)"}
            return report, packs["MetalFloat8E4M3"], False

    if not tensorops_available:
        report["status"] = "SKIP(tensorops_unavailable)"
        for dtype in dtype_args:
            stage = "baseline" if dtype == "Float16" else "candidate"
            report["stages"][dtype] = {"status": "BLOCKED(tensorops_unavailable)"}
            records.append(
                execution_record(
                    backend="tensorops",
                    role=stage,
                    dtype=dtype,
                    status="SKIP(tensorops_unavailable)",
                    source=source,
                )
            )
        return report, packs["MetalFloat8E4M3"], True

    baseline_quality: dict[str, Any] | None = None
    blocked_reason: str | None = None
    for dtype in ("Float16", "MetalFloat8E4M3", "MetalFloat4E2M1", "Int2"):
        pack = packs[dtype]
        if blocked_reason is not None:
            status = f"BLOCKED({blocked_reason})"
            report["stages"][dtype] = {"status": status}
            records.append(
                execution_record(
                    backend="tensorops",
                    role="candidate",
                    dtype=dtype,
                    status=status,
                    source=source,
                )
            )
            continue
        if pack is None:
            status = "SKIP(pack_missing)"
            report["stages"][dtype] = {"status": status}
            records.append(
                execution_record(
                    backend="tensorops",
                    role="candidate",
                    dtype=dtype,
                    status=status,
                    source=source,
                )
            )
            if dtype == "MetalFloat4E2M1":
                blocked_reason = "fp4_not_run"
            continue
        try:
            normalized = validate_pack(pack)
        except (OSError, ValueError) as error:
            status = f"FAIL(pack_invalid:{error})"
            report["stages"][dtype] = {"status": status}
            records.append(
                execution_record(
                    backend="tensorops",
                    role="baseline" if dtype == "Float16" else "candidate",
                    dtype=dtype,
                    status="FAIL(pack_invalid)",
                    source=source,
                    result={**pack_record_metadata(pack), "error": str(error)},
                )
            )
            blocked_reason = f"{dtype}_failed"
            report["status"] = "FAIL"
            continue

        output = artifact_dir / f"precision_{dtype}.png"
        result = compare_tensorops_backend(
            helper, pack, source, reference, output, args.compare_runs
        )
        result["dtype"] = dtype
        result.update(pack_record_metadata(pack))
        stage_report: dict[str, Any] = {**result}
        if result["status"] != "PASS":
            stage_report["status"] = result["status"]
            records.append(
                execution_record(
                    backend="tensorops",
                    role="baseline" if dtype == "Float16" else "candidate",
                    dtype=dtype,
                    status=result["status"],
                    source=source,
                    output=output,
                    result=result,
                )
            )
            blocked_reason = f"{dtype}_failed"
            report["status"] = "FAIL"
            report["stages"][dtype] = stage_report
            continue
        if dtype == "Float16":
            baseline_quality = result["quality"]
            stage_report["role"] = "baseline"
        else:
            gate = precision_quality_gate(result["quality"], baseline_quality or {})
            stage_report["gate"] = gate
            if gate["status"] != "PASS":
                stage_report["status"] = "FAIL(quality_gate)"
                blocked_reason = f"{dtype}_quality_gate_failed"
                report["status"] = "FAIL"
        records.append(
            execution_record(
                backend="tensorops",
                role="baseline" if dtype == "Float16" else "candidate",
                dtype=dtype,
                status=stage_report["status"],
                source=source,
                output=output,
                result=stage_report,
            )
        )
        report["stages"][dtype] = stage_report
    return report, packs["MetalFloat8E4M3"], report["status"] == "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--helper", type=Path, default=ROOT / "Utils/mac/ASHelper")
    parser.add_argument("--batch-count", type=int, default=64)
    parser.add_argument("--compare-upscale", action="store_true")
    parser.add_argument(
        "--compare-fp8",
        action="store_true",
        help="compare the deterministic FP8SR fixture or an external --fp8-pack",
    )
    parser.add_argument(
        "--precision-ladder",
        action="store_true",
        help="run FP16 -> FP8 -> FP4 -> INT2, stopping on the first quality failure",
    )
    parser.add_argument("--fp16-pack", type=Path)
    parser.add_argument("--fp8-pack", type=Path)
    parser.add_argument("--fp4-pack", type=Path)
    parser.add_argument("--int2-pack", type=Path)
    parser.add_argument("--precision-reference", type=Path)
    parser.add_argument(
        "--coreml-reference-model",
        type=Path,
        help="optional compiled .mlmodelc used only as the FP8 quality reference",
    )
    parser.add_argument("--compare-runs", type=int, default=5)
    parser.add_argument("--record-jsonl", type=Path)
    parser.add_argument(
        "--gpu-tools",
        action="store_true",
        help="capture one representative TensorOps run with GPU CLI tools",
    )
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    if args.batch_count < 1 or args.batch_count > 128:
        fail("--batch-count must be between 1 and 128")
    if args.compare_runs < 1 or args.compare_runs > 20:
        fail("--compare-runs must be between 1 and 20")
    if args.coreml_reference_model is not None and not args.compare_fp8:
        if not args.precision_ladder:
            fail("--coreml-reference-model requires --compare-fp8 or --precision-ladder")
    if args.precision_reference is not None and not args.precision_ladder:
        fail("--precision-reference requires --precision-ladder")
    if (args.fp16_pack is not None or args.fp4_pack is not None or args.int2_pack is not None) and not args.precision_ladder:
        fail("precision pack options require --precision-ladder")
    if not args.helper.is_file() or not os.access(args.helper, os.X_OK):
        fail(f"ASHelper is not executable: {args.helper}")

    try:
        import PIL  # noqa: F401
    except ImportError as error:
        fail(f"Pillow is required; run ./install_mac.sh or set ORTHO4XP_PYTHON: {error}")

    metal_available, probe_output, host_metal_supported = parse_probe(args.probe)
    metalfx_spatial_available = "metalfx_spatial_available=true" in probe_output.splitlines()
    capability_result = subprocess.run(
        [str(args.helper), "--capabilities"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    capability_output = capability_result.stdout or ""
    tensorops_available = (
        capability_result.returncode == 0
        and (
            "tensorops_available=true" in capability_output.splitlines()
            or "fp8_tensorops_available=true" in capability_output.splitlines()
        )
    )
    fp8_tensorops_available = tensorops_available
    artifact_dir = Path(tempfile.mkdtemp(prefix="ortho4xp-metal-"))
    print(f"artifacts={artifact_dir}")
    overall_ok = True
    report: dict[str, Any] = {
        "helper": str(args.helper),
        "metal_available": metal_available,
        "host_metal_supported": host_metal_supported,
        "metalfx_spatial_available": metalfx_spatial_available,
        "tensorops_available": tensorops_available,
        "fp8_tensorops_available": fp8_tensorops_available,
        "ashelper_capabilities": capability_output,
        "probe": probe_output,
        "batch_count": args.batch_count,
        "compare_runs": args.compare_runs,
        "cases": [],
    }
    records: list[dict[str, Any]] = []
    precision_pack_for_gpu: Path | None = None

    try:
        source, mask, alpha_mask, source_4096, upscale_seed_2048 = make_fixtures(
            artifact_dir
        )
        from PIL import Image

        if args.precision_ladder:
            precision_report, precision_pack_for_gpu, precision_ok = run_precision_ladder(
                args,
                args.helper,
                artifact_dir,
                tensorops_available,
                records,
            )
            report["precision_ladder"] = precision_report
            print(
                "precision ladder="
                f"{precision_report['status']} "
                f"stages={json.dumps(precision_report['stages'], sort_keys=True)}"
            )
            if not precision_ok and tensorops_available:
                overall_ok = False

        if args.gpu_tools:
            if precision_pack_for_gpu is None:
                from fp8sr_pack import create_fixture

                precision_pack_for_gpu = create_fixture(
                    artifact_dir / "gpu-tools-pack", "MetalFloat8E4M3"
                )
            gpu_source = artifact_dir / "gpu-tools-source.png"
            # Keep the representative 512x512 fixture large enough for the
            # capture tool to attach before the short-lived CLI exits.
            with Image.open(source).convert("RGB") as gpu_image:
                gpu_image.save(gpu_source, format="PNG")
            gpu_output = artifact_dir / "gpu-tools-output.png"
            gpu_report = run_gpu_tool_verification(
                args.helper,
                precision_pack_for_gpu,
                gpu_source,
                gpu_output,
                artifact_dir,
            )
            report["gpu_tools"] = gpu_report
            records.append(
                execution_record(
                    backend="tensorops",
                    role="gpu_verification",
                    dtype="MetalFloat8E4M3",
                    status=gpu_report["status"],
                    source=gpu_source,
                    output=gpu_output,
                    gpu_tools=gpu_report,
                )
            )
            print(f"gpu tools={gpu_report['status']}")

        if args.compare_upscale:
            comparison_report: dict[str, Any] = {}
            if not metalfx_spatial_available:
                comparison_report["status"] = "SKIP(metalfx_spatial_unavailable)"
                print("upscale comparison=SKIP(metalfx_spatial_unavailable)")
            else:
                comparison_report["status"] = "PASS"
                transparent_source = artifact_dir / "comparison_transparent_input.png"
                transparent_output = artifact_dir / "comparison_transparent_output.png"
                with Image.open(source).convert("RGBA") as alpha_image:
                    alpha_image.putalpha(128)
                    alpha_image.save(transparent_source, format="PNG")
                alpha_rejection = subprocess.run(
                    [
                        str(args.helper),
                        "--metalfx-spatial-upscale",
                        str(transparent_source),
                        str(transparent_output),
                    ],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                alpha_status = "FAIL"
                alpha_quality: dict[str, Any] = {}
                if alpha_rejection.returncode == 0 and transparent_output.is_file():
                    try:
                        with Image.open(transparent_source).convert("RGBA") as input_image:
                            with Image.open(transparent_output) as output_image:
                                output_rgba = output_image.convert("RGBA")
                                alpha_rgb_mean = tuple(
                                    sum(pixel[channel] for pixel in output_rgba.getdata())
                                    / (output_rgba.width * output_rgba.height)
                                    for channel in range(3)
                                )
                                alpha_range = output_rgba.getchannel("A").getextrema()
                                expected_size = (input_image.width * 2, input_image.height * 2)
                                alpha_quality = {
                                    "size": list(output_rgba.size),
                                    "expected_size": list(expected_size),
                                    "rgb_mean": alpha_rgb_mean,
                                    "alpha_range": alpha_range,
                                }
                                alpha_status = (
                                    "PASS"
                                    if output_rgba.size == expected_size
                                    and alpha_range[0] < 255
                                    and alpha_rgb_mean[0] > 100.0
                                    and alpha_rgb_mean[1] > 45.0
                                    else "FAIL(rgba_split_or_premultiplied)"
                                )
                    except (OSError, ValueError) as error:
                        alpha_quality = {"error": str(error)}
                comparison_report["transparent_input"] = {
                    "status": alpha_status,
                    "exit": alpha_rejection.returncode,
                    "diagnostic": (alpha_rejection.stdout or "").strip(),
                    "quality": alpha_quality,
                }
                print(f"upscale comparison transparent input={alpha_status}")
                records.append(
                    execution_record(
                        backend="metalfx_spatial",
                        role="rgba_comparison",
                        dtype="RGBA8+alpha_bicubic",
                        status=alpha_status,
                        requested_backend="metalfx_spatial",
                        effective_backend=(
                            "metalfx_spatial" if alpha_status == "PASS" else "ci_lanczos"
                        ),
                        source=transparent_source,
                        output=transparent_output,
                        result={
                            "dispatch": "single",
                            "alpha_mode": "rgba_split_bicubic",
                            "diagnostic": (alpha_rejection.stdout or "").strip(),
                            "quality": alpha_quality,
                            "run_exit": alpha_rejection.returncode,
                        },
                    )
                )
                if alpha_status != "PASS":
                    comparison_report["status"] = "FAIL"
                    overall_ok = False
                for target_size in (1024, 4096):
                    comparison_source, comparison_reference = make_upscale_comparison_fixture(
                        artifact_dir, target_size
                    )
                    case_report: dict[str, Any] = {
                        "source": str(comparison_source),
                        "reference": str(comparison_reference),
                    }
                    for backend in ("ci_lanczos", "metalfx_spatial"):
                        backend_output = artifact_dir / (
                            f"comparison_{backend}_{target_size}.png"
                        )
                        result = compare_upscale_backend(
                            args.helper,
                            backend,
                            comparison_source,
                            comparison_reference,
                            backend_output,
                            args.compare_runs,
                        )
                        case_report[backend] = result
                        records.append(
                            execution_record(
                                backend=backend,
                                role="comparison",
                                status=result["status"],
                                source=comparison_source,
                                output=backend_output,
                                result=result,
                            )
                        )
                        print(
                            f"upscale comparison {target_size}px {backend}="
                            f"{result['status']} "
                            f"timing={json.dumps(result.get('timing_ms', {}), sort_keys=True)} "
                            f"quality={json.dumps(result.get('quality', {}), sort_keys=True)}"
                        )
                        if result["status"] != "PASS":
                            comparison_report["status"] = "FAIL"
                            overall_ok = False
                    batch_outputs = [
                        artifact_dir / f"comparison_metalfx_batch_{target_size}_{index}.png"
                        for index in range(2)
                    ]
                    batch_result = compare_metalfx_batch(
                        args.helper,
                        [
                            (comparison_source, batch_outputs[0], comparison_reference),
                            (comparison_source, batch_outputs[1], comparison_reference),
                        ],
                        args.compare_runs,
                    )
                    case_report["metalfx_spatial_batch"] = batch_result
                    records.append(
                        execution_record(
                            backend="metalfx_spatial",
                            role="comparison",
                            dtype="RGBA8",
                            status=batch_result["status"],
                            requested_backend="metalfx_spatial",
                            effective_backend=batch_result.get(
                                "effective_backend", "metalfx_spatial"
                            ),
                            source=comparison_source,
                            output=batch_outputs[0],
                            result=batch_result,
                        )
                    )
                    print(
                        f"upscale comparison {target_size}px metalfx_spatial batch="
                        f"{batch_result['status']} "
                        f"timing={json.dumps(batch_result.get('timing_ms', {}), sort_keys=True)}"
                    )
                    if batch_result["status"] != "PASS":
                        comparison_report["status"] = "FAIL"
                        overall_ok = False
                    comparison_report[str(target_size)] = case_report
            report["upscale_comparison"] = comparison_report

        if args.compare_fp8:
            fp8_report: dict[str, Any] = {}
            if not fp8_tensorops_available:
                fp8_report["status"] = "SKIP(fp8_tensorops_unavailable)"
                print("fp8 comparison=SKIP(fp8_tensorops_unavailable)")
            else:
                from fp8sr_pack import create_fixture, fp8sr_fp16_reference, validate_pack

                fp8_pack = args.fp8_pack
                if fp8_pack is None:
                    fp8_pack = create_fixture(artifact_dir / "fp8sr-fixture")
                try:
                    validate_pack(fp8_pack)
                except (OSError, ValueError) as error:
                    fp8_report["status"] = "FAIL"
                    fp8_report["error"] = str(error)
                    print(f"fp8 comparison=FAIL error={error}")
                    overall_ok = False
                else:
                    fp8_source = artifact_dir / "fp8_source_16.png"
                    fp8_reference = artifact_dir / "fp8_reference_32.png"
                    with Image.open(source).convert("RGB") as small_source:
                        resampling = getattr(Image, "Resampling", Image)
                        small_source.resize((16, 16), resampling.BOX).save(
                            fp8_source, format="PNG"
                        )
                    fp8sr_fp16_reference(fp8_pack, fp8_source, fp8_reference)
                    reference_for_fp8 = fp8_reference
                    coreml_failed = False
                    if args.coreml_reference_model is not None:
                        coreml_report: dict[str, Any] = {
                            "model": str(args.coreml_reference_model),
                        }
                        coreml_helper, compile_detail = build_coreml_reference_helper(
                            artifact_dir
                        )
                        if coreml_helper is None:
                            coreml_report["status"] = "FAIL(compile)"
                            coreml_report["diagnostic"] = compile_detail
                            coreml_failed = True
                        else:
                            coreml_output = artifact_dir / "coreml_reference_32.png"
                            coreml_result = subprocess.run(
                                [
                                    str(coreml_helper),
                                    str(args.coreml_reference_model),
                                    str(fp8_source),
                                    str(coreml_output),
                                ],
                                check=False,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            )
                            coreml_report["exit"] = coreml_result.returncode
                            coreml_report["diagnostic"] = (
                                coreml_result.stdout or ""
                            ).strip()
                            if coreml_result.returncode == 0 and coreml_output.is_file():
                                coreml_report["status"] = "PASS"
                                coreml_report["output"] = str(coreml_output)
                                reference_for_fp8 = coreml_output
                            else:
                                coreml_report["status"] = "FAIL(runtime)"
                                coreml_failed = True
                        fp8_report["coreml_reference"] = coreml_report
                        records.append(
                            execution_record(
                                backend="coreml",
                                role="reference_only",
                                status=coreml_report["status"],
                                source=fp8_source,
                                output=Path(coreml_report["output"])
                                if coreml_report.get("output")
                                else None,
                                result=coreml_report,
                            )
                        )

                    if coreml_failed:
                        fp8_report["status"] = "FAIL(coreml_reference)"
                        print("fp8 comparison=FAIL(coreml_reference)")
                        overall_ok = False
                    else:
                        fp8_output = artifact_dir / "fp8_tensorops_32.png"
                        fp8_result = compare_tensorops_backend(
                            args.helper,
                            fp8_pack,
                            fp8_source,
                            reference_for_fp8,
                            fp8_output,
                            args.compare_runs,
                        )
                        fp8_report.update(fp8_result)
                        fp8_report["reference"] = str(reference_for_fp8)
                        print(
                            "fp8 comparison 32px="
                            f"{fp8_result['status']} "
                            f"timing={json.dumps(fp8_result.get('timing_ms', {}), sort_keys=True)} "
                            f"quality={json.dumps(fp8_result.get('quality', {}), sort_keys=True)}"
                        )
                        if fp8_result["status"] != "PASS":
                            overall_ok = False
                        records.append(
                            execution_record(
                                backend="tensorops",
                                role="comparison",
                                dtype=fp8_result.get("dtype", "MetalFloat8E4M3"),
                                status=fp8_result["status"],
                                source=fp8_source,
                                output=fp8_output,
                                result=fp8_result,
                            )
                        )
            report["fp8_comparison"] = fp8_report

        upscale_cases = [
            ("1024px upscale fixture", source, artifact_dir / "source_upscaled.png", (1024, 1024)),
            (
                "4096px upscale fixture",
                upscale_seed_2048,
                artifact_dir / "source_upscaled_4096.png",
                (4096, 4096),
            ),
        ]
        upscale_report: dict[str, Any] = {}
        for label, upscale_input, upscale_output, expected_size in upscale_cases:
            upscale_result = run_command(
                label,
                [
                    str(args.helper),
                    "--ci-lanczos-upscale",
                    str(upscale_input),
                    str(upscale_output),
                ],
            )
            if upscale_result.returncode != 0 or not upscale_output.is_file():
                upscale_status = (
                    "SKIP(no Metal/Core Image context)"
                    if not metal_available
                    else "FAIL"
                )
                if metal_available:
                    overall_ok = False
            else:
                with Image.open(upscale_output) as upscaled:
                    if upscaled.size != expected_size:
                        upscale_status = f"FAIL(size={upscaled.size})"
                        overall_ok = False
                    else:
                        upscale_status = "PASS"
            upscale_report[label] = {
                "status": upscale_status,
                "input": str(upscale_input),
                "path": str(upscale_output),
                "expected_size": expected_size,
            }
        report["upscale"] = upscale_report
        high_res_source = (
            artifact_dir / "source_upscaled.png"
            if (artifact_dir / "source_upscaled.png").is_file()
            else source
        )
        high_res_upscaled_source = (
            artifact_dir / "source_upscaled_4096.png"
            if (artifact_dir / "source_upscaled_4096.png").is_file()
            else source_4096
        )

        direct_cpu = artifact_dir / "direct_cpu.dds"
        direct_gpu = artifact_dir / "direct_gpu_requested.dds"
        direct_cases = [
            (
                "direct CPU",
                [str(args.helper), "--convert", str(source), str(direct_cpu), "BC3"],
                [direct_cpu],
                False,
                False,
            ),
            (
                "direct GPU requested",
                [
                    str(args.helper),
                    "--convert",
                    str(high_res_source),
                    str(direct_gpu),
                    "BC3",
                    "--gpu",
                ],
                [direct_gpu],
                metal_available,
                False,
            ),
        ]
        for label, command, outputs, require_mips, allow_skip in direct_cases:
            result = run_command(label, command)
            status, infos = check_outputs(
                outputs,
                metal_available=metal_available,
                require_mipmaps=require_mips,
                allow_skip_without_metal=allow_skip,
                result=result,
            )
            print(f"status={status}")
            report["cases"].append({"label": label, "status": status, "outputs": infos})
            overall_ok = overall_ok and status.startswith(("PASS", "SKIP"))

        missing_parent_output = artifact_dir / "missing-parent" / "output.dds"
        write_contract_result = run_command(
            "DDS write error contract",
            [
                str(args.helper),
                "--convert",
                str(source),
                str(missing_parent_output),
                "BC3",
            ],
        )
        if write_contract_result.returncode == 0 and not missing_parent_output.is_file():
            write_contract_status = "FAIL(silent success without output)"
            overall_ok = False
        elif write_contract_result.returncode != 0 and not missing_parent_output.is_file():
            write_contract_status = "PASS"
        else:
            write_contract_status = "FAIL(unexpected output)"
            overall_ok = False
        print(f"status={write_contract_status}")
        report["cases"].append(
            {"label": "DDS write error contract", "status": write_contract_status}
        )

        batch_gpu_outputs = [
            artifact_dir / f"batch_gpu_{index:03d}.dds" for index in range(args.batch_count)
        ]
        gpu_tasks: list[str] = []
        for index in range(args.batch_count):
            gpu_tasks.extend(
                task_args(
                    source,
                    mask,
                    alpha_mask,
                    source_4096,
                    batch_gpu_outputs[index],
                    index,
                    high_res_source,
                    high_res_upscaled_source,
                )
            )

        batch_cases = [
            (
                "batch-v3 GPU",
                [str(args.helper), "--convert-batch-v3", "true", *gpu_tasks],
                batch_gpu_outputs,
                metal_available,
                True,
            ),
        ]
        for label, command, outputs, require_mips, allow_skip in batch_cases:
            result = run_command(label, command)
            status, infos = check_outputs(
                outputs,
                metal_available=metal_available,
                require_mipmaps=require_mips,
                allow_skip_without_metal=allow_skip,
                result=result,
            )
            print(f"status={status}")
            report["cases"].append({"label": label, "status": status, "outputs": infos})
            overall_ok = overall_ok and status.startswith(("PASS", "SKIP"))

        invalid_output = artifact_dir / "batch_invalid_input.dds"
        valid_output = artifact_dir / "batch_valid_with_invalid_sibling.dds"
        invalid_task = [
            str(artifact_dir / "missing-input.png"),
            "none",
            "1.0",
            "1.0",
            "1.0",
            "1.0",
            "0.0",
            "1.0",
            str(invalid_output),
            "BC3",
        ]
        valid_task = task_args(
            source,
            mask,
            alpha_mask,
            source_4096,
            valid_output,
            1,
            high_res_source,
            high_res_upscaled_source,
        )
        invalid_batch_result = run_command(
            "batch-v3 one invalid task",
            [str(args.helper), "--convert-batch-v3", "true", *invalid_task, *valid_task],
        )
        diagnostic_text = invalid_batch_result.stdout or ""
        invalid_batch_status = "PASS"
        if (
            invalid_batch_result.returncode == 0
            or invalid_output.exists()
            or "task=0" not in diagnostic_text
            or "input=" not in diagnostic_text
            or "output=" not in diagnostic_text
        ):
            invalid_batch_status = "FAIL(batch failure/diagnostic contract)"
            overall_ok = False
        elif metal_available:
            if not valid_output.is_file():
                invalid_batch_status = "FAIL(valid sibling output is missing)"
                overall_ok = False
            else:
                try:
                    dds_info(valid_output)
                except ValueError as error:
                    invalid_batch_status = f"FAIL(valid sibling output: {error})"
                    overall_ok = False
        elif valid_output.exists():
            try:
                dds_info(valid_output)
            except ValueError as error:
                invalid_batch_status = f"FAIL(unexpected sibling output: {error})"
                overall_ok = False
        print(f"status={invalid_batch_status}")
        report["cases"].append(
            {
                "label": "batch-v3 one invalid task",
                "status": invalid_batch_status,
                "valid_sibling_output": str(valid_output),
            }
        )

        levels_status = "PASS"
        levels_detail: dict[str, Any] = {}
        try:
            import O4_Imagery_Utils as imagery

            imagery.initialize_color_filters_dict()
            imagery.initialize_providers_dict()
            levels_codes = [
                code
                for code, filters in imagery.color_filters_dict.items()
                if any(item and item[0] == "levels" for item in filters)
            ]
            if not levels_codes:
                raise ValueError("no levels color filter was loaded")
            not_deferred = [
                code
                for code in levels_codes
                if not imagery.gpu_batch_color_filter_supported(code)
            ]
            if not not_deferred:
                raise ValueError("levels filter is still batch-compatible")
            levels_detail = {
                "levels_codes": levels_codes,
                "not_deferred_codes": not_deferred,
                "GeoPunt2012_provider": imagery.providers_dict.get("GeoPunt2012", {}).get(
                    "color_filters"
                ),
                "GeoPunt2012_can_defer": imagery.can_defer_gpu_batch("GeoPunt2012"),
            }
            if "GeoPunt2012" in imagery.providers_dict and imagery.can_defer_gpu_batch(
                "GeoPunt2012"
            ):
                raise ValueError("GeoPunt2012 was incorrectly deferred")
        except Exception as error:
            levels_status = f"FAIL({error})"
            overall_ok = False
        print(f"levels CPU route={levels_status} detail={json.dumps(levels_detail, sort_keys=True)}")
        report["cases"].append(
            {"label": "levels CPU route", "status": levels_status, "detail": levels_detail}
        )

        # A filtered task and the unfiltered direct GPU task use the same
        # source and dimensions. Their decoded base-level means should differ
        # when CIColorControls/CIColorMatrix were actually applied.
        if len(batch_gpu_outputs) > 1 and direct_gpu.is_file() and batch_gpu_outputs[1].is_file():
            raw_mean = rgb_mean(dds_info(direct_gpu))
            filtered_mean = rgb_mean(dds_info(batch_gpu_outputs[1]))
            color_delta = max(
                abs(raw_mean[channel] - filtered_mean[channel]) for channel in range(3)
            )
            color_status = "PASS" if color_delta >= 2.0 else "FAIL(no color effect detected)"
            print(
                "GPU color effect="
                f"{color_status} raw={json.dumps(raw_mean)} "
                f"filtered={json.dumps(filtered_mean)} delta={color_delta:.3f}"
            )
            report["gpu_color_effect"] = {
                "status": color_status,
                "raw_mean": raw_mean,
                "filtered_mean": filtered_mean,
                "max_channel_delta": color_delta,
            }
            if color_status != "PASS" and metal_available:
                overall_ok = False

        alpha_mask_output = artifact_dir / "batch_gpu_alpha_mask.dds"
        alpha_mask_result = run_command(
            "batch-v3 GPU alpha-channel control",
            [
                str(args.helper),
                "--convert-batch-v3",
                "true",
                str(high_res_source),
                str(alpha_mask),
                "1.0",
                "1.0",
                "1.0",
                "1.0",
                "0.0",
                "1.0",
                str(alpha_mask_output),
                "BC3",
            ],
        )
        alpha_status, alpha_infos = check_outputs(
            [alpha_mask_output],
            metal_available=metal_available,
            require_mipmaps=metal_available,
            allow_skip_without_metal=True,
            result=alpha_mask_result,
        )
        alpha_profile_result: dict[str, Any] | None = None
        if alpha_mask_output.is_file():
            try:
                alpha_profile_result = alpha_profile(dds_info(alpha_mask_output))
                if not (
                    alpha_profile_result["left_mean"] <= 96
                    and alpha_profile_result["right_mean"] >= 192
                ):
                    alpha_status = "FAIL(alpha channel control profile)"
                    if metal_available:
                        overall_ok = False
            except (ValueError, IndexError) as error:
                alpha_status = f"FAIL({error})"
                if metal_available:
                    overall_ok = False
        print(
            "alpha-channel control="
            f"{alpha_status} profile={json.dumps(alpha_profile_result, sort_keys=True)}"
        )
        report["cases"].append(
            {
                "label": "batch-v3 GPU alpha-channel control",
                "status": alpha_status,
                "outputs": alpha_infos,
                "alpha_profile": alpha_profile_result,
            }
        )
        overall_ok = overall_ok and alpha_status.startswith(("PASS", "SKIP"))

        # The first task deliberately uses a low-resolution mask against the
        # upscaled source. Inspect the decoded BC3 alpha away from the edge.
        for label, output in (("batch-v3 GPU", batch_gpu_outputs[0]),):
            mask_status = "FAIL"
            if not output.is_file():
                continue
            try:
                profile = alpha_profile(dds_info(output))
            except (ValueError, IndexError) as error:
                profile = {"error": str(error)}
            else:
                mask_passed = (
                    profile["left_mean"] <= 64
                    and profile["right_mean"] >= 192
                )
                mask_status = "PASS" if mask_passed else "FAIL(grayscale mask profile)"
                if not mask_passed and metal_available:
                    overall_ok = False
            print(f"{label} mask alpha={json.dumps(profile, sort_keys=True)}")
            report.setdefault("mask_alpha", {})[label] = profile
            report.setdefault("mask_alpha_checks", {})[label] = {
                "status": mask_status,
                "profile": profile,
            }

        for label, output in (("batch-v3 GPU", batch_gpu_outputs[0]),):
            if not output.is_file():
                continue
            try:
                high_res_info = dds_info(output)
                expected_dimensions = (1024, 1024)
                actual_dimensions = (
                    high_res_info["width"],
                    high_res_info["height"],
                )
                if actual_dimensions != expected_dimensions:
                    raise ValueError(
                        f"expected high-resolution DDS {expected_dimensions}, "
                        f"got {actual_dimensions}"
                    )
                report.setdefault("high_resolution_mask_case", {})[label] = "PASS"
            except (ValueError, IndexError) as error:
                report.setdefault("high_resolution_mask_case", {})[label] = str(error)
                if metal_available:
                    overall_ok = False

        if not any(record.get("backend") == "coreml" for record in records):
            records.append(
                execution_record(
                    backend="coreml",
                    role="reference_only",
                    status="SKIP(reference_not_requested)",
                )
            )
        write_execution_records(args.record_jsonl, records)
        report["execution_record_count"] = len(records)
        report["record_jsonl"] = str(args.record_jsonl) if args.record_jsonl else None
        report_path = artifact_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"report={report_path}")
        if overall_ok:
            if metal_available:
                print("PASS Metal verification environment")
            else:
                print("PASS CPU/fallback checks; Metal assertions were skipped because no device is available")
        else:
            print("FAIL Metal verification environment")
            print(f"kept_artifacts={artifact_dir}")
            return 1
        if args.keep_artifacts:
            print(f"kept_artifacts={artifact_dir}")
        return 0
    finally:
        if not args.keep_artifacts and overall_ok and artifact_dir.exists():
            shutil.rmtree(artifact_dir)


if __name__ == "__main__":
    sys.exit(main())
