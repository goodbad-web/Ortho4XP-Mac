#!/usr/bin/env python3
"""Run deterministic ASHelper/Metal checks without changing the repository."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


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


def make_fixtures(directory: Path) -> tuple[Path, Path, Path]:
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
    return source, mask_path, alpha_mask_path


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
    source: Path, mask: Path, output: Path, index: int, high_res_source: Path
) -> list[str]:
    selected_source = high_res_source if index == 0 else source
    if index % 3 == 0:
        selected_mask = str(mask)
        contrast, brightness, saturation = "1.0", "0.0", "1.0"
    elif index % 3 == 1:
        selected_mask = "none"
        contrast, brightness, saturation = "1.15", "0.08", "0.82"
    else:
        selected_mask = str(mask)
        contrast, brightness, saturation = "0.92", "-0.06", "1.18"
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
        "BC3",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--helper", type=Path, default=ROOT / "Utils/mac/ASHelper")
    parser.add_argument("--batch-count", type=int, default=64)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    if args.batch_count < 1 or args.batch_count > 128:
        fail("--batch-count must be between 1 and 128")
    if not args.helper.is_file() or not os.access(args.helper, os.X_OK):
        fail(f"ASHelper is not executable: {args.helper}")

    try:
        import PIL  # noqa: F401
    except ImportError as error:
        fail(f"Pillow is required; run ./install_mac.sh or set ORTHO4XP_PYTHON: {error}")

    metal_available, probe_output, host_metal_supported = parse_probe(args.probe)
    artifact_dir = Path(tempfile.mkdtemp(prefix="ortho4xp-metal-"))
    print(f"artifacts={artifact_dir}")
    overall_ok = True
    report: dict[str, Any] = {
        "helper": str(args.helper),
        "metal_available": metal_available,
        "host_metal_supported": host_metal_supported,
        "probe": probe_output,
        "batch_count": args.batch_count,
        "cases": [],
    }

    try:
        source, mask, alpha_mask = make_fixtures(artifact_dir)
        upscale_path = artifact_dir / "source_upscaled.png"
        upscale_result = run_command(
            "upscale fixture",
            [str(args.helper), "--upscale", str(source), str(upscale_path)],
        )
        upscale_status = "PASS"
        if upscale_result.returncode != 0 or not upscale_path.is_file():
            upscale_status = "SKIP(no Metal/Core Image context)" if not metal_available else "FAIL"
            overall_ok = overall_ok and not metal_available
        else:
            from PIL import Image

            with Image.open(upscale_path) as upscaled:
                if upscaled.size != (1024, 1024):
                    upscale_status = f"FAIL(size={upscaled.size})"
                    overall_ok = False
        report["upscale"] = {"status": upscale_status, "path": str(upscale_path)}
        high_res_source = upscale_path if upscale_path.is_file() else source

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
                    str(source),
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
            gpu_tasks.extend(task_args(source, mask, batch_gpu_outputs[index], index, high_res_source))

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

        if upscale_path.is_file():
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
