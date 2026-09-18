#!/usr/bin/env python3
"""Compare CPU GSI raster placement with the explicit ASHelper Metal PoC."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import O4_GSI_DEM_Utils as GSI  # noqa: E402
from O4_ASHelper_Server import ASHelperJSONLServer  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _usage_delta(before, after) -> dict[str, float]:
    return {
        "user_s": max(0.0, after.ru_utime - before.ru_utime),
        "sys_s": max(0.0, after.ru_stime - before.ru_stime),
    }


def _measure(callable_):
    before_self = resource.getrusage(resource.RUSAGE_SELF)
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    result = callable_()
    wall = time.perf_counter() - started
    after_self = resource.getrusage(resource.RUSAGE_SELF)
    after_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_usage = _usage_delta(before_self, after_self)
    child_usage = _usage_delta(before_children, after_children)
    return result, {
        "wall_s": wall,
        "user_s": self_usage["user_s"] + child_usage["user_s"],
        "sys_s": self_usage["sys_s"] + child_usage["sys_s"],
    }


def _metal_geometry(block: GSI.GSIBlock, region: GSI.GSIRegion, resolution: float):
    source_south, source_west, source_north, source_east = block.bounds
    dlon = (source_east - source_west) / block.values.shape[1]
    dlat = (source_north - source_south) / block.values.shape[0]
    return {
        "x_base": float(
            (region.west + 0.5 * resolution - source_west) / dlon - 0.5
        ),
        "y_base": float(
            (source_north - (region.north - 0.5 * resolution)) / dlat - 0.5
        ),
        "x_step": float(resolution / dlon),
        "y_step": float(resolution / dlat),
    }


def _merge_block(output: np.ndarray, sampled: np.ndarray) -> None:
    good = np.isfinite(sampled)
    output[good] = sampled[good]


def _cpu_result(blocks, region, resolution):
    raster, resolution_deg = GSI._grid_for_region(region, resolution)
    for block in blocks:
        GSI._insert_block(raster, block, region, resolution_deg)
    return raster


def _gpu_runner(server, blocks, region, resolution, workdir, width, height):
    tasks = []
    output_paths = []
    for index, block in enumerate(blocks):
        input_path = workdir / f"source-{index}.raw"
        output_path = workdir / f"target-{index}.raw"
        np.asarray(block.values, dtype=np.float32, order="C").tofile(input_path)
        output_paths.append(output_path)
        geometry = _metal_geometry(block, region, resolution)
        tasks.append(
            {
                "id": f"gsi-{index}",
                "input": str(input_path),
                "output": str(output_path),
                "width": int(width),
                "height": int(height),
                "stride": int(width * np.dtype(np.float32).itemsize),
                "source_width": int(block.values.shape[1]),
                "source_height": int(block.values.shape[0]),
                "source_stride": int(block.values.shape[1]),
                "target_stride": int(width),
                **geometry,
            }
        )
    response = server.gsi_raster_batch(tasks)
    results = response.get("results") or []
    if len(results) != len(blocks) or not all(item.get("ok") for item in results):
        error = next(
            (item.get("error") for item in results if not item.get("ok")),
            "invalid_gsi_raster_response",
        )
        raise RuntimeError(str(error))
    raster = np.full((height, width), np.nan, dtype=np.float32)
    for output_path in output_paths:
        sampled = np.fromfile(output_path, dtype=np.float32)
        if sampled.size != width * height:
            raise RuntimeError("Metal output shape mismatch")
        _merge_block(raster, sampled.reshape((height, width)))
    return raster, results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mesh-code", required=True)
    parser.add_argument("--ashelper", type=Path, default=ROOT / "Utils/mac/ASHelper")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--record-json", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be >= 0 and --iterations must be positive")

    record = {
        "status": "FAIL",
        "backend": "metal",
        "dispatch": "gsi_raster",
        "input_dir": str(args.input_dir.resolve()),
        "mesh_code": args.mesh_code,
        "ashelper": str(args.ashelper.resolve()),
        "warmup": args.warmup,
        "iterations": args.iterations,
    }

    if not args.ashelper.is_file() or not os.access(args.ashelper, os.X_OK):
        record.update({"status": "SKIP", "reason": "ashelper_missing"})
        return _finish(record, args.record_json)

    try:
        scan = GSI._scan_result_from_catalog(args.input_dir)
        region = GSI._region_for_mesh(args.mesh_code)
        entries = GSI._candidate_entries(scan, region)
        if not entries:
            raise RuntimeError("no_candidate_archives")
        GSI._verify_build_entries(entries, args.input_dir, None, None)
        blocks = []
        for entry in entries:
            blocks.extend(
                GSI._archive_blocks(
                    args.input_dir / entry["path"], region, None
                )
            )
        if not blocks:
            raise RuntimeError("no_overlapping_blocks")
        resolution_name, arcsec = GSI._target_resolution(blocks, "auto")
        del resolution_name
        resolution_deg = arcsec / 3600.0
        height, width = GSI._grid_for_region(region, arcsec)[0].shape
        record.update(
            {
                "block_count": len(blocks),
                "shape": [height, width],
                "resolution_deg": resolution_deg,
            }
        )

        for _ in range(args.warmup):
            _cpu_result(blocks, region, arcsec)

        cpu_times = []
        cpu_raster = None
        for _ in range(args.iterations):
            cpu_raster, timing = _measure(
                lambda: _cpu_result(blocks, region, arcsec)
            )
            cpu_times.append(timing)

        with tempfile.TemporaryDirectory(prefix="gsi-metal-") as temporary:
            workdir = Path(temporary)
            with ASHelperJSONLServer(str(args.ashelper)) as server:
                for _ in range(args.warmup):
                    _gpu_runner(
                        server, blocks, region, resolution_deg, workdir, width, height
                    )
                metal_times = []
                metal_raster = None
                metal_results = []
                for _ in range(args.iterations):
                    (metal_raster, metal_results), timing = _measure(
                        lambda: _gpu_runner(
                            server, blocks, region, resolution_deg, workdir, width, height
                        )
                    )
                    metal_times.append(timing)

        if not np.array_equal(
            np.isfinite(cpu_raster), np.isfinite(metal_raster)
        ):
            raise RuntimeError("valid_mask_mismatch")
        finite = np.isfinite(cpu_raster) & np.isfinite(metal_raster)
        max_error = float(
            np.max(np.abs(cpu_raster[finite] - metal_raster[finite]))
        ) if np.any(finite) else 0.0
        if max_error > 1e-5:
            raise RuntimeError(f"value_mismatch:{max_error}")

        record.update(
            {
                "status": "PASS",
                "valid_cells": int(np.isfinite(cpu_raster).sum()),
                "max_abs_error": max_error,
                "cpu": {
                    "median_wall_s": float(np.median([item["wall_s"] for item in cpu_times])),
                    "p95_wall_s": _percentile([item["wall_s"] for item in cpu_times], 0.95),
                    "samples": cpu_times,
                },
                "metal": {
                    "median_wall_s": float(np.median([item["wall_s"] for item in metal_times])),
                    "p95_wall_s": _percentile([item["wall_s"] for item in metal_times], 0.95),
                    "samples": metal_times,
                    "results": metal_results,
                },
            }
        )
    except RuntimeError as error:
        if str(error) in {"unavailable_device", "unavailable_command_queue", "unavailable_raster_pipelines"}:
            record.update({"status": "SKIP", "reason": str(error)})
        else:
            record.update({"status": "FAIL", "reason": str(error)})

    return _finish(record, args.record_json)


def _finish(record: dict, record_path: Path | None) -> int:
    text = json.dumps(record, ensure_ascii=False, indent=2)
    print(text)
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(text + "\n", encoding="utf-8")
    return 0 if record["status"] in {"PASS", "SKIP"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
