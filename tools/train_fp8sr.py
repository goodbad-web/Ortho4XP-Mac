#!/usr/bin/env python3
"""Train, export, quantize, and verify the fixed Ortho4XP FP8SR graph.

Training dependencies are intentionally optional.  Normal Ortho4XP startup
does not import this module's PyTorch path; install requirements-train.txt
only for the development pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageStat

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
VERIFY_ROOT = REPO_ROOT / "Utils" / "run"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from fp8sr_pack import (  # noqa: E402
    EXPECTED_LAYERS,
    FP8SRPackError,
    decode_fp8_e4m3,
    encode_fp8_e4m3,
    fp8sr_fp16_reference,
    validate_pack,
    write_pack,
)


FP8_MAX_FINITE = 448.0
DEFAULT_PATCH_SIZE = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class TrainingError(RuntimeError):
    """Raised for invalid training data or missing optional dependencies."""


@dataclass(frozen=True)
class ImagePair:
    lr: Path
    hr: Path


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise TrainingError(
            "PyTorch is required for training/export. "
            "Install the optional dependencies with: "
            "python -m pip install -r requirements-train.txt"
        ) from error
    return torch, nn, functional


def _require_safetensors() -> Any:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise TrainingError(
            "safetensors is required for checkpoint I/O. "
            "Install requirements-train.txt."
        ) from error
    return load_file, save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if hasattr(torch, "mps") and mps_backend is not None and mps_backend.is_available():
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if callable(manual_seed):
            manual_seed(seed)


def _select_device(requested: str, torch: Any) -> Any:
    if requested not in {"auto", "mps", "cpu", "cuda"}:
        raise TrainingError(f"unsupported device: {requested}")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = mps_backend is not None and mps_backend.is_available()
    if requested in {"auto", "mps"} and mps_available:
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise TrainingError("CUDA was requested but is unavailable")
    if requested == "mps":
        print("training_device=mps_unavailable fallback=cpu", file=sys.stderr)
    return torch.device("cpu")


def list_pairs(dataset_root: str | Path, split: str) -> list[ImagePair]:
    root = Path(dataset_root).expanduser().resolve()
    lr_root = root / split / "lr"
    hr_root = root / split / "hr"
    if not lr_root.is_dir() or not hr_root.is_dir():
        raise TrainingError(
            f"dataset must contain {split}/lr and {split}/hr: {root}"
        )
    pairs: list[ImagePair] = []
    for lr_path in sorted(path for path in lr_root.rglob("*") if path.is_file()):
        if lr_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        relative = lr_path.relative_to(lr_root)
        hr_path = hr_root / relative
        if not hr_path.is_file():
            raise TrainingError(f"missing HR pair for {split}/{relative}")
        with Image.open(lr_path) as lr_image, Image.open(hr_path) as hr_image:
            if lr_image.size[0] * 2 != hr_image.size[0] or lr_image.size[1] * 2 != hr_image.size[1]:
                raise TrainingError(
                    f"pair must be exactly 2x: LR={lr_path} {lr_image.size}, "
                    f"HR={hr_path} {hr_image.size}"
                )
        pairs.append(ImagePair(lr_path, hr_path))
    if not pairs:
        raise TrainingError(f"no image pairs found under {lr_root}")
    return pairs


def _image_tensor(image: Image.Image, torch: Any) -> Any:
    import numpy as np

    data = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(data).permute(2, 0, 1).contiguous()


def _paired_dataset(torch: Any, pairs: list[ImagePair], patch_size: int, training: bool) -> Any:
    class PairedDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.pairs = pairs

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index: int) -> tuple[Any, Any, str]:
            pair = self.pairs[index]
            with Image.open(pair.lr) as lr_source, Image.open(pair.hr) as hr_source:
                lr_image = lr_source.convert("RGB")
                hr_image = hr_source.convert("RGB")
            if training:
                if lr_image.width < patch_size or lr_image.height < patch_size:
                    raise TrainingError(
                        f"LR image is smaller than patch_size={patch_size}: {pair.lr}"
                    )
                left = random.randint(0, lr_image.width - patch_size)
                top = random.randint(0, lr_image.height - patch_size)
                lr_image = lr_image.crop((left, top, left + patch_size, top + patch_size))
                hr_image = hr_image.crop(
                    (left * 2, top * 2, left * 2 + patch_size * 2, top * 2 + patch_size * 2)
                )
                if random.random() < 0.5:
                    lr_image = lr_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    hr_image = hr_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    lr_image = lr_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    hr_image = hr_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                rotation = random.randrange(4)
                if rotation:
                    angle = rotation * 90
                    lr_image = lr_image.rotate(angle, expand=True)
                    hr_image = hr_image.rotate(angle, expand=True)
            return _image_tensor(lr_image, torch), _image_tensor(hr_image, torch), str(pair.lr)

    return PairedDataset()


def build_model() -> Any:
    torch, nn, functional = _require_torch()

    class FP8SRModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv0 = nn.Conv2d(3, 32, 3, padding=0)
            self.conv1 = nn.Conv2d(32, 32, 3, padding=0)
            self.conv2 = nn.Conv2d(32, 12, 3, padding=0)
            self.shuffle = nn.PixelShuffle(2)

        @staticmethod
        def _edge_conv(layer: Any, value: Any) -> Any:
            value = functional.pad(value, (1, 1, 1, 1), mode="replicate")
            return layer(value)

        def forward(self, value: Any) -> Any:
            value = torch.relu(self._edge_conv(self.conv0, value))
            value = torch.relu(self._edge_conv(self.conv1, value))
            value = torch.relu(self._edge_conv(self.conv2, value))
            return self.shuffle(value)

    return FP8SRModel()


def _tensor_metrics(output: Any, target: Any) -> dict[str, float]:
    difference = (output.float() - target.float()) * 255.0
    mae = float(difference.abs().mean().item())
    rmse = float(difference.square().mean().sqrt().item())
    psnr = float("inf") if rmse == 0 else 20.0 * math.log10(255.0 / rmse)
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr}


def _image_metrics(output: Path, reference: Path) -> dict[str, float]:
    with Image.open(output).convert("RGB") as output_image, Image.open(reference).convert("RGB") as reference_image:
        if output_image.size != reference_image.size:
            raise TrainingError(
                f"output size {output_image.size} != reference size {reference_image.size}"
            )
        difference = ImageChops.difference(output_image, reference_image)
        stats = ImageStat.Stat(difference)
        mae_rgb = [float(value) for value in stats.mean]
        rmse_rgb = [float(value) for value in stats.rms]
        mae = sum(mae_rgb) / 3.0
        rmse = math.sqrt(sum(value * value for value in rmse_rgb) / 3.0)
        psnr = float("inf") if rmse == 0 else 20.0 * math.log10(255.0 / rmse)
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr}


def _aggregate_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        raise TrainingError("no validation metrics were produced")
    rmse = math.sqrt(sum(value["rmse"] ** 2 for value in values) / len(values))
    return {
        "mae": sum(value["mae"] for value in values) / len(values),
        "rmse": rmse,
        "psnr_db": float("inf") if rmse == 0 else 20.0 * math.log10(255.0 / rmse),
    }


def _save_checkpoint(model: Any, checkpoint: Path, metadata: dict[str, Any]) -> None:
    _, save_file = _require_safetensors()
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(
        tensors,
        str(checkpoint),
        metadata={"fp8sr_metadata": json.dumps(_json_safe(metadata), sort_keys=True)},
    )
    _write_json(checkpoint.with_suffix(".json"), metadata)


def _load_checkpoint(checkpoint: Path, device: Any) -> Any:
    load_file, _ = _require_safetensors()
    model = build_model()
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    torch, _, _ = _require_torch()
    model.to(device=device, dtype=torch.float16)
    return model


def _evaluate_model(model: Any, pairs: list[ImagePair], device: Any, torch: Any) -> dict[str, float]:
    dataset = _paired_dataset(torch, pairs, DEFAULT_PATCH_SIZE, training=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    values: list[dict[str, float]] = []
    losses: list[float] = []
    with torch.no_grad():
        for lr, hr, _ in loader:
            lr = lr.to(device=device, dtype=torch.float16)
            hr = hr.to(device=device, dtype=torch.float16)
            prediction = model(lr)
            clipped = prediction.float().clamp(0.0, 1.0)
            losses.append(float(torch.nn.functional.l1_loss(clipped, hr.float()).item()))
            values.append(_tensor_metrics(clipped, hr))
    result = _aggregate_metrics(values)
    result["l1"] = sum(losses) / len(losses)
    return result


def train(args: argparse.Namespace) -> Path:
    torch, _, _ = _require_torch()
    _set_seed(args.seed, torch)
    device = _select_device(args.device, torch)
    train_pairs = list_pairs(args.dataset, "train")
    val_pairs = list_pairs(args.dataset, "val")
    train_dataset = _paired_dataset(torch, train_pairs, args.patch_size, training=True)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    model = build_model().to(device=device, dtype=torch.float16)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(loader))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_l1 = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        train_losses: list[float] = []
        for lr, hr, _ in loader:
            lr = lr.to(device=device, dtype=torch.float16)
            hr = hr.to(device=device, dtype=torch.float16)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr)
            loss = torch.nn.functional.l1_loss(prediction.float().clamp(0.0, 1.0), hr.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
        validation = _evaluate_model(model, val_pairs, device, torch)
        entry = {
            "epoch": epoch + 1,
            "train_l1": sum(train_losses) / len(train_losses),
            "validation": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False, sort_keys=True))
        if validation["l1"] < best_l1:
            best_l1 = validation["l1"]
            metadata = {
                "format": "FP8SR-training-checkpoint",
                "version": 1,
                "model": "3x3:3->32->32->12+PixelShuffle2",
                "input_layout": "NCHW",
                "pixel_range": "RGB-sRGB-[0,1]",
                "device": str(device),
                "seed": args.seed,
                "dataset": str(Path(args.dataset).expanduser().resolve()),
                "train_pairs": len(train_pairs),
                "validation_pairs": len(val_pairs),
                "patch_size": args.patch_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "loss": "L1",
                "best_epoch": epoch + 1,
                "best_validation": validation,
            }
            _save_checkpoint(model, output_dir / "fp16_best.safetensors", metadata)
    _write_json(output_dir / "training_history.json", history)
    if not (output_dir / "fp16_best.safetensors").is_file():
        raise TrainingError("training completed without a best checkpoint")
    print(f"fp16_checkpoint={output_dir / 'fp16_best.safetensors'}")
    return output_dir / "fp16_best.safetensors"


def _checkpoint_layers(checkpoint: Path, device: str = "cpu") -> tuple[Any, list[dict[str, object]]]:
    torch, _, _ = _require_torch()
    load_file, _ = _require_safetensors()
    state = load_file(str(checkpoint), device=device)
    layers: list[dict[str, object]] = []
    for name, kernel, in_channels, out_channels in EXPECTED_LAYERS:
        weight = state[f"{name}.weight"]
        bias = state[f"{name}.bias"]
        expected = (out_channels, in_channels, kernel, kernel)
        if tuple(int(dimension) for dimension in weight.shape) != expected:
            raise TrainingError(f"{name} weight shape is {tuple(weight.shape)}, expected {expected}")
        if tuple(int(dimension) for dimension in bias.shape) != (out_channels,):
            raise TrainingError(f"{name} bias shape is {tuple(bias.shape)}, expected {(out_channels,)}")
        layers.append({"name": name, "weight": weight, "bias": bias, "scale": 1.0})
    return torch, layers


def export_fp16(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise TrainingError(f"checkpoint is missing: {checkpoint}")
    _, layers = _checkpoint_layers(checkpoint)
    pack = write_pack(
        args.output_pack,
        layers,
        "Float16",
        metadata={"source_checkpoint_sha256": _sha256(checkpoint), "export": "fp16"},
    )
    print(f"fp16_pack={pack}")
    return pack


def quantize_fp8(args: argparse.Namespace) -> Path:
    torch, layers = _checkpoint_layers(Path(args.checkpoint).expanduser().resolve())
    quantized: list[dict[str, object]] = []
    report: dict[str, Any] = {"dtype": "MetalFloat8E4M3", "layers": []}
    for layer in layers:
        name = str(layer["name"])
        original = layer["weight"].float()
        maximum = float(original.abs().max().item())
        scale = maximum / FP8_MAX_FINITE if maximum else 1.0
        scaled = original / scale
        flattened = scaled.reshape(-1).tolist()
        encoded = [encode_fp8_e4m3(float(value)) for value in flattened]
        restored = torch.tensor(
            [decode_fp8_e4m3(code) * scale for code in encoded], dtype=torch.float32
        ).reshape(original.shape)
        error = restored - original
        report["layers"].append(
            {
                "name": name,
                "scale": scale,
                "max_abs": maximum,
                "weight_mae": float(error.abs().mean().item()),
                "weight_rmse": float(error.square().mean().sqrt().item()),
            }
        )
        quantized.append(
            {
                "name": name,
                "weight": scaled,
                "bias": layer["bias"],
                "scale": scale,
            }
        )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    pack = write_pack(
        args.output_pack,
        quantized,
        "MetalFloat8E4M3",
        metadata={
            "source_checkpoint_sha256": _sha256(checkpoint),
            "quantization": "symmetric_per_layer_ptq",
            "fp8_max_finite": FP8_MAX_FINITE,
        },
    )
    report["pack"] = str(pack)
    report["manifest_sha256"] = _sha256(pack / "manifest.json")
    _write_json(pack / "quantization.json", report)
    print(f"fp8_pack={pack}")
    return pack


def _lanczos_output(lr: Path, target_size: tuple[int, int], destination: Path) -> None:
    with Image.open(lr).convert("RGB") as source:
        source.resize(target_size, Image.Resampling.LANCZOS).save(destination, format="PNG")


def _gate_fp16_against_lanczos(fp16: dict[str, float], lanczos: dict[str, float]) -> dict[str, Any]:
    return {
        "status": "PASS"
        if fp16["psnr_db"] >= lanczos["psnr_db"]
        and fp16["mae"] <= lanczos["mae"]
        and fp16["rmse"] <= lanczos["rmse"]
        else "FAIL",
        "baseline": lanczos,
        "candidate": fp16,
        "checks": {
            "psnr_not_below_lanczos": fp16["psnr_db"] >= lanczos["psnr_db"],
            "mae_not_above_lanczos": fp16["mae"] <= lanczos["mae"],
            "rmse_not_above_lanczos": fp16["rmse"] <= lanczos["rmse"],
        },
    }


def _fp8_gate(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    psnr_pass = (
        math.isinf(baseline["psnr_db"])
        and math.isinf(candidate["psnr_db"])
    ) or (
        not math.isinf(baseline["psnr_db"])
        and candidate["psnr_db"] >= baseline["psnr_db"] - 0.25
    )
    mae_pass = candidate["mae"] <= baseline["mae"] * 1.05 if baseline["mae"] else candidate["mae"] == 0
    rmse_pass = candidate["rmse"] <= baseline["rmse"] * 1.05 if baseline["rmse"] else candidate["rmse"] == 0
    return {
        "status": "PASS" if psnr_pass and mae_pass and rmse_pass else "FAIL",
        "limits": {"psnr_drop_db": 0.25, "error_increase_ratio": 0.05},
        "baseline": baseline,
        "candidate": candidate,
        "checks": {"psnr": psnr_pass, "mae": mae_pass, "rmse": rmse_pass},
    }


def _record(
    backend: str,
    role: str,
    dtype: str,
    status: str,
    source: Path | None,
    output: Path | None,
    quality: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    pack: Path | None = None,
    gpu_tools: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if quality is not None:
        result["quality"] = quality
    if gate is not None:
        result["gate"] = gate
    if pack is not None:
        result["pack"] = str(pack)
        result["manifest_sha256"] = _sha256(pack / "manifest.json")
        manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
        result["pack_version"] = manifest.get("version")
    record = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": backend,
        "role": role,
        "dtype": dtype,
        "requested_backend": backend,
        "effective_backend": backend,
        "status": status,
        "fallback_reason": None,
        "exit_code": 0 if status.startswith("PASS") else None,
        **result,
    }
    if source is not None:
        record["input"] = str(source)
        with Image.open(source) as image:
            record["input_size"] = list(image.size)
    if output is not None:
        record["output"] = str(output)
        with Image.open(output) as image:
            record["output_size"] = list(image.size)
    if gpu_tools is not None:
        record["gpu_tools"] = gpu_tools
        record["gpu_evidence_paths"] = [
            str(path)
            for path in gpu_tools.get("gpu_evidence_paths", [])
        ]
        record["tensorops_dispatch_observed"] = bool(
            gpu_tools.get("tensorops_dispatch_observed", False)
        )
        record["neural_accelerator_confirmed"] = False
    return record


def verify(args: argparse.Namespace) -> int:
    from verify_metal import run_gpu_tool_verification

    dataset = Path(args.dataset).expanduser().resolve()
    val_pairs = list_pairs(dataset, "val")
    fp16_pack = Path(args.fp16_pack).expanduser().resolve()
    fp8_pack = Path(args.fp8_pack).expanduser().resolve() if args.fp8_pack else None
    validate_pack(fp16_pack)
    if fp8_pack is not None:
        validate_pack(fp8_pack)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representative_dir = output_dir / "representatives"
    representative_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    fp16_values: list[dict[str, float]] = []
    fp8_values: list[dict[str, float]] = []
    lanczos_values: list[dict[str, float]] = []
    for index, pair in enumerate(val_pairs):
        with Image.open(pair.hr) as hr_image:
            target_size = hr_image.size
        stem = f"{index:04d}_{pair.lr.stem}"
        lanczos_path = representative_dir / f"{stem}_lanczos.png"
        fp16_path = representative_dir / f"{stem}_fp16.png"
        _lanczos_output(pair.lr, target_size, lanczos_path)
        fp8sr_fp16_reference(fp16_pack, pair.lr, fp16_path)
        lanczos_quality = _image_metrics(lanczos_path, pair.hr)
        fp16_quality = _image_metrics(fp16_path, pair.hr)
        lanczos_values.append(lanczos_quality)
        fp16_values.append(fp16_quality)
        if index < args.representative_count:
            shutil.copy2(pair.lr, representative_dir / f"{stem}_lr{pair.lr.suffix.lower()}")
            shutil.copy2(pair.hr, representative_dir / f"{stem}_hr{pair.hr.suffix.lower()}")
        if fp8_pack is not None:
            fp8_path = representative_dir / f"{stem}_fp8.png"
            fp8sr_fp16_reference(fp8_pack, pair.lr, fp8_path)
            fp8_values.append(_image_metrics(fp8_path, pair.hr))

    lanczos_quality = _aggregate_metrics(lanczos_values)
    fp16_quality = _aggregate_metrics(fp16_values)
    fp16_gate = _gate_fp16_against_lanczos(fp16_quality, lanczos_quality)
    first_stem = f"0000_{val_pairs[0].lr.stem}"
    first_lanczos = representative_dir / f"{first_stem}_lanczos.png"
    first_fp16 = representative_dir / f"{first_stem}_fp16.png"
    first_fp8 = representative_dir / f"{first_stem}_fp8.png"
    records.append(
        _record(
            "ci_lanczos",
            "reference",
            "Float32",
            "PASS",
            val_pairs[0].lr,
            first_lanczos,
            lanczos_quality,
        )
    )
    records.append(
        _record(
            "tensorops",
            "baseline",
            "Float16",
            "PASS" if fp16_gate["status"] == "PASS" else "FAIL(fp16_gate)",
            val_pairs[0].lr,
            first_fp16,
            fp16_quality,
            fp16_gate,
            fp16_pack,
        )
    )
    report: dict[str, Any] = {
        "status": "PASS" if fp16_gate["status"] == "PASS" else "FAIL(fp16_gate)",
        "dataset": str(dataset),
        "validation_pairs": len(val_pairs),
        "lanczos": lanczos_quality,
        "fp16": fp16_quality,
        "fp16_gate": fp16_gate,
        "stages": {"FP16": {"status": fp16_gate["status"], "gate": fp16_gate}},
    }
    if fp16_gate["status"] != "PASS":
        if fp8_pack is not None:
            report["stages"]["FP8"] = {"status": "BLOCKED(fp16_gate)"}
            records.append(
                _record(
                    "tensorops",
                    "candidate",
                    "MetalFloat8E4M3",
                    "BLOCKED(fp16_gate)",
                    val_pairs[0].lr,
                    first_fp8 if first_fp8.is_file() else None,
                    pack=fp8_pack,
                )
            )
    elif fp8_pack is not None:
        fp8_quality = _aggregate_metrics(fp8_values)
        fp8_gate = _fp8_gate(fp8_quality, fp16_quality)
        report["fp8"] = fp8_quality
        report["fp8_gate"] = fp8_gate
        report["stages"]["FP8"] = {"status": fp8_gate["status"], "gate": fp8_gate}
        records.append(
            _record(
                "tensorops",
                "candidate",
                "MetalFloat8E4M3",
                fp8_gate["status"],
                val_pairs[0].lr,
                first_fp8,
                fp8_quality,
                fp8_gate,
                fp8_pack,
            )
        )
        if fp8_gate["status"] == "PASS" and args.helper and sys.platform == "darwin":
            helper = Path(args.helper).expanduser().resolve()
            representative = val_pairs[0]
            with Image.open(representative.hr) as hr_image:
                target_size = hr_image.size
            gpu_output = output_dir / "gpu_representative_fp8.png"
            from verify_metal import compare_tensorops_backend

            gpu_result = compare_tensorops_backend(
                helper,
                fp8_pack,
                representative.lr,
                representative.hr,
                gpu_output,
                args.compare_runs,
            )
            diagnostic = str(gpu_result.get("diagnostic", ""))
            if gpu_result.get("status") != "PASS" and "metal_unavailable" in diagnostic:
                gpu_result["status"] = "SKIP(metal_unavailable)"
            dispatch_observed = bool(gpu_result.get("tensorops_dispatch_observed"))
            gpu_result["tensorops_dispatch_observed"] = dispatch_observed
            report["gpu"] = gpu_result
            records.append(_record("tensorops", "candidate", "MetalFloat8E4M3", gpu_result.get("status", "FAIL"), representative.lr, gpu_output if gpu_output.is_file() else None, gpu_result.get("quality"), fp8_gate, fp8_pack, {"tensorops_dispatch_observed": dispatch_observed}))
            if args.gpu_tools:
                gpu_tools = run_gpu_tool_verification(
                    helper,
                    fp8_pack,
                    representative.lr,
                    output_dir / "gpu_capture_output.png",
                    output_dir / "gpu-evidence",
                )
                report["gpu_tools"] = gpu_tools
                if gpu_tools.get("status", "").startswith("PASS"):
                    report["status"] = "PASS" if gpu_tools.get("tensorops_dispatch_observed") else "FAIL(gpu_evidence)"
                elif not gpu_tools.get("status", "").startswith("SKIP"):
                    report["status"] = "FAIL(gpu_evidence)"
        elif fp8_gate["status"] != "PASS":
            report["status"] = "FAIL(fp8_gate)"
    record_path = Path(args.record_jsonl).expanduser().resolve() if args.record_jsonl else output_dir / "execution.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    report["record_jsonl"] = str(record_path)
    _write_json(output_dir / "verification.json", report)
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


def _add_common_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    _add_common_training_arguments(train_parser)
    train_parser.set_defaults(handler=train)
    export_parser = subparsers.add_parser("export-fp16")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output-pack", type=Path, required=True)
    export_parser.set_defaults(handler=export_fp16)
    quantize_parser = subparsers.add_parser("quantize-fp8")
    quantize_parser.add_argument("--checkpoint", type=Path, required=True)
    quantize_parser.add_argument("--output-pack", type=Path, required=True)
    quantize_parser.set_defaults(handler=quantize_fp8)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--dataset", type=Path, required=True)
    verify_parser.add_argument("--fp16-pack", type=Path, required=True)
    verify_parser.add_argument("--fp8-pack", type=Path)
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument("--record-jsonl", type=Path)
    verify_parser.add_argument("--helper", type=Path, default=REPO_ROOT / "Utils/mac/ASHelper")
    verify_parser.add_argument("--compare-runs", type=int, default=3)
    verify_parser.add_argument("--representative-count", type=int, default=1)
    verify_parser.add_argument("--gpu-tools", action="store_true")
    verify_parser.set_defaults(handler=verify)
    run_parser = subparsers.add_parser("run")
    _add_common_training_arguments(run_parser)
    run_parser.add_argument("--record-jsonl", type=Path)
    run_parser.add_argument("--compare-runs", type=int, default=3)
    run_parser.add_argument("--gpu-tools", action="store_true")
    run_parser.set_defaults(handler=None)
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    checkpoint = train(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    fp16_pack = export_fp16(
        argparse.Namespace(checkpoint=checkpoint, output_pack=output_dir / "fp16-pack")
    )
    baseline_args = argparse.Namespace(
        dataset=args.dataset,
        fp16_pack=fp16_pack,
        fp8_pack=None,
        output_dir=output_dir / "verification-fp16",
        record_jsonl=None,
        helper=args.helper if hasattr(args, "helper") else REPO_ROOT / "Utils/mac/ASHelper",
        compare_runs=args.compare_runs,
        representative_count=1,
        gpu_tools=False,
    )
    if verify(baseline_args) != 0:
        print("FP8 quantization blocked: FP16 quality gate failed", file=sys.stderr)
        return 1
    fp8_pack = quantize_fp8(
        argparse.Namespace(checkpoint=checkpoint, output_pack=output_dir / "fp8-pack")
    )
    final_args = argparse.Namespace(
        dataset=args.dataset,
        fp16_pack=fp16_pack,
        fp8_pack=fp8_pack,
        output_dir=output_dir / "verification",
        record_jsonl=args.record_jsonl,
        helper=args.helper if hasattr(args, "helper") else REPO_ROOT / "Utils/mac/ASHelper",
        compare_runs=args.compare_runs,
        representative_count=1,
        gpu_tools=args.gpu_tools,
    )
    return verify(final_args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "run":
            return run_pipeline(args)
        result = args.handler(args)
        return result if isinstance(result, int) else 0
    except (TrainingError, FP8SRPackError, OSError, ValueError) as error:
        print(f"train_fp8sr: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
