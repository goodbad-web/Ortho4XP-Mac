#!/usr/bin/env python3
"""Validate and create small external FP8SR packs for Ortho4XP.

The repository intentionally does not contain model weights.  This utility is
also used by the deterministic Metal verification runner to create a tiny
known-good pack in a temporary directory.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


class FP8SRPackError(ValueError):
    """Raised when an external FP8SR pack violates the runtime contract."""


EXPECTED_LAYERS = (
    ("conv0", 3, 3, 32),
    ("conv1", 3, 32, 32),
    ("conv2", 3, 32, 12),
)
ROW_STRIDE_BYTES = 128
SUPPORTED_WEIGHT_DTYPES = (
    "Float16",
    "MetalFloat8E4M3",
    "MetalFloat4E2M1",
    "Int2",
)
LEGACY_VERSION = 1
VARIANT_VERSION = 2


def _load_manifest(pack: Path) -> dict:
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file():
        raise FP8SRPackError("manifest.json is missing")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise FP8SRPackError(f"manifest.json is invalid: {error}") from error
    if not isinstance(manifest, dict):
        raise FP8SRPackError("manifest.json must contain an object")
    return manifest


def _relative_file(pack: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FP8SRPackError(f"{field} must be a relative file path")
    path = (pack / value).resolve()
    root = pack.resolve()
    if path != root and root not in path.parents:
        raise FP8SRPackError(f"{field} escapes the pack directory")
    return path


def validate_pack(pack_path: str | Path) -> dict:
    """Validate a pack and return normalized layer metadata."""
    pack = Path(pack_path).expanduser().resolve()
    if not pack.is_dir():
        raise FP8SRPackError("FP8SR pack directory is missing")
    manifest = _load_manifest(pack)
    exact_values = {
        "format": "FP8SR",
        "upscale_factor": 2,
        "layout": "NHWC",
        "input_channels": 3,
        "output_channels": 3,
        "activation_dtype": "Float16",
        "accumulation_dtype": "Float16",
        "weight_row_stride_bytes": ROW_STRIDE_BYTES,
    }
    if manifest.get("version") not in (LEGACY_VERSION, VARIANT_VERSION):
        raise FP8SRPackError(
            f"manifest.version must be {LEGACY_VERSION} or {VARIANT_VERSION}, "
            f"got {manifest.get('version')!r}"
        )
    weight_dtype = manifest.get("weight_dtype")
    if weight_dtype not in SUPPORTED_WEIGHT_DTYPES:
        raise FP8SRPackError(
            f"manifest.weight_dtype must be one of {SUPPORTED_WEIGHT_DTYPES!r}, "
            f"got {weight_dtype!r}"
        )
    if manifest.get("version") == LEGACY_VERSION and weight_dtype != "MetalFloat8E4M3":
        raise FP8SRPackError("version 1 packs must use MetalFloat8E4M3 weights")
    for key, expected in exact_values.items():
        if manifest.get(key) != expected:
            raise FP8SRPackError(
                f"manifest.{key} must be {expected!r}, got {manifest.get(key)!r}"
            )

    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != len(EXPECTED_LAYERS):
        raise FP8SRPackError("manifest.layers must contain exactly three layers")

    normalized_layers = []
    for entry, expected in zip(layers, EXPECTED_LAYERS):
        if not isinstance(entry, dict):
            raise FP8SRPackError("each layer must be an object")
        name, kernel, in_channels, out_channels = expected
        for key, value in {
            "name": name,
            "kernel": kernel,
            "in_channels": in_channels,
            "out_channels": out_channels,
        }.items():
            if entry.get(key) != value:
                raise FP8SRPackError(
                    f"layer {name}: {key} must be {value!r}, got {entry.get(key)!r}"
                )
        k_elements = kernel * kernel * in_channels
        k_padded = ((k_elements + 31) // 32) * 32
        out_padded = ((out_channels + 31) // 32) * 32
        if k_padded % 32 or out_padded % 32:
            raise FP8SRPackError(f"layer {name}: tensor dimensions are not 32-aligned")
        weight_path = _relative_file(pack, entry.get("weights"), f"layer {name}.weights")
        bias_path = _relative_file(pack, entry.get("bias"), f"layer {name}.bias")
        if not weight_path.is_file() or weight_path.stat().st_size != k_padded * ROW_STRIDE_BYTES:
            actual = weight_path.stat().st_size if weight_path.is_file() else "missing"
            raise FP8SRPackError(
                f"layer {name}: weights size is {actual}, expected {k_padded * ROW_STRIDE_BYTES}"
            )
        weight_data = weight_path.read_bytes()
        _validate_finite_weights(
            weight_data,
            k_padded,
            out_padded,
            str(weight_dtype),
            name,
        )
        if not bias_path.is_file() or bias_path.stat().st_size != out_padded * 2:
            actual = bias_path.stat().st_size if bias_path.is_file() else "missing"
            raise FP8SRPackError(
                f"layer {name}: bias size is {actual}, expected {out_padded * 2}"
            )
        scale = entry.get("scale")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise FP8SRPackError(f"layer {name}: scale must be a finite positive number")
        normalized_layers.append(
            {
                "name": name,
                "kernel": kernel,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "k_padded": k_padded,
                "out_padded": out_padded,
                "weights": weight_path,
                "bias": bias_path,
                "scale": float(scale),
            }
        )
    return {
        "pack": pack,
        "manifest": manifest,
        "version": int(manifest["version"]),
        "weight_dtype": str(weight_dtype),
        "layers": normalized_layers,
    }


def decode_fp8_e4m3(value: int) -> float:
    """Decode Apple's finite E4M3 representation for reference tests."""
    value &= 0xFF
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 3) & 0x0F
    mantissa = value & 0x07
    if exponent == 0:
        return sign * mantissa * (2.0 ** -6)
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


_FP8_FINITE_VALUES = tuple(
    (code, decode_fp8_e4m3(code))
    for code in range(256)
    if math.isfinite(decode_fp8_e4m3(code))
)


def encode_fp8_e4m3(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("FP8 fixture values must be finite")
    return min(_FP8_FINITE_VALUES, key=lambda pair: abs(pair[1] - value))[0]


def decode_fp4_e2m1(value: int) -> float:
    """Decode the packed Metal FP4 E2M1 value used by MSL."""
    value &= 0x0F
    sign = -1.0 if value & 0x08 else 1.0
    exponent = (value >> 1) & 0x03
    mantissa = value & 0x01
    if exponent == 0:
        return sign * mantissa * 0.5
    return sign * (1.0 + mantissa * 0.5) * (2.0 ** (exponent - 1))


_FP4_FINITE_VALUES = tuple(
    (code, decode_fp4_e2m1(code)) for code in range(16)
)


def encode_fp4_e2m1(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("FP4 fixture values must be finite")
    return min(_FP4_FINITE_VALUES, key=lambda pair: abs(pair[1] - value))[0]


def decode_int2(value: int) -> int:
    value &= 0x03
    return value - 4 if value & 0x02 else value


def encode_int2(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("INT2 fixture values must be finite")
    return min(range(4), key=lambda code: abs(decode_int2(code) - value))


def _weight_bytes_per_element(weight_dtype: str) -> float:
    return {
        "Float16": 2.0,
        "MetalFloat8E4M3": 1.0,
        "MetalFloat4E2M1": 0.5,
        "Int2": 0.25,
    }[weight_dtype]


def _read_weight(weights: bytes, feature: int, output_channel: int, weight_dtype: str) -> float:
    row = feature * ROW_STRIDE_BYTES
    if weight_dtype == "Float16":
        return struct.unpack_from("<e", weights, row + output_channel * 2)[0]
    if weight_dtype == "MetalFloat8E4M3":
        return decode_fp8_e4m3(weights[row + output_channel])
    if weight_dtype == "MetalFloat4E2M1":
        packed = weights[row + output_channel // 2]
        return decode_fp4_e2m1(packed >> 4 if output_channel & 1 else packed)
    packed = weights[row + output_channel // 4]
    return float(decode_int2((packed >> ((output_channel % 4) * 2)) & 0x03))


def _validate_finite_weights(
    weights: bytes,
    k_padded: int,
    out_padded: int,
    weight_dtype: str,
    layer_name: str,
) -> None:
    for feature in range(k_padded):
        for output_channel in range(out_padded):
            value = _read_weight(weights, feature, output_channel, weight_dtype)
            if not math.isfinite(value):
                raise FP8SRPackError(
                    f"layer {layer_name}: weights contain a non-finite value "
                    f"at feature={feature}, output_channel={output_channel}"
                )


def _round_fp16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def fp8sr_fp16_reference(pack_path: str | Path, input_path: str | Path, output_path: str | Path) -> Path:
    """Run any supported FP8SR weight variant with FP16-rounded arithmetic.

    This is intentionally a small verification reference, not a production
    inference path. It makes the same RGB/NHWC, edge-clamped 3x3 graph and
    PixelShuffle2 decisions as the Metal implementation.
    """
    from PIL import Image

    normalized = validate_pack(pack_path)
    weight_dtype = normalized["weight_dtype"]
    with Image.open(input_path).convert("RGB") as source:
        width, height = source.size
        source_pixels = source.load()
        previous = [
            [_round_fp16(source_pixels[x, y][channel] / 255.0) for channel in range(3)]
            for y in range(height)
            for x in range(width)
        ]

    for layer in normalized["layers"]:
        kernel = layer["kernel"]
        in_channels = layer["in_channels"]
        out_channels = layer["out_channels"]
        weights = layer["weights"].read_bytes()
        bias_data = layer["bias"].read_bytes()
        biases = [
            _round_fp16(struct.unpack_from("<e", bias_data, channel * 2)[0])
            for channel in range(out_channels)
        ]
        output = []
        radius = kernel // 2
        for y in range(height):
            for x in range(width):
                pixel_values = [_round_fp16(0.0) for _ in range(32)]
                for output_channel in range(out_channels):
                    accumulator = _round_fp16(0.0)
                    feature = 0
                    for ky in range(kernel):
                        sample_y = min(height - 1, max(0, y + ky - radius))
                        for kx in range(kernel):
                            sample_x = min(width - 1, max(0, x + kx - radius))
                            source_pixel = sample_y * width + sample_x
                            for input_channel in range(in_channels):
                                activation = previous[source_pixel][input_channel]
                                weight = _round_fp16(
                                    _read_weight(weights, feature, output_channel, weight_dtype)
                                )
                                accumulator = _round_fp16(
                                    accumulator + _round_fp16(activation * weight)
                                )
                                feature += 1
                    value = _round_fp16(accumulator * layer["scale"] + biases[output_channel])
                    pixel_values[output_channel] = _round_fp16(max(0.0, value))
                output.append(pixel_values)
        previous = output

    output_image = Image.new("RGB", (width * 2, height * 2))
    output_pixels = output_image.load()
    for pixel, values in enumerate(previous):
        x = pixel % width
        y = pixel // width
        for dy in range(2):
            for dx in range(2):
                subpixel = (dy * 2 + dx) * 3
                rgb = list(output_pixels[x * 2 + dx, y * 2 + dy])
                for channel in range(3):
                    value = min(1.0, max(0.0, values[subpixel + channel]))
                    rgb[channel] = int(round(value * 255.0))
                output_pixels[x * 2 + dx, y * 2 + dy] = tuple(rgb)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(destination, format="PNG")
    return destination


def _write_layer(
    pack: Path,
    name: str,
    kernel: int,
    in_channels: int,
    out_channels: int,
    weight_dtype: str,
) -> None:
    k_padded = ((kernel * kernel * in_channels + 31) // 32) * 32
    out_padded = ((out_channels + 31) // 32) * 32
    weights = bytearray(k_padded * ROW_STRIDE_BYTES)
    center = (kernel // 2) * kernel * in_channels + (kernel // 2) * in_channels
    for output_channel in range(out_channels):
        for input_channel in range(in_channels):
            value = 0.0
            if name == "conv0" and output_channel < 3 and input_channel == output_channel:
                value = 1.0
            elif name == "conv1" and output_channel == input_channel:
                value = 1.0
            elif name == "conv2" and input_channel == output_channel % in_channels:
                value = 1.0
            feature = center + input_channel
            if weight_dtype == "Float16":
                struct.pack_into("<e", weights, feature * ROW_STRIDE_BYTES + output_channel * 2, value)
            elif weight_dtype == "MetalFloat8E4M3":
                weights[feature * ROW_STRIDE_BYTES + output_channel] = encode_fp8_e4m3(value)
            elif weight_dtype == "MetalFloat4E2M1":
                code = encode_fp4_e2m1(value)
                offset = feature * ROW_STRIDE_BYTES + output_channel // 2
                shift = (output_channel % 2) * 4
                weights[offset] |= code << shift
            elif weight_dtype == "Int2":
                code = encode_int2(value)
                offset = feature * ROW_STRIDE_BYTES + output_channel // 4
                shift = (output_channel % 4) * 2
                weights[offset] |= code << shift
    extension = {
        "Float16": "f16w",
        "MetalFloat8E4M3": "fp8",
        "MetalFloat4E2M1": "fp4",
        "Int2": "int2",
    }[weight_dtype]
    (pack / f"{name}.{extension}").write_bytes(weights)
    (pack / f"{name}.f16").write_bytes(b"".join(struct.pack("<e", 0.0) for _ in range(out_padded)))


def create_fixture(
    pack_path: str | Path,
    weight_dtype: str = "MetalFloat8E4M3",
) -> Path:
    """Create a deterministic, tiny FP8SR pack outside the repository."""
    if weight_dtype not in SUPPORTED_WEIGHT_DTYPES:
        raise FP8SRPackError(f"unsupported fixture dtype {weight_dtype!r}")
    pack = Path(pack_path).expanduser().resolve()
    pack.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "FP8SR",
        "version": LEGACY_VERSION if weight_dtype == "MetalFloat8E4M3" else VARIANT_VERSION,
        "upscale_factor": 2,
        "layout": "NHWC",
        "input_channels": 3,
        "output_channels": 3,
        "weight_dtype": weight_dtype,
        "activation_dtype": "Float16",
        "accumulation_dtype": "Float16",
        "weight_row_stride_bytes": ROW_STRIDE_BYTES,
        "layers": [],
    }
    for name, kernel, in_channels, out_channels in EXPECTED_LAYERS:
        _write_layer(pack, name, kernel, in_channels, out_channels, weight_dtype)
        extension = {
            "Float16": "f16w",
            "MetalFloat8E4M3": "fp8",
            "MetalFloat4E2M1": "fp4",
            "Int2": "int2",
        }[weight_dtype]
        manifest["layers"].append(
            {
                "name": name,
                "kernel": kernel,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "weights": f"{name}.{extension}",
                "bias": f"{name}.f16",
                "scale": 1.0,
            }
        )
    (pack / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    validate_pack(pack)
    return pack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", type=Path, metavar="PACK")
    group.add_argument("--create-fixture", type=Path, metavar="PACK")
    parser.add_argument(
        "--dtype",
        choices=SUPPORTED_WEIGHT_DTYPES,
        default="MetalFloat8E4M3",
        help="weight dtype for --create-fixture (variant packs use version 2)",
    )
    args = parser.parse_args()
    try:
        if args.validate:
            normalized = validate_pack(args.validate)
            print(f"FP8SR valid: {normalized['pack']}")
        else:
            print(f"FP8SR fixture: {create_fixture(args.create_fixture, args.dtype)}")
    except (FP8SRPackError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
