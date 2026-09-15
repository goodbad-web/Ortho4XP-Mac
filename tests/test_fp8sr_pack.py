import json
import sys
from pathlib import Path

import pytest


TOOLS_ROOT = Path(__file__).parents[1] / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from fp8sr_pack import (  # noqa: E402
    FP8SRPackError,
    create_fixture,
    fp8sr_fp16_reference,
    validate_pack,
)


def test_fixture_is_valid_and_has_the_fixed_fp8sr_graph(tmp_path):
    pack = create_fixture(tmp_path / "fixture")

    normalized = validate_pack(pack)

    assert normalized["manifest"]["weight_dtype"] == "MetalFloat8E4M3"
    assert normalized["manifest"]["activation_dtype"] == "Float16"
    assert normalized["manifest"]["accumulation_dtype"] == "Float16"
    assert [layer["name"] for layer in normalized["layers"]] == [
        "conv0",
        "conv1",
        "conv2",
    ]
    assert [layer["k_padded"] for layer in normalized["layers"]] == [32, 288, 288]


def test_fixture_has_deterministic_identity_center_weights(tmp_path):
    pack = create_fixture(tmp_path / "fixture")
    conv0 = (pack / "conv0.fp8").read_bytes()

    # E4M3 1.0 is 0x38.  The three RGB center weights are stored in the
    # first three output-channel bytes of their 128-byte rows.
    assert conv0[12 * 128 : 12 * 128 + 3] == bytes((0x38, 0, 0))
    assert conv0[13 * 128 : 13 * 128 + 3] == bytes((0, 0x38, 0))
    assert conv0[14 * 128 : 14 * 128 + 3] == bytes((0, 0, 0x38))


def test_fp16_reference_matches_fixture_pixel_shuffle_contract(tmp_path):
    from PIL import Image

    pack = create_fixture(tmp_path / "fixture")
    source = tmp_path / "input.png"
    Image.new("RGB", (2, 2), (20, 40, 60)).save(source)
    reference = tmp_path / "reference.png"

    fp8sr_fp16_reference(pack, source, reference)

    with Image.open(reference) as image:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0)) == (20, 40, 60)
        assert image.getpixel((1, 0)) == (0, 0, 0)
        assert image.getpixel((0, 1)) == (0, 0, 0)


def test_rejects_weight_size_and_path_escape(tmp_path):
    pack = create_fixture(tmp_path / "fixture")
    (pack / "conv1.fp8").write_bytes((pack / "conv1.fp8").read_bytes()[:-1])
    with pytest.raises(FP8SRPackError, match="conv1: weights size"):
        validate_pack(pack)

    pack = create_fixture(tmp_path / "escape")
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"][0]["weights"] = "../outside.fp8"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FP8SRPackError, match="escapes"):
        validate_pack(pack)


def test_rejects_nonfinite_scale(tmp_path):
    pack = create_fixture(tmp_path / "fixture")
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"][1]["scale"] = "nan"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FP8SRPackError, match="scale"):
        validate_pack(pack)


def test_rejects_nonfinite_raw_fp16_weight(tmp_path):
    pack = create_fixture(tmp_path / "fixture", "Float16")
    weights_path = pack / "conv0.f16w"
    weights = bytearray(weights_path.read_bytes())
    # conv0's center feature is 12; 0x7c00 is FP16 +infinity.
    weights[12 * 128 : 12 * 128 + 2] = b"\x00\x7c"
    weights_path.write_bytes(weights)

    with pytest.raises(FP8SRPackError, match="non-finite"):
        validate_pack(pack)
