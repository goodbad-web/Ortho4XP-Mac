import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Config_Utils as CFG  # noqa: E402
import O4_DSF_Utils as DSF  # noqa: E402
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Tile_Utils as TILE  # noqa: E402


def _write_dxt1_dds(path, width=4, height=4, mipmaps=None):
    mipmaps = mipmaps or max(width, height).bit_length()
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 28, mipmaps)
    struct.pack_into("<I", header, 76, 32)
    header[84:88] = b"DXT1"

    # Keep the calculation explicit for non-square and non-4-aligned fixtures.
    payload_size = 0
    level_width = width
    level_height = height
    for _ in range(mipmaps):
        payload_size += (
            max(1, (level_width + 3) // 4)
            * max(1, (level_height + 3) // 4)
            * 8
        )
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)
    path.write_bytes(bytes(header) + bytes(payload_size))


def test_auto_format_selects_the_smallest_xplane_compatible_bc_format():
    assert IMG.resolve_dds_format("AUTO", has_alpha=False) == "BC1"
    assert IMG.resolve_dds_format("AUTO", has_alpha=True) == "BC3"
    assert IMG.resolve_dds_format("BC1", has_alpha=True) == "BC3"
    assert IMG.resolve_dds_format("BC3", has_alpha=False) == "BC3"


def test_masked_xp11_contract_uses_the_actual_mask_alpha():
    tile = SimpleNamespace(imprint_masks_to_dds=True, dds_format="AUTO")
    partial_mask = Image.new("L", (4, 4), color=255)
    partial_mask.putpixel((0, 0), 0)
    opaque_mask = Image.new("L", (4, 4), color=255)

    assert DSF._masked_dds_requires_alpha(tile, partial_mask)
    assert not DSF._masked_dds_requires_alpha(tile, opaque_mask)

    tile.dds_format = "BC1"
    assert DSF._masked_dds_requires_alpha(tile, opaque_mask)


def test_non_water_texture_contract_uses_shared_mask_alpha(monkeypatch):
    tile = SimpleNamespace(imprint_masks_to_dds=True, dds_format="AUTO")
    partial_mask = Image.new("L", (4, 4), color=255)
    partial_mask.putpixel((0, 0), 0)

    monkeypatch.setattr(
        DSF.MASK,
        "needs_mask",
        lambda _tile, *_attributes: partial_mask,
    )

    assert DSF._texture_contract_has_alpha(tile, (1, 2, 16, "BI"))

    tile.imprint_masks_to_dds = False
    assert not DSF._texture_contract_has_alpha(tile, (1, 2, 16, "BI"))


def test_upscale_scope_only_targets_explicit_airport_texture_keys():
    tile = SimpleNamespace(
        upscale_backend="tensorops",
        upscale_scope="airport",
        airport_highres_texture_keys=frozenset({(10, 20, 16, "prov")}),
    )

    assert IMG.should_upscale_texture(tile, 10, 20, 16, "prov")
    assert not IMG.should_upscale_texture(tile, 11, 20, 16, "prov")
    assert IMG.expected_texture_dimensions(tile, 10, 20, 16, "prov") == (8192, 8192)
    assert IMG.expected_texture_dimensions(tile, 11, 20, 16, "prov") == (4096, 4096)

    tile.upscale_scope = "none"
    assert not IMG.should_upscale_texture(tile, 10, 20, 16, "prov")


def test_scope_and_auto_format_are_exposed_by_configuration():
    assert "AUTO" in CFG.cfg_vars["dds_format"]["values"]
    assert "upscale_scope" in CFG.list_dsf_vars
    assert CFG.cfg_vars["upscale_scope"]["values"] == ("none", "all", "airport")


def test_validate_dds_requires_a_full_mipmap_chain_when_requested(tmp_path):
    full = tmp_path / "full.dds"
    _write_dxt1_dds(full)
    valid, error = IMG.validate_dds_file(
        full,
        expected_format="BC1",
        expected_dimensions=(4, 4),
        require_mipmaps=True,
    )
    assert valid, error
    assert IMG.read_dds_dimensions(full) == (4, 4)

    partial = tmp_path / "partial.dds"
    _write_dxt1_dds(partial, mipmaps=2)
    valid, error = IMG.validate_dds_file(partial, require_mipmaps=True)
    assert not valid
    assert "full chain" in error


def test_terrain_load_center_uses_the_final_dds_dimension(tmp_path):
    textures = tmp_path / "textures"
    terrain = tmp_path / "terrain"
    textures.mkdir()
    terrain.mkdir()
    dds_name = "0_0_test16.dds"
    _write_dxt1_dds(textures / dds_name)
    terrain_file = terrain / "0_0_test16.ter"
    terrain_file.write_text(
        "LOAD_CENTER 1 2 123 4096\n"
        f"BASE_TEX_NOWRAP ../textures/{dds_name}\n",
        encoding="utf-8",
    )

    count = TILE._sync_terrain_load_centers(SimpleNamespace(build_dir=str(tmp_path)))

    assert count == 1
    assert terrain_file.read_text(encoding="utf-8").startswith(
        "LOAD_CENTER 1 2 123 4\n"
    )
