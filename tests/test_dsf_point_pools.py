import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_DSF_Utils as DSF  # noqa: E402


def _node_coords(lons, lats, altitudes=None):
    altitudes = altitudes if altitudes is not None else np.zeros(len(lons))
    coords = np.zeros(5 * len(lons), dtype=np.float64)
    coords[0::5] = lons
    coords[1::5] = lats
    coords[2::5] = altitudes
    return coords


def test_integer_point_codes_match_legacy_binary_representation():
    fractions = np.array([0.0, 0.000001, 0.25, 0.5, 0.999999, 1.0])
    coords = _node_coords(133 + fractions, 34 + fractions)

    x_codes, y_codes = DSF._integer_point_codes(coords, 133, 34)
    expected = np.array(
        [int(DSF.float2qquad(value), 2) for value in fractions],
        dtype=np.uint32,
    )

    np.testing.assert_array_equal(x_codes, expected)
    np.testing.assert_array_equal(y_codes, expected)


def test_integer_pool_partition_assigns_each_node_once():
    rng = np.random.default_rng(1234)
    fractions_x = rng.random(500)
    fractions_y = rng.random(500)
    coords = _node_coords(133 + fractions_x, 34 + fractions_y)
    x_codes, y_codes = DSF._integer_point_codes(coords, 133, 34)

    pool_ids, pool_nodes, prefixes = DSF._build_integer_point_pools(
        x_codes,
        y_codes,
        bucket_size=25,
    )

    assert pool_ids.shape == (500,)
    assert len(pool_nodes) == len(prefixes)
    assert all(len(nodes) <= 25 for nodes in pool_nodes)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(pool_nodes)), np.arange(500, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        pool_ids[np.concatenate(pool_nodes)],
        np.concatenate(
            [np.full(len(nodes), index, dtype=np.int32)
             for index, nodes in enumerate(pool_nodes)]
        ),
    )

    for nodes, (level, prefix_x, prefix_y) in zip(pool_nodes, prefixes):
        shift = 24 - level
        np.testing.assert_array_equal(x_codes[nodes] >> shift, prefix_x)
        np.testing.assert_array_equal(y_codes[nodes] >> shift, prefix_y)


def test_pool_local_coordinates_match_legacy_bit_slices():
    codes = np.array([0, 1, 0x123456, 0xFFFFFF], dtype=np.uint32)
    for level in (3, 8, 9, 16):
        expected = np.array(
            [
                int(DSF.float2qquad(int(code) / float(1 << 24))[level:level + 16] or "0", 2)
                for code in codes
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(
            DSF._pool_local_coordinates(codes, level), expected
        )


def test_precomputed_triangle_uv_matches_scalar_reference():
    coords = _node_coords(
        np.array([133.1, 133.2, 133.3, 133.4]),
        np.array([34.1, 34.2, 34.3, 34.4]),
    )
    oriented_nodes = np.array([[0, 2, 1], [1, 3, 0]], dtype=np.uint32)
    attributes = [(0, 0, 16, "BI"), (16, 32, 17, "BI")]

    actual = DSF._precompute_triangle_uvs(coords, oriented_nodes, attributes)
    for triangle_index, nodes in enumerate(oriented_nodes):
        for vertex_index, node in enumerate(nodes):
            s, t = DSF.numpy_st_coord(
                coords[5 * node + 1],
                coords[5 * node],
                *attributes[triangle_index][:3],
            )
            assert actual[triangle_index, vertex_index, 0] == round(s * 65535)
            assert actual[triangle_index, vertex_index, 1] == round(t * 65535)


def test_texture_requirements_are_planned_once_per_attribute(tmp_path, monkeypatch):
    tile = SimpleNamespace(
        build_dir=str(tmp_path),
        imprint_masks_to_dds=True,
        dds_format="AUTO",
        lat=34,
        lon=133,
        mesh_zl=16,
        mask_zl=15,
    )
    (tmp_path / "textures").mkdir()
    attributes = [(0, 0, 16, "BI"), (16, 0, 16, "BI"), (0, 0, 16, "BI")]
    tri_types = np.array([2, 0, 2], dtype=np.uint32)
    mask_calls = []
    contract_calls = []
    mask = Image.new("L", (4, 4), 255)
    for values in set(attributes):
        (tmp_path / "textures" / DSF.FNAMES.dds_file_name_from_attributes(*values)).touch()

    def fake_needs_mask(_tile, *values):
        mask_calls.append(values)
        return mask

    def fake_contract(_tile, values, has_alpha):
        contract_calls.append((values, has_alpha))
        return True

    monkeypatch.setattr(DSF.MASK, "needs_mask", fake_needs_mask)
    monkeypatch.setattr(DSF, "_texture_contract_matches", fake_contract)

    plans = DSF._plan_texture_requirements(tile, attributes, tri_types)

    assert set(plans) == set(attributes)
    assert len(mask_calls) == 2
    assert len(contract_calls) == 2
    assert all(not plan["rebuild"] for plan in plans.values())
