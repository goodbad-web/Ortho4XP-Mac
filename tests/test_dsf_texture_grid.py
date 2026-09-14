import sys
from types import SimpleNamespace

import numpy as np
import pytest


SRC_ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import O4_DSF_Utils as DSF  # noqa: E402


def _tile_with_full_high_resolution_zone():
    return SimpleNamespace(
        lat=34,
        lon=133,
        mesh_zl=16,
        default_zl=16,
        default_website="BI",
        cover_airports_with_highres="False",
        cover_zl=18,
        cover_extent=0,
        zone_list=[
            (
                [34, 133, 34, 134, 35, 134, 35, 133, 34, 133],
                18,
                "BI",
            )
        ],
    )


def test_target_coordinate_uses_unaligned_mesh_bounds():
    tile = _tile_with_full_high_resolution_zone()
    bounds = DSF._mesh_orthogrid_bounds(tile)
    til_x, til_y = DSF.numpy_wgs84_to_orthogrid(
        34.59791740831474,
        133.98265657435277,
        tile.mesh_zl,
    )

    assert bounds == (56979, 25958, 57161, 26179)
    assert (int(til_x), int(til_y)) == (57158, 26047)
    DSF._validate_mesh_orthogrid_indices(til_x, til_y, bounds)


def test_target_coordinate_does_not_fall_back_to_previous_last_texture_column():
    dico = DSF.zone_list_to_ortho_dico(
        _tile_with_full_high_resolution_zone()
    )

    assert dico[(57158, 26047)] == (228624, 104176, 18, "BI")
    assert dico[(57158, 26047)] != (228608, 104176, 18, "BI")


def test_out_of_bounds_mesh_indices_are_rejected_instead_of_clipped():
    bounds = (56979, 25958, 57161, 26179)

    with pytest.raises(ValueError, match="outside tile bounds"):
        DSF._validate_mesh_orthogrid_indices(
            np.array([57162, 57158]),
            np.array([26047, 26180]),
            bounds,
        )
