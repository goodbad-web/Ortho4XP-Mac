import sys
import math
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from O4_DSF_Budget import (  # noqa: E402
    DEFAULT_DSF_NODE_BUDGET,
    MAX_AUTO_REDUCE_ATTEMPTS,
    normalize_budget,
    retry_settings,
    retry_stage_for_settings,
    stable_id_key,
    summarize_dsf_pools,
    validate_dsf_commands,
)


def test_retry_settings_apply_all_reductions_from_baseline_in_one_attempt():
    base = {
        "max_levelled_segs": 100000,
        "water_simplification": 1.0,
        "cover_zl": 18,
        "curvature_tol": 3.0,
        "limit_tris": 0.8,
    }

    assert retry_settings(base, 0, 16) == {}
    actual = retry_settings(base, 1, 16)
    assert actual["max_levelled_segs"] == 50000
    assert actual["water_simplification"] == 2.0
    assert actual["cover_zl"] == 17
    assert math.isclose(actual["curvature_tol"], 3.75)
    assert math.isclose(actual["limit_tris"], 0.64)

    try:
        retry_settings(base, 2, 16)
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("a second automatic reduction must be rejected")


def test_retry_settings_keep_mesh_zoom_as_lower_bound():
    base = {
        "max_levelled_segs": 40000,
        "water_simplification": 0.0,
        "cover_zl": 17,
        "curvature_tol": 1.0,
        "limit_tris": 1.0,
    }
    assert retry_settings(base, MAX_AUTO_REDUCE_ATTEMPTS, 18)["cover_zl"] == 18


def test_retry_stage_uses_conservative_dependency_table():
    assert retry_stage_for_settings({"curvature_tol", "limit_tris"}) == "mesh"
    assert retry_stage_for_settings({"cover_zl"}) == "vector data"
    assert retry_stage_for_settings({"unknown_setting"}) == "vector data"


def test_summarize_dsf_pools_separates_budget_from_structural_validity():
    metrics = summarize_dsf_pools(
        textured_node_count=5,
        pool_lengths=[2, 3],
        pool_planes=[7, 7],
        pool_data=[list(range(14)), list(range(21))],
        pool_count=2,
    )

    assert metrics["structurally_valid"] is True
    assert metrics["point_count"] == 5
    assert metrics["pool_point_count"] == 5
    assert metrics["max_pool_points"] == 3


class _SizedData:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_summarize_dsf_pools_rejects_invalid_16_bit_pool():
    metrics = summarize_dsf_pools(
        textured_node_count=65_536,
        pool_lengths=[65_536],
        pool_planes=[7],
        pool_data=[_SizedData(7 * 65_536)],
        pool_count=1,
    )

    assert metrics["structurally_valid"] is False
    assert "more than 65535" in " ".join(metrics["structural_errors"])


def test_validate_dsf_commands_rejects_out_of_range_point_references():
    assert validate_dsf_commands({0: {0: [0, 1]}}, [2]) == ()
    errors = validate_dsf_commands({0: {0: [0, 2]}}, [2])
    assert "outside pool 0" in " ".join(errors)


def test_budget_and_identifier_helpers_are_defensive_and_stable():
    assert normalize_budget("invalid") == DEFAULT_DSF_NODE_BUDGET
    assert normalize_budget(0) == 1
    assert sorted(["10", "2", "-1"], key=stable_id_key) == ["-1", "2", "10"]
