"""Pure helpers for DSF point-budget diagnostics and auto-reduction."""

from math import floor


DEFAULT_DSF_NODE_BUDGET = 1_000_000
MAX_AUTO_REDUCE_ATTEMPTS = 3
MIN_LEVELLED_SEGMENTS = 25_000


def stable_id_key(value):
    """Return a deterministic ordering key for OSM and R-tree identifiers."""
    text = str(value)
    try:
        return (0, int(text), text)
    except (TypeError, ValueError):
        return (1, text)


def normalize_budget(value, default=DEFAULT_DSF_NODE_BUDGET):
    """Return a positive integer budget while tolerating old/invalid configs."""
    try:
        budget = int(value)
    except (TypeError, ValueError):
        budget = default
    return max(1, budget)


def retry_settings(base_settings, attempt, mesh_zl):
    """Return cumulative settings for one automatic reduction attempt.

    ``attempt`` is zero for the baseline and one through three for the
    configured reduction attempts. Values are always derived from the original
    baseline so repeated retries do not compound floating point rounding.
    """
    if attempt < 0 or attempt > MAX_AUTO_REDUCE_ATTEMPTS:
        raise ValueError("attempt must be between 0 and 3")

    settings = {}
    if attempt >= 1:
        settings["max_levelled_segs"] = max(
            MIN_LEVELLED_SEGMENTS,
            floor(float(base_settings["max_levelled_segs"]) * 0.5),
        )
    if attempt >= 2:
        settings["water_simplification"] = max(
            2.0,
            float(base_settings["water_simplification"]) * 2.0,
        )
    if attempt >= 3:
        settings["cover_zl"] = max(
            int(mesh_zl),
            int(base_settings["cover_zl"]) - 1,
        )
        settings["curvature_tol"] = float(base_settings["curvature_tol"]) * 1.25
        settings["limit_tris"] = float(base_settings["limit_tris"]) * 0.8
    return settings


def summarize_dsf_pools(
    textured_node_count,
    pool_lengths,
    pool_planes,
    pool_data,
    pool_count=None,
):
    """Return DSF pool metrics and structural validation results.

    Point references in the generated command streams are 16-bit values, so a
    pool with more than 65535 points is structurally invalid for this writer.
    The function deliberately separates structural validity from the advisory
    point budget.
    """
    lengths = [int(value) for value in pool_lengths]
    planes = [int(value) for value in pool_planes]
    if pool_count is None:
        pool_count = len(lengths)

    reasons = []
    if len(lengths) != len(planes):
        reasons.append("pool length/plane arrays have different sizes")
    if len(lengths) != pool_count:
        reasons.append("pool count does not match pool arrays")
    if int(pool_count) > 65535:
        reasons.append("DSF pool count exceeds the 16-bit pool index range")

    active_lengths = []
    for index, length in enumerate(lengths):
        plane = planes[index] if index < len(planes) else 0
        if length < 0:
            reasons.append(f"pool {index} has a negative point count")
        if length > 65535:
            reasons.append(f"pool {index} has more than 65535 point references")
        if plane <= 0:
            reasons.append(f"pool {index} has an invalid plane count")
        try:
            data_length = len(pool_data[index])
        except (IndexError, KeyError, TypeError):
            reasons.append(f"pool {index} has no data")
            continue
        if plane > 0 and data_length != plane * length:
            reasons.append(
                f"pool {index} data length {data_length} does not match "
                f"plane {plane} * points {length}"
            )
        if length:
            active_lengths.append(length)

    total_pool_points = sum(lengths)
    if int(textured_node_count) != total_pool_points:
        reasons.append(
            "textured node count does not match the sum of DSF pool points"
        )

    return {
        "point_count": int(textured_node_count),
        "pool_point_count": total_pool_points,
        "pool_count": int(pool_count),
        "active_pool_count": len(active_lengths),
        "max_pool_points": max(active_lengths, default=0),
        "structurally_valid": not reasons,
        "structural_errors": tuple(reasons),
    }


def validate_dsf_commands(textured_tris, pool_lengths):
    """Validate point references used by DSF command streams."""
    lengths = [int(value) for value in pool_lengths]
    active_pools = {index for index, length in enumerate(lengths) if length}
    reasons = []

    for terrain_index, terrain_commands in textured_tris.items():
        for pool_index, data in terrain_commands.items():
            if pool_index == "cross-pool":
                if len(data) % 2:
                    reasons.append(
                        f"terrain {terrain_index} has an odd cross-pool command length"
                    )
                    continue
                references = zip(data[0::2], data[1::2])
            else:
                references = ((pool_index, point_index) for point_index in data)

            for referenced_pool, point_index in references:
                try:
                    referenced_pool = int(referenced_pool)
                    point_index = int(point_index)
                except (TypeError, ValueError):
                    reasons.append(
                        f"terrain {terrain_index} contains a non-integer "
                        "point reference"
                    )
                    continue
                if referenced_pool not in active_pools:
                    reasons.append(
                        f"terrain {terrain_index} references inactive pool "
                        f"{referenced_pool}"
                    )
                    continue
                if point_index < 0 or point_index >= lengths[referenced_pool]:
                    reasons.append(
                        f"terrain {terrain_index} references point {point_index} "
                        f"outside pool {referenced_pool}"
                    )
    return tuple(reasons)
