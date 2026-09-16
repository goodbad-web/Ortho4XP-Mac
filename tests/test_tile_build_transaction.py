import sys
from pathlib import Path
from types import SimpleNamespace

SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Tile_Utils as TILE  # noqa: E402


def _tile(tmp_path, monkeypatch, lat=1, lon=2):
    mask_root = tmp_path / "Masks"
    monkeypatch.setattr(TILE.FNAMES, "Mask_dir", str(mask_root))
    build_dir = tmp_path / "Tiles" / "zOrtho4XP_test"
    return SimpleNamespace(
        lat=lat,
        lon=lon,
        custom_build_dir=str(tmp_path / "Tiles") + "/",
        grouped=False,
        build_dir=str(build_dir),
        mesh_zl=16,
        max_levelled_segs=100000,
        water_simplification=1.0,
        cover_zl=18,
        curvature_tol=3.0,
        limit_tris=0.8,
        dsf_node_budget=100,
    )


def _write_outputs(tile, label):
    build_dir = Path(tile.build_dir)
    (build_dir / "Earth nav data" / TILE.FNAMES.round_latlon(tile.lat, tile.lon)).mkdir(
        parents=True, exist_ok=True
    )
    (build_dir / "terrain").mkdir(exist_ok=True)
    (build_dir / "textures").mkdir(exist_ok=True)
    (Path(TILE.FNAMES.mask_dir(tile.lat, tile.lon))).mkdir(
        parents=True, exist_ok=True
    )
    data_prefix = "Data" + TILE.FNAMES.short_latlon(tile.lat, tile.lon)
    (build_dir / (data_prefix + ".mesh")).write_text(label, encoding="utf-8")
    dsf_path = build_dir / "Earth nav data" / TILE.FNAMES.long_latlon(
        tile.lat, tile.lon
    )
    (dsf_path.with_suffix(".dsf")).write_text(label, encoding="utf-8")
    (build_dir / "terrain" / ("100_200_BI16_" + label + ".ter")).write_text(
        label, encoding="utf-8"
    )
    (build_dir / "textures" / "100_200_BI16.dds").write_text(
        label, encoding="utf-8"
    )
    (build_dir / "textures" / "100_200_BI16_ZL16.png").write_text(
        label, encoding="utf-8"
    )
    Path(TILE.FNAMES.mask_dir(tile.lat, tile.lon), "100_200.png").write_text(
        label, encoding="utf-8"
    )


def _write_level_output(tile, level, include_terrain=True, label="candidate"):
    build_dir = Path(tile.build_dir)
    (build_dir / "terrain").mkdir(parents=True, exist_ok=True)
    (build_dir / "textures").mkdir(parents=True, exist_ok=True)
    texture_name = "100_200_BI{}.dds".format(level)
    (build_dir / "textures" / texture_name).write_text(label, encoding="utf-8")
    if include_terrain:
        (build_dir / "terrain" / texture_name.replace(".dds", ".ter")).write_text(
            label, encoding="utf-8"
        )


def test_transaction_restores_outputs_and_preserves_unmanaged_files(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    build_dir = Path(tile.build_dir)
    (build_dir / "terrain").mkdir(parents=True)
    (build_dir / "textures").mkdir()
    (build_dir / "terrain" / "sentinel.txt").write_text("keep", encoding="utf-8")
    (build_dir / "terrain" / "sentinel.ter").write_text("keep", encoding="utf-8")
    (build_dir / "textures" / "sentinel.png").write_text("keep", encoding="utf-8")
    (build_dir / "textures" / "sentinel.dds").write_text("keep", encoding="utf-8")
    mask_dir = Path(TILE.FNAMES.mask_dir(tile.lat, tile.lon))
    mask_dir.mkdir(parents=True)
    (mask_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
    (mask_dir / "sentinel.png").write_text("keep", encoding="utf-8")
    _write_outputs(tile, "initial")

    transaction = TILE._BuildTransaction(tile)
    assert not (build_dir / "textures" / "100_200_BI16.dds").exists()
    assert (build_dir / "textures" / "sentinel.dds").exists()
    _write_outputs(tile, "candidate")
    candidate = transaction.capture_candidate(0)
    transaction.restore_snapshot(candidate)

    assert (build_dir / "Data+01+002.mesh").read_text(encoding="utf-8") == "candidate"
    assert (build_dir / "textures" / "sentinel.png").read_text(encoding="utf-8") == "keep"
    assert (build_dir / "textures" / "sentinel.dds").read_text(encoding="utf-8") == "keep"
    assert (build_dir / "terrain" / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert (build_dir / "terrain" / "sentinel.ter").read_text(encoding="utf-8") == "keep"
    assert (mask_dir / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert (mask_dir / "sentinel.png").read_text(encoding="utf-8") == "keep"
    transaction.cleanup()
    assert not Path(transaction.root).exists()
    assert not (build_dir / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_standalone_transaction_keeps_inputs_visible_and_restores_in_place_writes(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_outputs(tile, "initial")
    build_dir = Path(tile.build_dir)
    transaction = TILE._BuildTransaction(tile, preserve_inputs=True)

    assert (build_dir / "Data+01+002.mesh").read_text(encoding="utf-8") == "initial"
    (build_dir / "terrain" / "100_200_BI16_initial.ter").write_text(
        "candidate", encoding="utf-8"
    )
    dds_tmp = build_dir / "textures" / "100_200_BI16.dds.tmp"
    dds_tmp.write_text(
        "candidate", encoding="utf-8"
    )
    dds_tmp.replace(build_dir / "textures" / "100_200_BI16.dds")
    (build_dir / "textures" / "300_400_BI16.dds").write_text(
        "new", encoding="utf-8"
    )

    transaction.restore_snapshot("initial")
    transaction._write_marker("restored", None, None)
    transaction.cleanup()

    assert (
        build_dir / "terrain" / "100_200_BI16_initial.ter"
    ).read_text(encoding="utf-8") == "initial"
    assert (build_dir / "textures" / "100_200_BI16.dds").read_text(
        encoding="utf-8"
    ) == "initial"
    assert not (build_dir / "textures" / "300_400_BI16.dds").exists()
    assert (build_dir / "Data+01+002.mesh").read_text(encoding="utf-8") == "initial"
    assert not (build_dir / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_recovery_discards_an_interrupted_standalone_snapshot(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_outputs(tile, "initial")
    build_dir = Path(tile.build_dir)
    transaction = TILE._BuildTransaction(tile, preserve_inputs=True)
    transaction._write_marker("snapshotting", None, None)

    assert TILE._recover_build_transaction(tile) is True
    assert (
        build_dir / "terrain" / "100_200_BI16_initial.ter"
    ).read_text(encoding="utf-8") == "initial"
    assert (build_dir / "textures" / "100_200_BI16.dds").read_text(
        encoding="utf-8"
    ) == "initial"
    assert not (build_dir / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_standalone_build_restores_outputs_when_step_three_fails(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_outputs(tile, "initial")
    build_dir = Path(tile.build_dir)

    monkeypatch.setattr(TILE.UI, "is_building_all", False)
    monkeypatch.setattr(TILE.UI, "is_working", 0)
    monkeypatch.setattr(TILE.UI, "initialize_build_log", lambda *args: None)
    monkeypatch.setattr(TILE.UI, "flush_build_log", lambda *args: None)
    monkeypatch.setattr(TILE.UI, "exit_message_and_bottom_line", lambda *args: None)

    def failed_build(current_tile, persist_config=True):
        current_dir = Path(current_tile.build_dir)
        (current_dir / "terrain" / "100_200_BI16_initial.ter").write_text(
            "candidate", encoding="utf-8"
        )
        dds_tmp = current_dir / "textures" / "100_200_BI16.dds.tmp"
        dds_tmp.write_text(
            "candidate", encoding="utf-8"
        )
        dds_tmp.replace(current_dir / "textures" / "100_200_BI16.dds")
        (current_dir / "textures" / "300_400_BI16.dds").write_text(
            "new", encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(TILE, "_build_tile", failed_build)

    assert TILE.build_tile(tile) == 0
    assert (
        build_dir / "terrain" / "100_200_BI16_initial.ter"
    ).read_text(encoding="utf-8") == "initial"
    assert (build_dir / "textures" / "100_200_BI16.dds").read_text(
        encoding="utf-8"
    ) == "initial"
    assert not (build_dir / "textures" / "300_400_BI16.dds").exists()
    assert not (build_dir / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_standalone_build_commits_outputs_after_step_three_succeeds(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_outputs(tile, "initial")
    build_dir = Path(tile.build_dir)

    monkeypatch.setattr(TILE.UI, "is_building_all", False)
    monkeypatch.setattr(TILE.UI, "is_working", 0)
    monkeypatch.setattr(TILE.UI, "initialize_build_log", lambda *args: None)
    monkeypatch.setattr(TILE.UI, "flush_build_log", lambda *args: None)

    def successful_build(current_tile, persist_config=True):
        current_dir = Path(current_tile.build_dir)
        (current_dir / "terrain" / "100_200_BI16_initial.ter").write_text(
            "candidate", encoding="utf-8"
        )
        return 1

    monkeypatch.setattr(TILE, "_build_tile", successful_build)

    assert TILE.build_tile(tile) == 1
    assert (
        build_dir / "terrain" / "100_200_BI16_initial.ter"
    ).read_text(encoding="utf-8") == "candidate"
    assert not (build_dir / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_recovery_marker_restores_best_snapshot_after_interruption(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_outputs(tile, "initial")
    transaction = TILE._BuildTransaction(tile)
    _write_outputs(tile, "best")
    candidate = transaction.capture_candidate(0)
    transaction.set_best_snapshot(candidate)
    (Path(tile.build_dir) / "Data+01+002.mesh").write_text(
        "partial", encoding="utf-8"
    )

    assert TILE._recover_build_transaction(tile) is True
    assert (Path(tile.build_dir) / "Data+01+002.mesh").read_text(encoding="utf-8") == "best"
    assert not (Path(tile.build_dir) / TILE._BUILD_TRANSACTION_MARKER).exists()
    assert not Path(transaction.root).exists()


def test_recovery_finishes_output_restore_and_persists_matching_config(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_outputs(tile, "initial")
    transaction = TILE._BuildTransaction(tile)
    _write_outputs(tile, "best")
    candidate = transaction.capture_candidate(0)
    settings = {
        "max_levelled_segs": 50000,
        "water_simplification": 2.0,
        "cover_zl": 17,
        "curvature_tol": 3.75,
        "limit_tris": 0.64,
    }
    transaction._write_marker("restoring", candidate, settings, candidate)
    transaction.discard_current("before-restore")
    transaction.restore_snapshot_files(candidate)

    def write_config():
        Path(tile.build_dir, "selected.cfg").write_text(
            str(tile.max_levelled_segs), encoding="utf-8"
        )
        return 1

    tile.write_to_config = write_config
    assert TILE._recover_build_transaction(tile) is True
    assert (Path(tile.build_dir) / "Data+01+002.mesh").read_text(encoding="utf-8") == "best"
    assert tile.max_levelled_segs == 50000
    assert (Path(tile.build_dir) / "selected.cfg").read_text(encoding="utf-8") == "50000"
    assert not Path(tile.build_dir, TILE._BUILD_TRANSACTION_MARKER).exists()


def test_grouped_transaction_carries_shared_assets_into_candidate(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    tile.grouped = True
    build_dir = Path(tile.build_dir)
    (build_dir / "terrain").mkdir(parents=True)
    (build_dir / "textures").mkdir()
    (build_dir / "terrain" / "200_300_OTHER.ter").write_text(
        "other tile", encoding="utf-8"
    )
    (build_dir / "textures" / "200_300_OTHER.dds").write_text(
        "other tile", encoding="utf-8"
    )

    transaction = TILE._BuildTransaction(tile)
    transaction.prepare_attempt()
    assert (build_dir / "textures" / "200_300_OTHER.dds").exists()
    _write_outputs(tile, "grouped")
    candidate = transaction.capture_candidate(0)
    transaction.restore_snapshot(candidate)

    assert (build_dir / "terrain" / "200_300_OTHER.ter").read_text(encoding="utf-8") == "other tile"
    assert (build_dir / "textures" / "200_300_OTHER.dds").read_text(encoding="utf-8") == "other tile"
    transaction.cleanup()


def test_remove_unwanted_textures_walks_terrain_and_preserves_unmanaged_files(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    _write_level_output(tile, 18)
    _write_level_output(tile, 19, include_terrain=False)
    nested_terrain_dir = Path(tile.build_dir) / "terrain" / "nested"
    nested_terrain_dir.mkdir()
    (Path(tile.build_dir) / "terrain" / "100_200_BI18.ter").rename(
        nested_terrain_dir / "100_200_BI18.ter"
    )
    (Path(tile.build_dir) / "textures" / "sentinel.dds").write_text(
        "keep", encoding="utf-8"
    )
    cache_file = tmp_path / "Orthophotos" / "100_200_BI19.jpg"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("keep", encoding="utf-8")

    removed = TILE.remove_unwanted_textures(tile)

    assert removed == [str(Path(tile.build_dir) / "textures" / "100_200_BI19.dds")]
    assert (Path(tile.build_dir) / "textures" / "100_200_BI18.dds").exists()
    assert not (Path(tile.build_dir) / "textures" / "100_200_BI19.dds").exists()
    assert (Path(tile.build_dir) / "textures" / "sentinel.dds").exists()
    assert cache_file.exists()


def test_grouped_remove_unwanted_textures_preserves_shared_outputs(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    tile.grouped = True
    _write_level_output(tile, 18)
    _write_level_output(tile, 19, include_terrain=False)

    assert TILE.remove_unwanted_textures(tile) == []
    assert (Path(tile.build_dir) / "textures" / "100_200_BI19.dds").exists()


def test_build_all_runs_at_most_two_attempts_and_saves_selected_settings(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    attempts = []

    def fake_pipeline(current_tile):
        attempt = len(attempts)
        attempts.append(attempt)
        _write_outputs(current_tile, "reduced" if attempt else "baseline")
        current_tile.last_dsf_metrics = {
            "point_count": 80 if attempt else 120,
            "budget_exceeded": True,
            "structurally_valid": True,
        }
        return 1

    def write_config():
        Path(tile.build_dir).mkdir(parents=True, exist_ok=True)
        Path(tile.build_dir, "selected.cfg").write_text(
            "max_levelled_segs=" + str(tile.max_levelled_segs), encoding="utf-8"
        )
        return 1

    tile.write_to_config = write_config
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 1
    assert attempts == [0, 1]
    assert tile.max_levelled_segs == 50000
    assert tile.water_simplification == 2.0
    assert tile.cover_zl == 17
    assert (Path(tile.build_dir) / "Data+01+002.mesh").read_text(encoding="utf-8") == "reduced"
    assert "50000" in (Path(tile.build_dir) / "selected.cfg").read_text(encoding="utf-8")
    assert not (Path(tile.build_dir) / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_selected_reduced_attempt_removes_old_unreferenced_textures(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    attempts = []

    def fake_pipeline(current_tile):
        attempt = len(attempts)
        attempts.append(attempt)
        if attempt == 0:
            _write_level_output(current_tile, 19, label="baseline")
        else:
            old_terrain = Path(current_tile.build_dir) / "terrain" / "100_200_BI19.ter"
            old_terrain.unlink()
            _write_level_output(current_tile, 18, label="reduced")
        current_tile.last_dsf_metrics = {
            "point_count": 80 if attempt else 120,
            "budget_exceeded": attempt == 0,
            "structurally_valid": True,
        }
        return 1

    tile.write_to_config = lambda: 1
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 1
    assert attempts == [0, 1]
    assert (Path(tile.build_dir) / "textures" / "100_200_BI18.dds").exists()
    assert not (Path(tile.build_dir) / "textures" / "100_200_BI19.dds").exists()


def test_failed_reduced_attempt_does_not_prune_baseline_textures(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    attempts = []

    def fake_pipeline(current_tile):
        attempt = len(attempts)
        attempts.append(attempt)
        if attempt == 0:
            _write_level_output(current_tile, 19, label="baseline")
            current_tile.last_dsf_metrics = {
                "point_count": 120,
                "budget_exceeded": True,
                "structurally_valid": True,
            }
            return 1
        _write_level_output(current_tile, 18, label="failed")
        return 0

    tile.write_to_config = lambda: 1
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 1
    assert attempts == [0, 1]
    assert (Path(tile.build_dir) / "textures" / "100_200_BI19.dds").exists()
    assert not (Path(tile.build_dir) / "textures" / "100_200_BI18.dds").exists()


def test_failed_reduced_attempt_keeps_baseline_as_one_consistent_state(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    attempts = []

    def fake_pipeline(current_tile):
        attempt = len(attempts)
        attempts.append(attempt)
        _write_outputs(current_tile, "baseline" if attempt == 0 else "failed")
        if attempt == 0:
            current_tile.last_dsf_metrics = {
                "point_count": 120,
                "budget_exceeded": True,
                "structurally_valid": True,
            }
            return 1
        return 0

    def write_config():
        Path(tile.build_dir).mkdir(parents=True, exist_ok=True)
        Path(tile.build_dir, "selected.cfg").write_text(
            "max_levelled_segs=" + str(tile.max_levelled_segs), encoding="utf-8"
        )
        return 1

    tile.write_to_config = write_config
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 1
    assert attempts == [0, 1]
    assert tile.max_levelled_segs == 100000
    assert (Path(tile.build_dir) / "Data+01+002.mesh").read_text(encoding="utf-8") == "baseline"
    assert "100000" in (Path(tile.build_dir) / "selected.cfg").read_text(encoding="utf-8")


def test_baseline_within_budget_does_not_run_reduced_attempt(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    attempts = []

    def fake_pipeline(current_tile):
        attempts.append(len(attempts))
        _write_outputs(current_tile, "baseline")
        current_tile.last_dsf_metrics = {
            "point_count": 80,
            "budget_exceeded": False,
            "structurally_valid": True,
        }
        return 1

    tile.write_to_config = lambda: 1
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 1
    assert attempts == [0]
    assert tile.max_levelled_segs == 100000


def test_equal_point_count_keeps_baseline_for_quality(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    attempts = []

    def fake_pipeline(current_tile):
        attempt = len(attempts)
        attempts.append(attempt)
        _write_outputs(current_tile, "reduced" if attempt else "baseline")
        current_tile.last_dsf_metrics = {
            "point_count": 120,
            "budget_exceeded": True,
            "structurally_valid": True,
        }
        return 1

    def write_config():
        Path(tile.build_dir, "selected.cfg").write_text(
            str(tile.max_levelled_segs), encoding="utf-8"
        )
        return 1

    tile.write_to_config = write_config
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 1
    assert attempts == [0, 1]
    assert tile.max_levelled_segs == 100000
    assert (Path(tile.build_dir) / "Data+01+002.mesh").read_text(encoding="utf-8") == "baseline"


def test_config_failure_rolls_back_selected_outputs_to_initial_snapshot(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    build_dir = Path(tile.build_dir)
    _write_outputs(tile, "initial")
    (build_dir / "selected.cfg").write_text("old", encoding="utf-8")

    def fake_pipeline(current_tile):
        _write_outputs(current_tile, "candidate")
        current_tile.last_dsf_metrics = {
            "point_count": 80,
            "budget_exceeded": False,
            "structurally_valid": True,
        }
        return 1

    tile.write_to_config = lambda: 0
    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 0
    assert (build_dir / "Data+01+002.mesh").read_text(encoding="utf-8") == "initial"
    assert (build_dir / "selected.cfg").read_text(encoding="utf-8") == "old"
    assert not (build_dir / TILE._BUILD_TRANSACTION_MARKER).exists()


def test_baseline_failure_restores_prebuild_outputs(
    tmp_path, monkeypatch
):
    tile = _tile(tmp_path, monkeypatch)
    build_dir = Path(tile.build_dir)
    _write_outputs(tile, "initial")

    def fake_pipeline(current_tile):
        _write_outputs(current_tile, "failed")
        return 0

    monkeypatch.setattr(TILE, "_run_pipeline_once", fake_pipeline)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    assert TILE._build_all(tile, include_overlays=False) == 0
    assert (build_dir / "Data+01+002.mesh").read_text(encoding="utf-8") == "initial"


def test_batch_skips_missing_cfg_and_continues(tmp_path, monkeypatch):
    tile = _tile(tmp_path, monkeypatch)
    seen = []
    config_results = iter((False, True))

    def read_config():
        return next(config_results)

    tile.read_from_config = read_config
    tile.make_dirs = lambda: None
    monkeypatch.setattr(
        TILE.VMAP,
        "build_poly_file",
        lambda current_tile: seen.append((current_tile.lat, current_tile.lon)) or 1,
    )
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    result = TILE.build_tile_list(
        tile,
        [(1, 2), (3, 4)],
        do_osm=True,
        do_mesh=False,
        do_mask=False,
        do_dsf=False,
        do_ovl=False,
        do_ptc=True,
    )
    assert result == 0
    assert seen == [(3, 4)]


def test_batch_continues_after_non_cancelled_stage_failure(tmp_path, monkeypatch):
    tile = _tile(tmp_path, monkeypatch)
    tile.read_from_config = lambda: True
    tile.make_dirs = lambda: None
    seen = []
    outcomes = iter((0, 1))

    def build_vector(current_tile):
        seen.append((current_tile.lat, current_tile.lon))
        return next(outcomes)

    monkeypatch.setattr(TILE.VMAP, "build_poly_file", build_vector)
    monkeypatch.setattr(TILE.UI, "red_flag", False)

    result = TILE.build_tile_list(
        tile,
        [(1, 2), (3, 4)],
        do_osm=True,
        do_mesh=False,
        do_mask=False,
        do_dsf=False,
        do_ovl=False,
        do_ptc=True,
    )
    assert result == 0
    assert seen == [(1, 2), (3, 4)]


def test_pipeline_reports_exception_and_cancellation_separately(monkeypatch):
    tile = SimpleNamespace()
    messages = []
    monkeypatch.setattr(TILE.UI, "logprint", lambda *args: None)
    monkeypatch.setattr(
        TILE.UI,
        "exit_message_and_bottom_line",
        lambda *args: messages.append(" ".join(map(str, args))),
    )

    def raise_vector(_):
        raise RuntimeError("vector exploded")

    monkeypatch.setattr(TILE.VMAP, "build_poly_file", raise_vector)
    monkeypatch.setattr(TILE.UI, "red_flag", False)
    assert TILE._run_pipeline_once(tile) == 0
    assert tile.last_pipeline_failure["cancelled"] is False
    assert "vector data stage failed: vector exploded" in messages[-1]

    def cancel_vector(_):
        TILE.UI.red_flag = True
        return 0

    monkeypatch.setattr(TILE.VMAP, "build_poly_file", cancel_vector)
    assert TILE._run_pipeline_once(tile) == 0
    assert tile.last_pipeline_failure["cancelled"] is True
    assert "cancelled" in messages[-1]
