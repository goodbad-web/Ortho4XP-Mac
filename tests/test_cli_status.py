import Ortho4XP


def test_continuous_build_cli_statuses_are_distinct():
    assert Ortho4XP._continuous_build_exit_code(Ortho4XP.OSM.OSM_COMPLETE) == 0
    assert (
        Ortho4XP._continuous_build_exit_code(Ortho4XP.OSM.OSM_DEGRADED)
        == Ortho4XP.CLI_DEGRADED_EXIT_CODE
    )
    assert Ortho4XP._continuous_build_exit_code(Ortho4XP.OSM.OSM_FAILED) == 1


def test_cancelled_continuous_build_is_failed_even_if_stage_returns_complete():
    assert (
        Ortho4XP._continuous_build_exit_code(
            Ortho4XP.OSM.OSM_COMPLETE, cancelled=True
        )
        == 1
    )
