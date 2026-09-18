import json
import sys

import pytest

sys.path.insert(0, "src")

from muxp_engine.engine import MuxpApplyError, discover_muxp_files, write_manifest


def _muxp(*, update_id="airport_test", version="1.0", tile="+34+132", source=None, command="cut_polygon"):
    source_line = "" if source is None else "source_dsf: {}\n".format(source)
    if command == "cut_polygon":
        body = """cut_polygon:
  coordinates:
  - 34.0100 132.0100
  - 34.0200 132.0100
  - 34.0200 132.0200
  - 34.0100 132.0100
  terrain: terrain_Water
"""
    elif command == "unflatten_default_apt":
        body = """unflatten_default_apt:
  name: TEST
"""
    else:
        body = "{}:\n".format(command)
    return (
        "muxp_version: 0.3\n"
        "id: {}\n"
        "version: {}\n"
        "{}"
        "tile: {}\n"
        "area: 34.0 34.1 132.0 132.1\n"
        "{}"
    ).format(update_id, version, source_line, tile, body)


def test_discovery_is_recursive_and_deterministic(tmp_path):
    root = tmp_path / "MUXP"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "b" / "z.muxp").write_text(_muxp(update_id="z"), encoding="utf-8")
    (root / "a" / "a.muxp").write_text(_muxp(update_id="a"), encoding="utf-8")

    files = discover_muxp_files(str(tmp_path), "MUXP", "+34+132")

    assert [item.relative_path for item in files] == ["a/a.muxp", "b/z.muxp"]


def test_same_id_uses_latest_numeric_version(tmp_path):
    root = tmp_path / "MUXP"
    root.mkdir()
    (root / "old.muxp").write_text(_muxp(version="1.0"), encoding="utf-8")
    (root / "new.muxp").write_text(_muxp(version="2.0"), encoding="utf-8")

    files = discover_muxp_files(str(tmp_path), "MUXP", "+34+132")

    assert len(files) == 1
    assert files[0].relative_path == "new.muxp"
    assert files[0].update["version"] == "2.0"


def test_same_id_and_version_with_different_content_fails(tmp_path):
    root = tmp_path / "MUXP"
    root.mkdir()
    (root / "a.muxp").write_text(_muxp(version="1.0"), encoding="utf-8")
    (root / "b.muxp").write_text(
        _muxp(version="1.0").replace("terrain_Water", "terrain_Concrete"),
        encoding="utf-8",
    )

    with pytest.raises(MuxpApplyError, match="different contents"):
        discover_muxp_files(str(tmp_path), "MUXP", "+34+132")


def test_default_only_source_is_rejected(tmp_path):
    root = tmp_path / "MUXP"
    root.mkdir()
    (root / "default.muxp").write_text(
        _muxp(source="DEFAULT"), encoding="utf-8"
    )

    with pytest.raises(MuxpApplyError, match="source_dsf"):
        discover_muxp_files(str(tmp_path), "MUXP", "+34+132")


def test_ortho4xp_source_with_default_fallback_is_allowed(tmp_path):
    root = tmp_path / "MUXP"
    root.mkdir()
    (root / "ortho.muxp").write_text(
        _muxp(source="pack=*Ortho4XP DEFAULT"), encoding="utf-8"
    )

    files = discover_muxp_files(str(tmp_path), "MUXP", "+34+132")

    assert len(files) == 1


@pytest.mark.parametrize("source", ["*Ortho4XP", "*Ortho4xp", "*Ortho4XP DEFAULT"])
def test_legacy_ortho4xp_source_forms_are_allowed(tmp_path, source):
    root = tmp_path / "MUXP"
    root.mkdir()
    (root / "ortho.muxp").write_text(_muxp(source=source), encoding="utf-8")

    files = discover_muxp_files(str(tmp_path), "MUXP", "+34+132")

    assert len(files) == 1


def test_side_effect_command_is_valid_but_gated_by_apply(tmp_path):
    root = tmp_path / "MUXP"
    root.mkdir()
    path = root / "side-effect.muxp"
    path.write_text(
        _muxp(command="unflatten_default_apt"), encoding="utf-8"
    )
    files = discover_muxp_files(str(tmp_path), "MUXP", "+34+132")
    assert files[0].update["commands"][0]["command"] == "unflatten_default_apt"


def test_manifest_is_written_atomically(tmp_path):
    path = write_manifest(str(tmp_path), {"result": "no_match", "tile": "+34+132"})
    assert json.loads((tmp_path / "Ortho4XP_muxp.json").read_text(encoding="utf-8")) == {
        "result": "no_match",
        "tile": "+34+132",
    }
    assert path.endswith("Ortho4XP_muxp.json")
