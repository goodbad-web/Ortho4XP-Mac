import sys
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_GUI_Utils as GUI  # noqa: E402


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_tile_from_interface_does_not_override_current_settings_from_saved_cfg(tmp_path, monkeypatch):
    config_path = tmp_path / (
        "Ortho4XP_" + GUI.FNAMES.short_latlon(34, 133) + ".cfg"
    )
    config_path.write_text("default_website=OLD\ndefault_zl=12\n", encoding="utf-8")

    class _Tile:
        build_dir = str(tmp_path)

        def read_from_config(self, *args, **kwargs):
            raise AssertionError("saved tile config must not be loaded implicitly")

    tile = _Tile()
    monkeypatch.setattr(GUI.CFG, "Tile", lambda lat, lon, build_dir: tile)

    gui = GUI.Ortho4XP_GUI.__new__(GUI.Ortho4XP_GUI)
    gui.get_lat_lon = lambda: (34, 133)
    gui.custom_build_dir = _Value(str(tmp_path))

    assert gui.tile_from_interface() is tile
