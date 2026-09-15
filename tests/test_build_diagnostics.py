import sys
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Config_Utils as CFG  # noqa: E402
import O4_OSM_Utils as OSM  # noqa: E402
import O4_Vector_Map as VMAP  # noqa: E402


def _config_tile(tmp_path):
    tile = CFG.Tile.__new__(CFG.Tile)
    tile.build_dir = str(tmp_path)
    tile.lat = 1
    tile.lon = 2
    for variable in CFG.list_tile_vars:
        setattr(tile, variable, CFG.cfg_vars[variable].get("default"))
    return tile


def test_tile_config_write_is_atomic_and_keeps_previous_config_on_activation_error(
    tmp_path, monkeypatch
):
    tile = _config_tile(tmp_path)
    config_path = tmp_path / "tile.cfg"
    config_path.write_text("old configuration\n", encoding="utf-8")
    real_replace = CFG.os.replace
    replace_calls = []

    def fail_second_replace(source, destination):
        replace_calls.append((source, destination))
        if len(replace_calls) == 2:
            raise OSError("simulated activation failure")
        return real_replace(source, destination)

    monkeypatch.setattr(CFG.os, "replace", fail_second_replace)
    assert tile.write_to_config(str(config_path)) == 0
    assert config_path.read_text(encoding="utf-8") == "old configuration\n"
    assert not Path(str(config_path) + ".tmp").exists()


def test_overpass_logs_status_attempt_server_and_success_server(monkeypatch):
    class Response:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content
            self.text = content.decode("utf-8", errors="replace")

    responses = iter(
        (
            Response(429, b"rate limited"),
            Response(200, b"<osm></osm>"),
        )
    )

    class Session:
        def post(self, *args, **kwargs):
            return next(responses)

    logs = []
    monkeypatch.setattr(OSM.requests, "Session", lambda: Session())
    monkeypatch.setattr(OSM, "overpass_server_choice", "DE")
    monkeypatch.setattr(OSM, "max_osm_tentatives", 1)
    monkeypatch.setattr(OSM.UI, "logprint", lambda *args: logs.append(" ".join(map(str, args))))
    monkeypatch.setattr(OSM.UI, "red_flag", False)

    result = OSM.get_overpass_data('way["highway"="primary"]', (1, 2, 3, 4))

    assert result == b"<osm></osm>"
    joined = "\n".join(logs)
    assert "attempt= 1 server= DE" in joined
    assert "status= 429" in joined
    assert "success_server= LZ" in joined


def test_water_polygon_exception_is_reported_with_context(monkeypatch):
    from shapely.geometry import GeometryCollection, Polygon

    tile = SimpleNamespace(
        lat=1,
        lon=2,
        clean_bad_geometries=True,
        custom_water="",
        max_area=1000000,
        dem=SimpleNamespace(alt_vec=lambda value: value),
        min_area=1.0,
        water_simplification=1.0,
    )
    logs = []
    monkeypatch.setattr(VMAP.FNAMES, "custom_water", lambda lat, lon: "")
    monkeypatch.setattr(VMAP.FNAMES, "custom_water_dir", lambda lat, lon: "")
    monkeypatch.setattr(VMAP.OSM, "OSM_queries_to_OSM_layer", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        VMAP.OSM,
        "OSM_to_MultiPolygon",
        lambda *args, **kwargs: (Polygon([(0, 0), (1, 0), (0, 1)]), GeometryCollection()),
    )
    monkeypatch.setattr(
        VMAP.VECT,
        "MultiPolygon_to_Indexed_Polygons",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad geometry")),
    )
    monkeypatch.setattr(VMAP.UI, "logprint", lambda *args: logs.append(" ".join(map(str, args))))
    monkeypatch.setattr(VMAP.UI, "red_flag", False)

    assert VMAP.include_water(SimpleNamespace(), tile) == 0
    assert "Water polygon indexing failed" in "\n".join(logs)
