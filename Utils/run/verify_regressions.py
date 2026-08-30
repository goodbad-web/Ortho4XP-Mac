#!/usr/bin/env python3
"""Run deterministic regression checks without changing the repository data."""

from __future__ import annotations

import bz2
import hashlib
import os
import queue
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.chdir(ROOT)


class SkipTest(Exception):
    pass


def atom(name: str, payload: bytes) -> bytes:
    return name.encode("ascii") + struct.pack("<I", 8 + len(payload)) + payload


def dsf_with_nmed(nmed: bytes = b"nmed-data", include_smed: bool = False) -> bytes:
    nfed = atom("NFED", atom("NMED", nmed))
    atoms = nfed
    if include_smed:
        elevation = struct.pack("<" + "h" * 64, *([100] * 64))
        bathymetry = struct.pack("<" + "h" * 64, *([200] * 64))
        atoms += atom("SMED", atom("ELEV", elevation) + atom("BATH", bathymetry))
    body = b"XPLNEDSF" + struct.pack("<I", 1) + atoms
    return body + hashlib.md5(body).digest()


def osm_payload() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="verify">
  <node id="1" lat="35.0000000" lon="139.0000000">
    <tag k="name" v="A &amp; B &quot;quoted&quot;"/>
  </node>
  <node id="2" lat="35.1000000" lon="139.1000000"/>
  <node id="3" lat="35.1000000" lon="139.0000000"/>
  <way id="10" version="1">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="3"/>
    <nd ref="1"/>
    <tag k="natural" v="coastline"/>
  </way>
</osm>
"""


def fake_executable(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_global_scenery_resolution(tmp: Path, modules: dict[str, object]) -> None:
    fnames = modules["FNAMES"]
    lat, lon = 35, 139
    relative = Path("Earth nav data") / (fnames.long_latlon(lat, lon) + ".dsf")

    direct_root = tmp / "direct"
    direct_file = direct_root / relative
    direct_file.parent.mkdir(parents=True)
    direct_file.write_bytes(b"direct")
    assert fnames.global_scenery_dsf_candidates(str(direct_root), lat, lon) == [
        str(direct_file)
    ]

    parent_root = tmp / "Global Scenery"
    child_file = parent_root / "X-Plane 12 Global Scenery" / relative
    child_file.parent.mkdir(parents=True)
    child_file.write_bytes(b"child")
    candidates = fnames.global_scenery_dsf_candidates(str(parent_root), lat, lon)
    assert candidates == [str(child_file)]

    installation_root = tmp / "X-Plane"
    nested_file = installation_root / "Global Scenery" / "X-Plane 12 Global Scenery" / relative
    nested_file.parent.mkdir(parents=True)
    nested_file.write_bytes(b"nested")
    assert fnames.global_scenery_dsf_candidates(str(installation_root), lat, lon) == [
        str(nested_file)
    ]

    second_file = parent_root / "X-Plane 11 Global Scenery" / relative
    second_file.parent.mkdir(parents=True)
    second_file.write_bytes(b"second")
    assert len(fnames.global_scenery_dsf_candidates(str(parent_root), lat, lon)) == 2
    assert fnames.resolve_global_scenery_dsf(str(parent_root), lat, lon) is None
    assert fnames.global_scenery_dsf_candidates(str(tmp / "missing"), lat, lon) == []


def test_dsf_extraction(tmp: Path, modules: dict[str, object]) -> None:
    dsf = modules["DSF"]
    ovl = modules["OVL"]
    fnames = modules["FNAMES"]
    source_root = tmp / "scenery"
    relative = Path("Earth nav data") / (fnames.long_latlon(35, 139) + ".dsf")
    source = source_root / "X-Plane 12 Global Scenery" / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(dsf_with_nmed())
    tmp_dir = tmp / "runtime-tmp"
    old_root, old_tmp = ovl.custom_overlay_src, fnames.Tmp_dir
    try:
        ovl.custom_overlay_src = str(source_root)
        fnames.Tmp_dir = str(tmp_dir)
        result = dsf.extract_elevation_and_bathymetry_data(35, 139)
        assert result == (b"nmed-data", b"")
        assert not (tmp_dir / "+35+139.dsf").exists()

        source.write_bytes(dsf_with_nmed(include_smed=True))
        result = dsf.extract_elevation_and_bathymetry_data(35, 139)
        assert result is not None and result[0] == b"nmed-data"
        assert result[1].startswith(b"ELEV") and b"BATH" in result[1]

        source.write_bytes(b"XPLNEDSF-broken")
        assert dsf.extract_elevation_and_bathymetry_data(35, 139) is None

        source.write_bytes(b"7z fake archive")
        bad_7z = fake_executable(
            tmp / "7z-fails",
            "import sys\nsys.exit(7)\n",
        )
        old_unzip = ovl.unzip_cmd
        try:
            ovl.unzip_cmd = str(bad_7z)
            assert dsf.extract_elevation_and_bathymetry_data(35, 139) is None
        finally:
            ovl.unzip_cmd = old_unzip
    finally:
        ovl.custom_overlay_src, fnames.Tmp_dir = old_root, old_tmp


def test_osm_cache_and_xml(tmp: Path, modules: dict[str, object]) -> None:
    osm = modules["OSM"]
    fnames = modules["FNAMES"]
    payload = osm_payload()
    layer = osm.OSM_layer()
    assert layer.update_dicosm(payload) == 1
    assert layer.dicosmtags["n"][next(iter(layer.dicosmtags["n"]))]["name"] == 'A & B "quoted"'

    cached = tmp / "roundtrip.osm.bz2"
    assert layer.write_to_file(str(cached)) == 1
    assert cached.exists() and not (tmp / "roundtrip.osm.bz2.tmp.bz2").exists()
    with bz2.open(cached, "rb") as source:
        roundtrip = source.read()
    parsed = osm.ElementTree.fromstring(roundtrip)
    assert parsed.tag == "osm"
    assert 'A &amp; B "quoted"' in roundtrip.decode("utf-8")
    roundtrip_layer = osm.OSM_layer()
    assert roundtrip_layer.update_dicosm(str(cached)) == 1
    assert roundtrip_layer.dicosmtags["n"][-1]["name"] == 'A & B "quoted"'

    layer.update_dicosm(b"<osm><node id=\"broken\">")
    assert not layer.dicosmn and not layer.dicosmw

    cache = tmp / "cache.osm.bz2"
    cache.write_bytes(b"not xml")
    old_cached = fnames.osm_old_cached
    cached_name = fnames.osm_cached
    original_get = osm.get_overpass_data
    try:
        fnames.osm_cached = lambda lat, lon, suffix: str(cache)
        fnames.osm_old_cached = lambda lat, lon, query: str(tmp / "no-old-cache")
        osm.get_overpass_data = lambda query, bbox, server_code=None: payload
        refetched = osm.OSM_layer()
        assert osm.OSM_queries_to_OSM_layer(
            ['way["natural"="coastline"]'],
            refetched,
            35,
            139,
            cached_suffix="coastline",
        ) == 1
        assert cache.exists() and (tmp / "cache.osm.bz2.bad").exists()
        assert refetched.dicosmn
    finally:
        fnames.osm_cached = cached_name
        fnames.osm_old_cached = old_cached
        osm.get_overpass_data = original_get

    already_bad = Path(str(cache) + ".bad")
    already_bad.write_bytes(b"old")
    cache.write_bytes(b"new bad")
    assert osm._quarantine_osm_cache(str(cache)) is True
    assert Path(str(cache) + ".bad.1").exists()

    # If one legacy query cache is corrupt after another was accepted, the
    # accepted result must not be mixed with a fresh response.
    query_one = 'way["natural"="coastline"]'
    query_two = 'way["waterway"="riverbank"]'
    old_cache_paths = {
        query_one: tmp / "legacy-one.osm.bz2",
        query_two: tmp / "legacy-two.osm.bz2",
    }
    with bz2.open(old_cache_paths[query_one], "wb") as output:
        output.write(payload)
    old_cache_paths[query_two].write_bytes(b"corrupt legacy cache")
    calls: list[str] = []
    old_cached = fnames.osm_old_cached
    cached_name = fnames.osm_cached
    original_get = osm.get_overpass_data
    try:
        fnames.osm_cached = lambda lat, lon, suffix: str(tmp / "mixed.osm.bz2")
        fnames.osm_old_cached = lambda lat, lon, query: str(old_cache_paths[query])
        osm.get_overpass_data = lambda query, bbox, server_code=None: calls.append(query) or payload
        mixed_layer = osm.OSM_layer()
        assert osm.OSM_queries_to_OSM_layer(
            [query_one, query_two], mixed_layer, 35, 139, cached_suffix="mixed"
        ) == 1
        assert calls == [query_one, query_two]
        assert Path(str(old_cache_paths[query_two]) + ".bad").exists()
    finally:
        fnames.osm_old_cached = old_cached
        fnames.osm_cached = cached_name
        osm.get_overpass_data = original_get


def test_overpass_fallback(tmp: Path, modules: dict[str, object]) -> None:
    osm = modules["OSM"]
    calls: list[str] = []
    valid = osm_payload()

    class Response:
        def __init__(self, status: int, content: bytes):
            self.status_code = status
            self.content = content
            self.text = content.decode("utf-8", errors="replace")

    class Session:
        def post(self, url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                return Response(503, b"busy")
            if len(calls) == 2:
                return Response(200, b"<osm>")
            return Response(200, valid)

    class AlwaysFailSession:
        def post(self, url, **kwargs):
            calls.append(url)
            return Response(503, b"busy")

    old_session = osm.requests.Session
    old_tentatives = osm.max_osm_tentatives
    old_sleep = osm.time.sleep
    old_choice = osm.overpass_server_choice
    try:
        osm.max_osm_tentatives = 1
        osm.overpass_server_choice = "DE"
        osm.requests.Session = Session
        result = osm.get_overpass_data('way["natural"="coastline"]', (35, 139, 36, 140))
        assert result == valid
        assert calls[:3] == [
            osm.overpass_servers["DE"],
            osm.overpass_servers["LZ"],
            osm.overpass_servers["CH"],
        ]

        calls.clear()
        osm.requests.Session = AlwaysFailSession
        osm.time.sleep = lambda _: None
        assert osm.get_overpass_data("way[]", (35, 139, 36, 140)) == 0
        assert len(calls) == 5
        calls.clear()
        osm.max_osm_tentatives = 10
        assert osm.get_overpass_data("way[]", (35, 139, 36, 140)) == 0
        assert len(calls) == 5 * osm.max_osm_tentatives and len(set(calls)) == 5
    finally:
        osm.requests.Session = old_session
        osm.max_osm_tentatives = old_tentatives
        osm.time.sleep = old_sleep
        osm.overpass_server_choice = old_choice


def test_parallel_and_activation(tmp: Path, modules: dict[str, object]) -> None:
    parallel = modules["PARALLEL"]
    tile = modules["TILE"]
    ui = modules["UI"]
    ui.red_flag = False

    work = queue.Queue()
    work.put((1,))
    assert parallel.parallel_execute(lambda value: 0, work, 2) == 0

    def raises(value):
        raise RuntimeError("worker failure")

    work = queue.Queue()
    work.put((1,))
    assert parallel.parallel_execute(raises, work, 2) == 0

    active = tmp / "active.dsf"
    temporary = tmp / "active.dsf.tmp"
    active.write_bytes(b"old")
    temporary.write_bytes(b"new")
    tile._activate_dsf(str(temporary), str(active))
    assert active.read_bytes() == b"new"
    assert (tmp / "active.dsf.bak").read_bytes() == b"old"
    assert not temporary.exists()

    active.write_bytes(b"stable")
    try:
        tile._activate_dsf(str(tmp / "missing.dsf.tmp"), str(active))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing DSF temporary file was accepted")
    assert active.read_bytes() == b"stable"

    # A later-stage failure must discard an unactivated DSF temporary file.
    class Tile:
        lat = 35
        lon = 139
        build_dir = str(tmp / "tile-cleanup")

    temporary_dsf = Path(Tile.build_dir) / "Earth nav data" / "+30+130" / "+35+139.dsf.tmp"
    temporary_dsf.parent.mkdir(parents=True)
    temporary_dsf.write_bytes(b"incomplete")
    original_build = tile._build_tile
    try:
        tile._build_tile = lambda _: 0
        assert tile.build_tile(Tile()) == 0
        assert not temporary_dsf.exists()
    finally:
        tile._build_tile = original_build


def test_missing_dds_output(tmp: Path, modules: dict[str, object]) -> None:
    img = modules["IMG"]
    ui = modules["UI"]
    from PIL import Image

    source = tmp / "source.png"
    Image.new("RGB", (4, 4), (12, 34, 56)).save(source)
    fake_converter = fake_executable(
        tmp / "converter-succeeds-without-output",
        "import sys\nsys.exit(0)\n",
    )

    class Tile:
        lat = 35
        lon = 139
        build_dir = str(tmp / "tile")
        imprint_masks_to_dds = False
        mask_zl = 15

    Path(Tile.build_dir).mkdir()
    old_converter = img.dds_convert_cmd
    old_providers = img.providers_dict
    old_combined = img.local_combined_providers_dict
    old_as_helper = img.as_helper_cmd
    old_root = ui.Ortho4XP_dir
    old_dds_converter = getattr(ui, "dds_converter", None)
    old_dds_format = getattr(ui, "dds_format", None)
    old_sleep = img.time.sleep
    try:
        img.dds_convert_cmd = str(fake_converter)
        img.providers_dict = {}
        img.local_combined_providers_dict = {}
        img.as_helper_cmd = None
        ui.Ortho4XP_dir = str(tmp)
        ui.dds_converter = "nvcompress"
        ui.dds_format = "BC3"
        img.time.sleep = lambda _: None
        result = img.convert_texture(
            Tile(), 0, 0, 12, "TEST", prepared_file=str(source)
        )
        assert result == 0
        output = Path(Tile.build_dir) / "textures"
        assert not output.exists() or not list(output.glob("*.dds"))

        malformed_converter = fake_executable(
            tmp / "converter-writes-invalid-dds",
            "import sys\n"
            "with open(sys.argv[-1], 'wb') as stream:\n"
            "    stream.write(b'DDS ' + b'\\0' * 125)\n"
            "sys.exit(0)\n",
        )
        img.dds_convert_cmd = str(malformed_converter)
        result = img.convert_texture(
            Tile(), 0, 0, 12, "TEST", prepared_file=str(source)
        )
        assert result == 0
        assert not output.exists() or not list(output.glob("*.dds"))
    finally:
        img.dds_convert_cmd = old_converter
        img.providers_dict = old_providers
        img.local_combined_providers_dict = old_combined
        img.as_helper_cmd = old_as_helper
        ui.Ortho4XP_dir = old_root
        img.time.sleep = old_sleep
        if old_dds_converter is None:
            try:
                del ui.dds_converter
            except AttributeError:
                pass
        else:
            ui.dds_converter = old_dds_converter
        if old_dds_format is None:
            try:
                del ui.dds_format
            except AttributeError:
                pass
        else:
            ui.dds_format = old_dds_format


def test_levels_route(modules: dict[str, object]) -> None:
    img = modules["IMG"]
    img.initialize_color_filters_dict()
    assert "GeoPunt2012" in img.color_filters_dict
    assert not img.gpu_batch_color_filter_supported("GeoPunt2012")
    assert img.gpu_batch_color_filter_supported("none")


def test_cli_exit_code(tmp: Path) -> None:
    for name in ("src", "Utils", "Providers", "Extents", "Filters"):
        (tmp / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp / "Ortho4XP.cfg").symlink_to(ROOT / "Ortho4XP.cfg")
    result = subprocess.run(
        [sys.executable, str(ROOT / "Ortho4XP.py"), "not-a-latitude"],
        cwd=tmp,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 2, result.stdout


def test_gui_trace() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SkipTest("Tk display/window server is unavailable")
    gui_module = __import__("O4_GUI_Utils")
    ui = __import__("O4_UI_Utils")
    try:
        app = gui_module.Ortho4XP_GUI()
    except Exception as error:
        if "display" in str(error).lower() or "window server" in str(error).lower():
            raise SkipTest(str(error))
        raise
    try:
        if app.map_list:
            app.default_website.set(app.map_list[0])
        app.default_zl.set("15")
        app.update_idletasks()
    finally:
        ui.gui = None
        try:
            app.after_cancel(app.callback_pgrb)
            app.after_cancel(app.callback_status)
            app.after_cancel(app.callback_console)
        except Exception:
            pass
        try:
            sys.stdout = app.stdout_orig
        except Exception:
            pass
        app.destroy()


def main() -> int:
    import O4_DSF_Utils as dsf
    import O4_File_Names as fnames
    import O4_Imagery_Utils as img
    import O4_OSM_Utils as osm
    import O4_Overlay_Utils as ovl
    import O4_Parallel_Utils as parallel
    import O4_Tile_Utils as tile
    import O4_UI_Utils as ui

    ui.verbosity = -1
    ui.log = False
    modules = {
        "DSF": dsf,
        "FNAMES": fnames,
        "IMG": img,
        "OSM": osm,
        "OVL": ovl,
        "PARALLEL": parallel,
        "TILE": tile,
        "UI": ui,
    }
    tests = [
        ("Global Scenery resolution", test_global_scenery_resolution),
        ("DSF extraction and archive failure", test_dsf_extraction),
        ("OSM XML and cache quarantine", test_osm_cache_and_xml),
        ("Overpass status and fallback", test_overpass_fallback),
        ("Parallel failure and DSF activation", test_parallel_and_activation),
        ("DDS output validation", test_missing_dds_output),
        ("levels CPU routing", test_levels_route),
        ("CLI invalid argument exit", test_cli_exit_code),
        ("Tk trace_add lifecycle", test_gui_trace),
    ]
    failures: list[str] = []
    skips: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ortho4xp-regressions-") as temp_dir:
        tmp = Path(temp_dir)
        for name, test in tests:
            try:
                if name == "CLI invalid argument exit":
                    test(tmp)
                elif name in {
                    "Global Scenery resolution",
                    "DSF extraction and archive failure",
                    "OSM XML and cache quarantine",
                    "Overpass status and fallback",
                    "Parallel failure and DSF activation",
                    "DDS output validation",
                }:
                    test(tmp, modules)
                else:
                    test(modules) if name == "levels CPU routing" else test()
            except SkipTest as error:
                skips.append(f"{name}: {error}")
                print(f"SKIP: {name} ({error})")
            except Exception as error:
                failures.append(f"{name}: {error}")
                print(f"FAIL: {name}: {error}")
            else:
                print(f"PASS: {name}")

    print(f"Summary: {len(tests) - len(failures) - len(skips)} passed, {len(skips)} skipped, {len(failures)} failed")
    for failure in failures:
        print(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
