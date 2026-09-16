from pathlib import Path
import json
import sys

import pytest


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_OSM_Utils as OSM  # noqa: E402


BUNDLED_CACHE_PATHS = sorted(
    (Path(__file__).parents[1] / "Extents" / "LowRes").glob("*.osm.bz2")
)


@pytest.mark.parametrize(
    "cache_path",
    BUNDLED_CACHE_PATHS,
    ids=lambda path: path.stem,
)
def test_bundled_lowres_osm_cache_has_verified_manifest_and_loads(cache_path):
    assert len(BUNDLED_CACHE_PATHS) == 27
    manifest_path = Path(str(cache_path) + ".manifest.json")
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "bundled"
    assert manifest["responses"] == []

    layer = OSM.OSM_layer()
    assert (
        OSM.OSM_query_to_OSM_layer(
            None,
            "",
            layer,
            "all",
            cached_file_name=str(cache_path),
        )
        == 1
    )
    assert layer.last_cache_info["source"] == "verified-cache"
    assert layer.dicosmn or layer.dicosmfirst["w"] or layer.dicosmfirst["r"]
