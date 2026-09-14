# X-Plane building enhancement generator

`xp_buildings.py` generates a separate, offline-first Custom Scenery package
from building footprints in OSM XML, GeoJSON, or Overpass JSON. It does not modify existing
Ortho4XP tiles, X-World, Japan Pro, or `scenery_packs.ini`.

## First PoC

```sh
.venv/bin/python tools/xp_buildings.py \
  --lat 34 --lon 133 \
  --overpass-json /path/to/buildings.json \
  --asset-root /path/to/own-xplane-objects \
  --output /path/to/Custom\ Scenery/zz_Ortho4XP_AI_Buildings_+34+133
```

The output contains a readable text DSF, `library.txt`, and
`generation-report.json`. Add `--dsftool Utils/mac/DSFTool` to create a binary
DSF.

The Overpass query should use `out geom`, for example:

```text
[out:json][timeout:120];way["building"](34.600,133.900,34.630,133.970);out geom;
```

The generated package expects self-owned assets at:

```text
objects/jp_house_a.obj
objects/jp_apartment_a.obj
objects/jp_commercial_a.obj
objects/jp_industrial_a.obj
```

For a display-path smoke test, create simple self-owned assets first:

```sh
.venv/bin/python tools/create_demo_assets.py /tmp/own-xplane-objects
```

`--asset-root` must contain those four OBJ files and their referenced texture
files. The entire directory is copied into the generated package's `objects/`
directory; no files are copied from Japan Pro or X-World.

When the asset directory comes from `create_demo_assets.py`, the DSF selects a
small/medium/large and low/mid/high variant according to each footprint's
estimated dimensions. If a custom asset directory only contains the four base
OBJ files, the generator falls back to those base files.

The generator does not copy Japan Pro or X-World files. `--mode extend` mixes
with lower-priority libraries; the default `replace` mode uses `EXPORT` for
the selected tile region.

Do not use `--exclude-rect` until the models have been compared in X-Plane.
X-Plane object exclusion is rectangular and may remove unrelated scenery.
The default is no exclusion, which is safest for the first comparison.
