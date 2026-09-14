# X-Plane building enhancement generator

`xp_buildings.py` generates a separate, offline-first Custom Scenery package
from building footprints in OSM XML or GeoJSON. It does not modify existing
Ortho4XP tiles, X-World, Japan Pro, or `scenery_packs.ini`.

## First PoC

```sh
.venv/bin/python tools/xp_buildings.py \
  --lat 34 --lon 133 \
  --osm /path/to/buildings.osm \
  --asset-root /path/to/own-xplane-objects \
  --output /path/to/Custom\ Scenery/zz_Ortho4XP_AI_Buildings_+34+133
```

The output contains a readable text DSF, `library.txt`, and
`generation-report.json`. Add `--dsftool Utils/mac/DSFTool` to create a binary
DSF.

The generated package expects self-owned assets at:

```text
objects/jp_house_a.obj
objects/jp_apartment_a.obj
objects/jp_commercial_a.obj
objects/jp_industrial_a.obj
```

`--asset-root` must contain those four OBJ files and their referenced texture
files. The entire directory is copied into the generated package's `objects/`
directory; no files are copied from Japan Pro or X-World.

The generator does not copy Japan Pro or X-World files. `--mode extend` mixes
with lower-priority libraries; the default `replace` mode uses `EXPORT` for
the selected tile region.

Do not use `--exclude-rect` until the models have been compared in X-Plane.
X-Plane object exclusion is rectangular and may remove unrelated scenery.
The default is no exclusion, which is safest for the first comparison.
