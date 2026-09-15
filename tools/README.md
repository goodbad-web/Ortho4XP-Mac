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

The Overpass query should use `out geom`, for example. For larger areas,
split the bbox into multiple requests and repeat `--overpass-json`; duplicate
way IDs at tile boundaries are ignored:

```text
[out:json][timeout:120];way["building"](34.600,133.900,34.630,133.970);out geom;
```

```sh
.venv/bin/python tools/xp_buildings.py \
  --lat 34 --lon 133 \
  --overpass-json /tmp/okayama-west.json \
  --overpass-json /tmp/okayama-east.json \
  --asset-root /tmp/own-xplane-objects \
  --output /tmp/zz_Ortho4XP_AI_Buildings_+34+133 \
  --dsftool Utils/mac/DSFTool
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

`--asset-root` is required and must contain those four OBJ files and their
referenced texture files. Generation fails before creating the package when the
directory or any base OBJ is missing. The entire directory is copied into the
generated package's `objects/` directory; no files are copied from Japan Pro or
X-World.

When the asset directory comes from `create_demo_assets.py`, the DSF selects a
small/medium/large and low/mid/high variant according to each footprint's
estimated dimensions. If a custom asset directory only contains the four base
OBJ files, the generator falls back to those base files.

The generator does not copy Japan Pro or X-World files. `--mode extend` mixes
with lower-priority libraries; the default `replace` mode uses `EXPORT` for
the selected tile region.

Do not use `--exclude-rect` until the models have been compared in X-Plane.
The option writes both `sim/exclude_obj` and `sim/exclude_fac`, because
X-World buildings may be represented by objects or facade polygons. Exclusion
is rectangular and may remove unrelated scenery inside the rectangle. The
default is no exclusion, which is safest for the first comparison.

## Blender + ComfyUI asset path

`create_demo_assets.py` is only a display-path smoke test. For reusable
building assets, generate four category textures with ComfyUI, then let
Blender generate the family/size/height variants. The texture batch client
uses ComfyUI's API-format workflow and keeps the seed and prompt in the job
file:

```sh
.venv/bin/python tools/comfyui_texture_batch.py \
  --workflow tools/comfyui_building_texture_api.json \
  --jobs tools/comfyui_building_texture_jobs.json \
  --output-dir /tmp/ortho4xp-building-textures
```

The checked-in workflow targets `SDXL/sd_xl_turbo_1.0_fp16.safetensors` and
uses four steps with CFG 1.0. If the file is stored under another ComfyUI
checkpoint path, change only `2.inputs.ckpt_name`. Use `--dry-run` to validate
the four expanded workflows without connecting to ComfyUI. The generated
files are `jp_house.png`, `jp_apartment.png`, `jp_commercial.png`, and
`jp_industrial.png`.

Run the asset generator inside Blender. It creates 40 reusable OBJ8 assets,
copies the category textures to each variant, writes `asset-manifest.json`,
and saves an optional inspectable `.blend`:

```sh
blender --background --python tools/blender_generate_assets.py -- \
  --output /tmp/ortho4xp-building-assets \
  --texture-root /tmp/ortho4xp-building-textures \
  --blend-output /tmp/ortho4xp-building-assets/building-families.blend
```

The output directory can then be passed unchanged as `--asset-root` to
`xp_buildings.py`. Geometry is generated from a small number of families and
the existing footprint/height classifier selects a variant, so this does not
create a unique heavy mesh or a unique 4K texture for every OSM building.
The OBJ8 UVs keep the front facade on the main image area and use narrower
vertical regions for the sides/back plus an upper band for roofs. This avoids
repeating the whole front photograph on every face; separate side and roof
generation can be added later when a fully seamless material is required.
The generator also has `--obj8-only` for environments that already have the
texture files but do not have Blender installed; that path does not claim to
produce a `.blend` file.

## MEGA CITY TOWERS landmark package

For a lightweight landmark overlay of Yao's MEGA CITY TOWERS, use the
dedicated procedural generator.  It creates separate WEST, EAST, and shared
podium OBJ8 objects for tile `+34+135`, each with near/mid/far LOD sections:

```sh
.venv/bin/python tools/create_megacity_towers.py \
  --output /tmp/zz_MegaCityTowers_+34+135 \
  --config tools/megacity_towers_config.json \
  --dsftool Utils/mac/DSFTool \
  --nvcompress Utils/mac/nvcompress
```

The config keeps the EAST public map reference and the provisional WEST
offset separate.  Verify the WEST footprint in a map or X-Plane before adding
an exclusion rectangle.  The facade atlas is self-owned and can be replaced
by `jp_*` ComfyUI output without changing the DSF or tower geometry.

For a geometry-guided facade pass, use
`comfyui_building_controlnet_api.json` with the batch client and provide a
facade guide plus a Blender-rendered grayscale depth guide for each job. The
workflow applies the built-in Canny preprocessor and then the SDXL Canny and
Depth ControlNets. The default checkpoint is `sd_xl_base_1.0.safetensors`,
which is the compatibility reference; change `2.ckpt_name` to
`SDXL/sd_xl_turbo_1.0_fp16.safetensors` for a fast comparison run. The
matching job list is `comfyui_building_controlnet_jobs.json`:

```sh
.venv/bin/python tools/comfyui_texture_batch.py \
  --workflow tools/comfyui_building_controlnet_api.json \
  --jobs tools/comfyui_building_controlnet_jobs.json \
  --output-dir /tmp/ortho4xp-building-controlnet
```

The depth guide is intentionally an external input: this installation has no
Depth Anything/MiDaS preprocessor node. Render it from the same Blender
camera and geometry as the facade guide so the ControlNet does not alter the
building footprint. The checked-in Blender helper renders paired guides for
the four reusable categories:

```sh
tools/run_blender_with_metal_preflight.sh --background \
  --python tools/render_controlnet_guides.py -- \
  --blend "/Users/hiroshi/X-Plane/Custom Scenery/zzz_Ortho4XP_AI_Buildings_+34+133_uv_test/objects/building-families.blend" \
  --output /tmp/ortho4xp-controlnet-inputs
```

Do not use a generated texture as the depth guide.

To create an inspectable Blender scene, run the same script through Blender
with `--blend-output`.  Blender is used only at generation time; X-Plane loads
the resulting OBJ8/DDS files and does not run ComfyUI during flight.

On macOS, invoke Blender through the Metal preflight wrapper so a locked or
headless session is reported clearly instead of triggering Blender's early
Metal crash:

```sh
tools/run_blender_with_metal_preflight.sh \
  --background \
  --python tools/create_megacity_towers.py \
  -- \
  --output /tmp/zz_MegaCityTowers_+34+135 \
  --config tools/megacity_towers_config.json \
  --dsftool Utils/mac/DSFTool \
  --nvcompress Utils/mac/nvcompress \
  --blend-output /tmp/zz_MegaCityTowers_+34+135/megacity-towers.blend
```

Set `OR4XP_BLENDER_BIN` when using another Blender installation.  The probe
does not change Blender preferences or the installed application.

## The Scene Johoku landmark package

For a lightweight procedural landmark overlay of The Scene Johoku in Nagoya's
Kita ward, use the dedicated generator.  It creates an elliptical 160 m Astro
Tower and two approximately 31 m Star buildings, each with near/mid/far LOD
sections:

```sh
.venv/bin/python tools/create_scene_johoku.py \
  --output /tmp/zz_TheSceneJohoku_+35+136 \
  --config tools/scene_johoku_config.json \
  --nvcompress Utils/mac/nvcompress
```

The initial Astro Tower coordinate in the JSON follows the supplied Google
Earth reference image, with small east/west offsets for the Star buildings.
Check the three footprints in X-Plane before adding any exclusion rectangle.
The package contains four separate OBJ8 objects: the Astro Tower, two Star
buildings, and a shared low podium.  The model uses a self-owned generic residential
facade atlas, so it does not copy photographs or logos.  Blender is optional;
use `--blend-output` through `run_blender_with_metal_preflight.sh` when an
inspectable `.blend` scene is useful.
