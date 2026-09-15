#!/usr/bin/env python3
"""Render paired facade and depth guides from a generated Blender asset pack.

Run inside Blender, for example::

    tools/run_blender_with_metal_preflight.sh --background \
      --python tools/render_controlnet_guides.py -- \
      --blend /path/to/building-families.blend \
      --output /tmp/ortho4xp-controlnet-inputs

The blend contains many reusable variants at the same origin. Only the
requested variant is made visible for each render, so the guide images do not
contain overlapping buildings. The facade guide is a neutral Workbench render
for the built-in Canny node; the depth guide is the normalized Z pass from the
same camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # Blender-only script.
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by Blender only.
    bpy = None
    Vector = None


CATEGORIES = ("house", "apartment", "commercial", "industrial")
VARIANTS = {
    "house": "jp_house_medium_mid",
    "apartment": "jp_apartment_medium_mid",
    "commercial": "jp_commercial_medium_mid",
    "industrial": "jp_industrial_medium_mid",
}


def _script_argv(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[sys.argv.index("--") + 1 :]) if "--" in sys.argv else []
    return values if argv is None else argv


def _bounds(obj: object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners))),
        Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners))),
    )


def _look_at(camera: object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _configure_camera(scene: object, obj: object, resolution: int) -> None:
    low, high = _bounds(obj)
    center = (low + high) * 0.5
    span = max(high.x - low.x, high.y - low.y) * 1.12
    distance = max(high.z - low.z, high.x - low.x, high.y - low.y) * 3.0

    camera_data = bpy.data.cameras.new("ControlNetGuideCamera")
    camera = bpy.data.objects.new("ControlNetGuideCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = Vector((center.x, center.y, low.z - distance))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = span
    _look_at(camera, center)
    scene.camera = camera

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False


def _visible_variant(name: str) -> object:
    target = bpy.data.objects.get(name)
    if target is None:
        raise RuntimeError(f"variant not found in blend: {name}")
    for obj in bpy.data.objects:
        obj.hide_render = obj != target
    return target


def _render_facade(scene: object, path: Path) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "SINGLE"
    shading.single_color = (0.72, 0.72, 0.72)
    shading.show_shadows = True
    shading.show_cavity = True
    shading.cavity_type = "WORLD"
    shading.curvature_ridge_factor = 1.5
    shading.curvature_valley_factor = 1.5
    scene.use_nodes = False
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def _render_depth(scene: object, target: object, path: Path, original_materials: list[object]) -> None:
    """Render view distance as a portable 8-bit depth guide PNG."""
    material = bpy.data.materials.new("ControlNetGuideDepthMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    camera_data = nodes.new("ShaderNodeCameraData")
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = max(target.dimensions) * 4.0
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    if hasattr(map_range, "clamp"):
        map_range.clamp = True
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(camera_data.outputs["View Distance"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    target.data.materials.clear()
    target.data.materials.append(material)

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ControlNetGuideWorld")
    scene.world.color = (0.0, 0.0, 0.0)
    bpy.ops.render.render(write_still=True)

    target.data.materials.clear()
    for original in original_materials:
        target.data.materials.append(original)
    bpy.data.materials.remove(material)
    scene.render.image_settings.color_mode = "RGBA"


def render_guides(blend: Path, output: Path, resolution: int, categories: tuple[str, ...]) -> list[Path]:
    if bpy is None:
        raise RuntimeError("run this script with Blender")
    if not blend.is_file():
        raise FileNotFoundError(blend)

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    generated: list[Path] = []
    for category in categories:
        target = _visible_variant(VARIANTS[category])
        _configure_camera(scene, target, resolution)
        facade = output / f"building_facade_{category}.png"
        depth = output / f"building_depth_{category}.png"
        _render_facade(scene, facade)
        original_materials = list(target.data.materials)
        _render_depth(scene, target, depth, original_materials)
        generated.extend((facade, depth))
    for obj in bpy.data.objects:
        obj.hide_render = False
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    args = parser.parse_args(_script_argv(argv))
    if args.resolution < 64 or args.resolution > 4096:
        parser.error("--resolution must be between 64 and 4096")
    categories = tuple(args.categories or CATEGORIES)
    generated = render_guides(args.blend, args.output, args.resolution, categories)
    for path in generated:
        print(f"generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
