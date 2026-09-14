#!/usr/bin/env python3
"""Generate reusable procedural building assets from Blender.

Run this script with Blender's Python interpreter:

    blender --background --python tools/blender_generate_assets.py -- \
        --output /tmp/ortho4xp-building-assets \
        --texture-root /tmp/ortho4xp-building-textures

The geometry is deliberately family-based rather than one unique mesh per
building.  That keeps the resulting X-Plane overlay small while allowing the
footprint/height classifier in ``xp_buildings.py`` to select useful variants.
The script writes X-Plane OBJ8 files directly and, when run by Blender, also
writes one inspectable .blend containing the generated meshes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:  # Blender is available only when the script is run by Blender.
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by Blender only.
    bpy = None


SIZE_BUCKETS = ("small", "medium", "large")
HEIGHT_BUCKETS = ("low", "mid", "high")
VARIANT_FOOTPRINTS = {
    "small": (8.0, 6.0),
    "medium": (16.0, 10.0),
    "large": (28.0, 18.0),
}
VARIANT_HEIGHTS = {"low": 6.0, "mid": 11.0, "high": 18.0}
VARIANT_STEMS = {
    "house": "house",
    "apartments": "apartment",
    "commercial": "commercial",
    "industrial": "industrial",
}
BASE_DIMENSIONS = {
    "house": (9.0, 6.0, 6.0),
    "apartments": (18.0, 10.0, 18.0),
    "commercial": (22.0, 7.0, 16.0),
    "industrial": (30.0, 8.0, 24.0),
}

# The ComfyUI inputs are facade-oriented images rather than six-sided texture
# atlases. Keep the front readable, and use narrower, less repetitive strips
# for the other faces. UV V=1 is the top of the source image.
_FRONT_UV = ((0.02, 0.06), (0.02, 0.94), (0.98, 0.94), (0.98, 0.06))
_LEFT_SIDE_UV = ((0.10, 0.08), (0.10, 0.92), (0.30, 0.92), (0.30, 0.08))
_RIGHT_SIDE_UV = ((0.70, 0.08), (0.70, 0.92), (0.90, 0.92), (0.90, 0.08))
_BACK_UV = ((0.30, 0.08), (0.30, 0.92), (0.70, 0.92), (0.70, 0.08))
_ROOF_UV = ((0.12, 0.76), (0.12, 0.96), (0.88, 0.96), (0.88, 0.76))
_BOTTOM_UV = ((0.0, 0.0), (0.0, 0.08), (1.0, 0.08), (1.0, 0.0))


@dataclass
class MeshData:
    """Small, renderer-independent mesh representation."""

    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    face_uvs: list[tuple[tuple[float, float], ...]] = field(default_factory=list)

    def add_face(
        self,
        points: Iterable[tuple[float, float, float]],
        uvs: Iterable[tuple[float, float]] | None = None,
    ) -> None:
        points = list(points)
        if len(points) < 3:
            return
        start = len(self.vertices)
        self.vertices.extend(points)
        self.faces.append(tuple(range(start, start + len(points))))
        if uvs is None:
            uvs = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
        uvs = list(uvs)
        if len(uvs) != len(points):
            raise ValueError("face UV count must match face vertex count")
        self.face_uvs.append(tuple(uvs))


def _box(
    mesh: MeshData,
    width: float,
    height: float,
    depth: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    top: bool = True,
) -> None:
    """Add a closed X-Plane-oriented box, with +Y as up."""
    hx, hz = width / 2.0, depth / 2.0
    bottom = y
    upper = y + height
    corners = {
        "bl": (x - hx, bottom, z - hz),
        "br": (x + hx, bottom, z - hz),
        "fr": (x + hx, bottom, z + hz),
        "fl": (x - hx, bottom, z + hz),
        "tl": (x - hx, upper, z - hz),
        "tr": (x + hx, upper, z - hz),
        "ur": (x + hx, upper, z + hz),
        "ul": (x - hx, upper, z + hz),
    }
    mesh.add_face((corners["bl"], corners["br"], corners["fr"], corners["fl"]), _BOTTOM_UV)
    mesh.add_face((corners["bl"], corners["fl"], corners["ul"], corners["tl"]), _LEFT_SIDE_UV)
    mesh.add_face((corners["br"], corners["tr"], corners["ur"], corners["fr"]), _RIGHT_SIDE_UV)
    mesh.add_face((corners["fr"], corners["ur"], corners["ul"], corners["fl"]), _FRONT_UV)
    mesh.add_face((corners["bl"], corners["tl"], corners["tr"], corners["br"]), _BACK_UV)
    if top:
        mesh.add_face((corners["ul"], corners["ur"], corners["tr"], corners["tl"]), _ROOF_UV)


def _gabled_roof(mesh: MeshData, width: float, eave_y: float, depth: float, roof_height: float) -> None:
    """Add a simple long-ridge roof with low polygon count."""
    hx, hz = width / 2.0, depth / 2.0
    ridge_y = eave_y + roof_height
    front_left = (-hx, eave_y, -hz)
    front_right = (hx, eave_y, -hz)
    back_right = (hx, eave_y, hz)
    back_left = (-hx, eave_y, hz)
    front_ridge_left = (-hx, ridge_y, 0.0)
    front_ridge_right = (hx, ridge_y, 0.0)
    roof_uv = _ROOF_UV
    mesh.add_face((front_left, front_ridge_left, front_ridge_right, front_right), roof_uv)
    mesh.add_face((front_ridge_left, back_left, back_right, front_ridge_right), roof_uv)
    mesh.add_face((front_left, back_left, front_ridge_left), ((0.12, 0.76), (0.88, 0.76), (0.5, 0.96)))
    mesh.add_face((front_right, front_ridge_right, back_right), ((0.12, 0.76), (0.5, 0.96), (0.88, 0.76)))


def _detail_level(height_bucket: str) -> int:
    return {"low": 0, "mid": 1, "high": 2}[height_bucket]


def build_mesh(category: str, width: float, height: float, depth: float, height_bucket: str) -> MeshData:
    """Build one reusable building family member."""
    if category not in VARIANT_STEMS:
        raise ValueError(f"unsupported building category: {category}")
    mesh = MeshData()
    detail = _detail_level(height_bucket)

    if category == "house":
        _box(mesh, width, height, depth, top=False)
        _gabled_roof(mesh, width, height, depth, max(1.2, min(3.0, height * 0.32)))
        if detail >= 1:
            _box(mesh, max(0.7, width * 0.12), height * 0.35, max(0.7, depth * 0.12), x=-width * 0.22, y=height, z=0.0)
        if detail >= 2:
            _box(mesh, width * 0.35, 0.18, depth * 0.15, y=0.15, z=-depth * 0.57)
    elif category == "apartments":
        _box(mesh, width, height, depth)
        levels = max(1, int(round(height / 3.0)))
        balcony_count = min(4, max(1, levels // 2)) if detail else 0
        for index in range(balcony_count):
            y = min(height - 2.0, (index + 1) * height / (balcony_count + 1))
            _box(mesh, width * 0.34, 0.16, 0.9, x=-width * 0.22, y=y, z=-depth * 0.54)
            _box(mesh, width * 0.34, 0.16, 0.9, x=width * 0.22, y=y, z=-depth * 0.54)
    elif category == "commercial":
        _box(mesh, width, height, depth)
        _box(mesh, width * 0.52, 0.22, 1.2, y=min(2.8, height * 0.4), z=-depth * 0.58)
        if detail >= 1:
            _box(mesh, width * 0.85, 0.35, depth * 0.08, y=height, z=0.0)
    elif category == "industrial":
        _box(mesh, width, height, depth, top=False)
        _gabled_roof(mesh, width, height, depth, max(1.5, min(4.0, height * 0.4)))
        if detail >= 1:
            for x in (-width * 0.22, width * 0.22):
                _box(mesh, width * 0.12, 0.08, depth * 0.28, x=x, y=height + 0.03, z=0.0)
    return mesh


def _normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in cross)) or 1.0
    return tuple(value / length for value in cross)


def obj8_text(mesh: MeshData, texture_name: str) -> str:
    """Serialize MeshData to the small, single-texture OBJ8 subset we use."""
    vertices: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float]]] = []
    indices: list[int] = []
    for face, face_uvs in zip(mesh.faces, mesh.face_uvs):
        points = [mesh.vertices[index] for index in face]
        normal = _normal(points[0], points[1], points[2])
        start = len(vertices)
        vertices.extend((point, normal, uv) for point, uv in zip(points, face_uvs))
        for index in range(1, len(face) - 1):
            indices.extend((start, start + index, start + index + 1))

    lines = [
        "A",
        "800",
        "OBJ",
        "",
        f"TEXTURE {texture_name}",
        "GLOBAL_specular 0.35",
        "ATTR_shadow",
        "",
        f"POINT_COUNTS {len(vertices)} 0 0 {len(indices)}",
    ]
    for (x, y, z), (nx, ny, nz), (u, v) in vertices:
        lines.append(f"VT {x:.4f} {y:.4f} {z:.4f} {nx:.5f} {ny:.5f} {nz:.5f} {u:.5f} {v:.5f}")
    lines.extend(f"IDX {index}" for index in indices)
    lines.append(f"TRIS 0 {len(indices)}")
    return "\n".join(lines) + "\n"


def _texture_candidates(texture_root: Path, category: str, target_name: str) -> list[Path]:
    stem = VARIANT_STEMS[category]
    return [
        texture_root / target_name,
        texture_root / f"jp_{stem}.png",
        texture_root / f"{stem}.png",
        texture_root / f"{category}.png",
    ]


def _placeholder_texture(path: Path, category: str) -> None:
    if bpy is None:
        raise RuntimeError("placeholder textures require Blender; provide --texture-root")
    colors = {
        "house": (0.62, 0.42, 0.28, 1.0),
        "apartments": (0.48, 0.55, 0.63, 1.0),
        "commercial": (0.62, 0.62, 0.54, 1.0),
        "industrial": (0.35, 0.42, 0.48, 1.0),
    }
    image = bpy.data.images.new(path.stem, width=256, height=256)
    image.pixels = list(colors[category]) * (256 * 256)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def materialize_texture(texture_root: Path | None, output: Path, category: str, target_name: str, allow_placeholder: bool) -> str:
    target = output / target_name
    if target.is_file():
        return target.name
    if texture_root:
        for candidate in _texture_candidates(texture_root, category, target_name):
            if candidate.is_file():
                shutil.copy2(candidate, target)
                return target.name
    if allow_placeholder:
        _placeholder_texture(target, category)
        return target.name
    raise FileNotFoundError(
        f"missing texture for {category}: expected {target_name} in {texture_root or '(no texture root)'}"
    )


def _asset_name(category: str, size: str, height: str) -> str:
    return f"jp_{VARIANT_STEMS[category]}_{size}_{height}"


def _base_asset_name(category: str) -> str:
    return f"jp_{VARIANT_STEMS[category]}_a"


def _variant_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for category in VARIANT_STEMS:
        width, height, depth = BASE_DIMENSIONS[category]
        records.append({
            "name": _base_asset_name(category),
            "category": category,
            "size": "base",
            "height": "base",
            "width_m": width,
            "height_m": height,
            "depth_m": depth,
        })
        for size in SIZE_BUCKETS:
            variant_width, variant_depth = VARIANT_FOOTPRINTS[size]
            for height_bucket in HEIGHT_BUCKETS:
                records.append({
                    "name": _asset_name(category, size, height_bucket),
                    "category": category,
                    "size": size,
                    "height": height_bucket,
                    "width_m": variant_width,
                    "height_m": VARIANT_HEIGHTS[height_bucket],
                    "depth_m": variant_depth,
                })
    return records


def _blender_mesh_object(name: str, mesh_data: MeshData, texture_path: Path) -> object:
    if bpy is None:
        raise RuntimeError("Blender is required to create a .blend file")
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(mesh_data.vertices, [], mesh_data.faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, face_uvs in zip(mesh.polygons, mesh_data.face_uvs):
        for loop_index, uv in zip(polygon.loop_indices, face_uvs):
            uv_layer.data[loop_index].uv = uv
    material = bpy.data.materials.new(f"{name}_material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF")
    image = bpy.data.images.load(str(texture_path), check_existing=True)
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def generate_pack(output: Path, texture_root: Path | None, allow_placeholder: bool, blend_output: Path | None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records = _variant_records()
    exported: list[dict[str, object]] = []
    blender_objects: list[tuple[str, MeshData, Path]] = []
    for record in records:
        category = str(record["category"])
        name = str(record["name"])
        width = float(record["width_m"])
        height = float(record["height_m"])
        depth = float(record["depth_m"])
        height_bucket = str(record["height"])
        mesh = build_mesh(category, width, height, depth, "mid" if height_bucket == "base" else height_bucket)
        texture_name = f"{name}.png"
        materialize_texture(texture_root, output, category, texture_name, allow_placeholder)
        obj_path = output / f"{name}.obj"
        obj_path.write_text(obj8_text(mesh, texture_name), encoding="utf-8")
        blender_objects.append((name, mesh, output / texture_name))
        exported.append({**record, "obj": obj_path.name, "texture": texture_name, "geometry_source": "blender-procedural"})

    if bpy is not None and blend_output is not None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        for name, mesh, texture_path in blender_objects:
            _blender_mesh_object(name, mesh, texture_path)
        blend_output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_output))

    manifest = {
        "schema_version": 1,
        "generator": "tools/blender_generate_assets.py",
        "geometry_source": "blender-procedural",
        "texture_policy": "ComfyUI facade texture with front, side, back, and roof UV regions",
        "assets": exported,
    }
    (output / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _script_argv(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--texture-root", type=Path)
    parser.add_argument("--blend-output", type=Path)
    parser.add_argument("--allow-placeholder", action="store_true")
    parser.add_argument(
        "--obj8-only",
        action="store_true",
        help="allow running with normal Python when texture files are supplied; no .blend is written",
    )
    args = parser.parse_args(_script_argv(argv))
    if bpy is None and args.allow_placeholder:
        parser.error("--allow-placeholder requires Blender; provide --texture-root for --obj8-only")
    if bpy is None and not args.obj8_only:
        parser.error("run this script with Blender, or pass --obj8-only for the OBJ8-only path")
    if args.blend_output and bpy is None:
        parser.error("--blend-output requires Blender")
    if not args.texture_root and not args.allow_placeholder:
        parser.error("--texture-root is required unless --allow-placeholder is set")
    manifest = generate_pack(args.output, args.texture_root, args.allow_placeholder, args.blend_output)
    print(f"generated {len(manifest['assets'])} Blender/OBJ8 assets in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
