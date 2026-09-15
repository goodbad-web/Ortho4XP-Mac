#!/usr/bin/env python3
"""Create a lightweight procedural X-Plane overlay for The Scene Johoku.

The model is an intentionally small landmark approximation of the Nagoya
residential complex.  It contains an elliptical 160 m Astro Tower and two
low-rise Star buildings.  Windows are represented by a facade atlas and a
small number of continuous bands; no interiors or per-window meshes are
generated.

Examples::

    .venv/bin/python tools/create_scene_johoku.py \
        --output /tmp/zz_TheSceneJohoku_+35+136 \
        --config tools/scene_johoku_config.json

    tools/run_blender_with_metal_preflight.sh --background \
        --python tools/create_scene_johoku.py -- \
        --output /tmp/zz_TheSceneJohoku_+35+136 \
        --config tools/scene_johoku_config.json \
        --blend-output /tmp/zz_TheSceneJohoku_+35+136/the-scene-johoku.blend

Coordinates and dimensions are deliberately configurable.  They are an
initial landmark placement, not a survey or a replacement for checking the
footprint in X-Plane.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:  # Blender is present only when the script is run by Blender.
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Blender-only branch.
    bpy = None


TEXTURE_NAME = "scene_johoku_facade"
SEGMENTS_NEAR = 24
SEGMENTS_MID = 12
SEGMENTS_FAR = 8


@dataclass
class MeshData:
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    face_uvs: list[tuple[tuple[float, float], ...]] = field(default_factory=list)

    def add_face(
        self,
        points: Iterable[tuple[float, float, float]],
        uvs: Iterable[tuple[float, float]],
    ) -> None:
        points = list(points)
        uvs = list(uvs)
        if len(points) < 3 or len(points) != len(uvs):
            raise ValueError("face requires matching points and UVs")
        start = len(self.vertices)
        self.vertices.extend(points)
        self.faces.append(tuple(range(start, start + len(points))))
        self.face_uvs.append(tuple(uvs))


def add_box(
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
    """Add a closed, axis-aligned box using a compact facade UV layout."""
    hx, hz = width / 2.0, depth / 2.0
    corners = {
        "bl": (x - hx, y, z - hz),
        "br": (x + hx, y, z - hz),
        "fr": (x + hx, y, z + hz),
        "fl": (x - hx, y, z + hz),
        "tl": (x - hx, y + height, z - hz),
        "tr": (x + hx, y + height, z - hz),
        "ur": (x + hx, y + height, z + hz),
        "ul": (x - hx, y + height, z + hz),
    }
    side_uv = ((0.02, 0.04), (0.02, 0.68), (0.20, 0.68), (0.20, 0.04))
    front_uv = ((0.02, 0.04), (0.02, 0.68), (0.98, 0.68), (0.98, 0.04))
    back_uv = ((0.78, 0.04), (0.78, 0.68), (0.98, 0.68), (0.98, 0.04))
    roof_uv = ((0.04, 0.72), (0.04, 0.98), (0.96, 0.98), (0.96, 0.72))
    bottom_uv = ((0.0, 0.0), (0.0, 0.04), (1.0, 0.04), (1.0, 0.0))
    mesh.add_face(
        (corners["bl"], corners["br"], corners["fr"], corners["fl"]), bottom_uv
    )
    mesh.add_face(
        (corners["bl"], corners["fl"], corners["ul"], corners["tl"]), side_uv
    )
    mesh.add_face(
        (corners["br"], corners["tr"], corners["ur"], corners["fr"]), side_uv
    )
    mesh.add_face(
        (corners["fr"], corners["ur"], corners["ul"], corners["fl"]), front_uv
    )
    mesh.add_face(
        (corners["bl"], corners["tl"], corners["tr"], corners["br"]), back_uv
    )
    if top:
        mesh.add_face(
            (corners["ul"], corners["ur"], corners["tr"], corners["tl"]), roof_uv
        )


def _ellipse_point(
    rx: float, rz: float, y: float, angle: float, *, x: float = 0.0, z: float = 0.0
) -> tuple[float, float, float]:
    return (x + rx * math.cos(angle), y, z + rz * math.sin(angle))


def add_ellipse_shell(
    mesh: MeshData,
    rx: float,
    rz: float,
    y0: float,
    y1: float,
    segments: int,
    *,
    x: float = 0.0,
    z: float = 0.0,
) -> None:
    """Add an elliptical shell and caps without generating interior detail."""
    for index in range(segments):
        a0 = 2.0 * math.pi * index / segments
        a1 = 2.0 * math.pi * (index + 1) / segments
        p0 = _ellipse_point(rx, rz, y0, a0, x=x, z=z)
        p1 = _ellipse_point(rx, rz, y0, a1, x=x, z=z)
        p2 = _ellipse_point(rx, rz, y1, a1, x=x, z=z)
        p3 = _ellipse_point(rx, rz, y1, a0, x=x, z=z)
        u0 = index / segments
        u1 = (index + 1) / segments
        # Reverse the ring winding so normals point out from the tower.  X-Plane
        # culls back-facing OBJ8 polygons by default.
        mesh.add_face((p0, p3, p2, p1), ((u0, 0.04), (u0, 0.68), (u1, 0.68), (u1, 0.04)))

    bottom = [_ellipse_point(rx, rz, y0, 2.0 * math.pi * i / segments, x=x, z=z) for i in range(segments)]
    top = [_ellipse_point(rx, rz, y1, 2.0 * math.pi * i / segments, x=x, z=z) for i in range(segments)]
    # The increasing-angle ring points downwards in OBJ8's right-handed
    # coordinate system.  Reverse only the top cap so both caps face out.
    mesh.add_face(tuple(bottom), tuple((0.0, 0.0) for _ in bottom))
    mesh.add_face(tuple(reversed(top)), tuple((0.0, 0.72) for _ in top))


def add_ellipse_band(
    mesh: MeshData,
    rx: float,
    rz: float,
    y: float,
    thickness: float,
    segments: int,
) -> None:
    """Add a shallow continuous balcony/floor band around an elliptical body."""
    outer0 = [_ellipse_point(rx + 0.7, rz + 0.7, y, 2.0 * math.pi * i / segments) for i in range(segments)]
    outer1 = [_ellipse_point(rx + 0.7, rz + 0.7, y + thickness, 2.0 * math.pi * i / segments) for i in range(segments)]
    for index in range(segments):
        next_index = (index + 1) % segments
        u0, u1 = index / segments, (index + 1) / segments
        mesh.add_face(
            (outer0[index], outer1[index], outer1[next_index], outer0[next_index]),
            ((u0, 0.70), (u0, 0.73), (u1, 0.73), (u1, 0.70)),
        )


def build_astro_tower_mesh(detail: str = "near") -> MeshData:
    """Build the central elliptical tower at one of three detail budgets."""
    if detail == "near":
        segments, band_step = SEGMENTS_NEAR, 1
    elif detail == "mid":
        segments, band_step = SEGMENTS_MID, 3
    elif detail == "far":
        segments, band_step = SEGMENTS_FAR, 99
    else:
        raise ValueError(f"unsupported detail: {detail}")

    mesh = MeshData()
    base_height = 7.0
    body_top = 153.0
    # Keep a proven box-core silhouette inside the elliptical shell.  This
    # makes the landmark robust across X-Plane render paths while the shell
    # and bands provide the rounded facade seen from near range.
    add_box(mesh, 52.0, body_top - base_height, 34.0, y=base_height)
    add_box(mesh, 62.0, base_height, 48.0, y=0.0)
    add_ellipse_shell(mesh, 29.0, 21.0, 0.0, body_top, segments)
    if detail != "far":
        for floor in range(1, 46):
            y = base_height + (body_top - base_height) * floor / 45.0
            if floor % band_step == 0:
                add_ellipse_band(mesh, 29.0, 21.0, y - 0.10, 0.20, segments)
        if detail == "near":
            for x in (-21.0, -10.5, 0.0, 10.5, 21.0):
                add_box(mesh, 0.72, body_top - base_height, 0.48, x=x, y=base_height, z=21.1)
                add_box(mesh, 0.72, body_top - base_height, 0.48, x=x, y=base_height, z=-21.1)

    # The top silhouette is kept simple but recognizable at distance.
    add_ellipse_shell(mesh, 30.0, 22.0, body_top, 157.0, segments)
    add_ellipse_shell(mesh, 18.0, 12.0, 157.0, 159.0, max(6, segments // 2))
    add_box(mesh, 3.0, 1.0, 3.0, y=159.0)
    return mesh


def build_star_mesh(detail: str = "near", *, width: float = 58.0, depth: float = 30.0) -> MeshData:
    """Build a low-rise Star building with a rectangular, low-poly profile."""
    if detail not in {"near", "mid", "far"}:
        raise ValueError(f"unsupported detail: {detail}")
    mesh = MeshData()
    height = 31.0
    add_box(mesh, width, height, depth)
    if detail != "far":
        step = 1 if detail == "near" else 3
        for floor in range(1, 12):
            y = height * floor / 11.0
            if floor % step == 0:
                add_box(mesh, width + 0.8, 0.14, depth + 0.8, y=y - 0.07)
        # A shallow central entrance volume is cheaper and more legible than
        # dozens of small balcony meshes.
        add_box(mesh, width * 0.24, 4.0, depth * 0.30, y=height, z=depth * 0.18)
    return mesh


def build_podium_mesh(detail: str = "near") -> MeshData:
    """Build the shared low podium used by the tower and Star buildings."""
    if detail == "near":
        segments = SEGMENTS_NEAR
    elif detail == "mid":
        segments = SEGMENTS_MID
    elif detail == "far":
        segments = SEGMENTS_FAR
    else:
        raise ValueError(f"unsupported detail: {detail}")

    mesh = MeshData()
    add_ellipse_shell(mesh, 50.0, 30.0, 0.0, 7.0, segments)
    if detail != "far":
        add_ellipse_band(mesh, 50.0, 30.0, 7.0, 0.35, segments)
        add_box(mesh, 38.0, 3.0, 18.0, y=7.0, z=20.0)
    if detail == "near":
        add_box(mesh, 12.0, 2.5, 28.0, y=10.0, z=18.0)
        add_box(mesh, 46.0, 1.2, 8.0, y=10.0, z=-18.0)
    return mesh


def _normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in cross)) or 1.0
    return tuple(value / length for value in cross)


@dataclass(frozen=True)
class LODSection:
    near_m: float
    far_m: float
    mesh: MeshData


def obj8_lod_text(sections: Iterable[LODSection], texture_name: str) -> str:
    """Serialize several LOD meshes into one OBJ8 object."""
    all_vertices: list[
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float]]
    ] = []
    all_indices: list[int] = []
    commands: list[tuple[float, float, int, int]] = []
    for section in sections:
        vertices = []
        indices = []
        for face, face_uvs in zip(section.mesh.faces, section.mesh.face_uvs):
            points = [section.mesh.vertices[index] for index in face]
            normal = _normal(points[0], points[1], points[2])
            start = len(vertices)
            vertices.extend((point, normal, uv) for point, uv in zip(points, face_uvs))
            for index in range(1, len(face) - 1):
                indices.extend((start, start + index, start + index + 1))
        vertex_offset = len(all_vertices)
        index_offset = len(all_indices)
        all_vertices.extend(vertices)
        all_indices.extend(vertex_offset + index for index in indices)
        commands.append((section.near_m, section.far_m, index_offset, len(indices)))

    lines = [
        "A", "800", "OBJ", "", f"TEXTURE {texture_name}",
        "GLOBAL_specular 0.20", "ATTR_shadow", "",
        f"POINT_COUNTS {len(all_vertices)} 0 0 {len(all_indices)}",
    ]
    for (x, y, z), (nx, ny, nz), (u, v) in all_vertices:
        lines.append(f"VT {x:.4f} {y:.4f} {z:.4f} {nx:.5f} {ny:.5f} {nz:.5f} {u:.5f} {v:.5f}")
    lines.extend(f"IDX {index}" for index in all_indices)
    for near_m, far_m, index_offset, index_count in commands:
        lines.append(f"ATTR_LOD {near_m:.1f} {far_m:.1f}")
        lines.append(f"TRIS {index_offset} {index_count}")
    return "\n".join(lines) + "\n"


def _draw_facade_texture(path: Path) -> None:
    """Write a generic self-owned blue-gray residential facade atlas."""
    if path.is_file():
        return
    if bpy is not None:
        image = bpy.data.images.new(path.stem, width=1024, height=1024)
        image.generated_color = (0.34, 0.43, 0.49, 1.0)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        return

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1024, 1024), (184, 190, 188))
    draw = ImageDraw.Draw(image)
    for y in range(42, 670, 28):
        draw.rectangle((0, y - 2, 1023, y + 2), fill=(235, 235, 228))
        for x in range(18, 1024, 38):
            draw.rectangle((x, y + 4, x + 26, y + 20), fill=(55, 73, 79))
            draw.line((x + 2, y + 6, x + 24, y + 6), fill=(180, 198, 199), width=2)
            draw.line((x + 13, y + 5, x + 13, y + 19), fill=(100, 120, 122), width=1)
    for x in (90, 296, 512, 728, 934):
        draw.rectangle((x, 20, x + 8, 700), fill=(220, 222, 218))
    draw.rectangle((0, 705, 1023, 725), fill=(102, 106, 103))
    draw.rectangle((0, 730, 1023, 1023), fill=(164, 164, 156))
    for x in range(20, 1024, 66):
        draw.rectangle((x, 768, x + 44, 890), fill=(63, 76, 81))
        draw.rectangle((x + 5, 774, x + 39, 828), fill=(191, 207, 205))
    for y in range(746, 1000, 58):
        draw.line((0, y, 1023, y), fill=(213, 206, 187), width=3)
    draw.rectangle((0, 970, 1023, 1023), fill=(94, 101, 96))
    image.save(path)


def _compress_texture(nvcompress: Path, png_path: Path, dds_path: Path) -> None:
    result = subprocess.run(
        [str(nvcompress), "-bc1", "-fast", "-silent", str(png_path), str(dds_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcompress failed: {result.stdout}\n{result.stderr}")


def _create_blend(
    path: Path,
    objects: list[tuple[str, MeshData, tuple[float, float, float]]],
    texture_path: Path,
) -> None:
    if bpy is None:
        raise RuntimeError("--blend-output requires Blender")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    image = bpy.data.images.load(str(texture_path), check_existing=True)
    for name, mesh_data, location in objects:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(mesh_data.vertices, [], mesh_data.faces)
        mesh.update()
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for polygon, face_uvs in zip(mesh.polygons, mesh_data.face_uvs):
            for loop_index, uv in zip(polygon.loop_indices, face_uvs):
                uv_layer.data[loop_index].uv = uv
        material = bpy.data.materials.new(f"{name}_material")
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
        mesh.materials.append(material)
        obj = bpy.data.objects.new(name, mesh)
        obj.location = location
        bpy.context.collection.objects.link(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def _load_config(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported The Scene Johoku config schema")
    for key in ("tile", "astro", "west_star", "east_star"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(f"missing or invalid config section: {key}")
    return data


def _coordinate(config: dict[str, object], name: str) -> tuple[float, float, float]:
    record = config[name]
    if not isinstance(record, dict):
        raise ValueError(f"invalid config section: {name}")
    return float(record["lat"]), float(record["lon"]), float(record.get("heading", 0.0))


def _write_dsf(path: Path, config: dict[str, object]) -> None:
    tile = config["tile"]
    astro_lat, astro_lon, astro_heading = _coordinate(config, "astro")
    west_lat, west_lon, west_heading = _coordinate(config, "west_star")
    east_lat, east_lon, east_heading = _coordinate(config, "east_star")
    lines = [
        "PROPERTY sim/planet earth", "PROPERTY sim/overlay 1",
        f"PROPERTY sim/west {int(tile['lon'])}", f"PROPERTY sim/east {int(tile['lon']) + 1}",
        f"PROPERTY sim/south {int(tile['lat'])}", f"PROPERTY sim/north {int(tile['lat']) + 1}",
        "OBJECT_DEF objects/scene_johoku_astro_tower.obj",
        "OBJECT_DEF objects/scene_johoku_west_star.obj",
        "OBJECT_DEF objects/scene_johoku_east_star.obj",
        "OBJECT_DEF objects/scene_johoku_podium.obj",
        f"OBJECT 0 {astro_lon:.7f} {astro_lat:.7f} {astro_heading:.2f}",
        f"OBJECT 1 {west_lon:.7f} {west_lat:.7f} {west_heading:.2f}",
        f"OBJECT 2 {east_lon:.7f} {east_lat:.7f} {east_heading:.2f}",
        f"OBJECT 3 {astro_lon:.7f} {astro_lat:.7f} {astro_heading:.2f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_package(
    output: Path,
    config_path: Path,
    dsftool: Path | None = None,
    nvcompress: Path | None = None,
    blend_output: Path | None = None,
) -> dict[str, object]:
    config = _load_config(config_path)
    tile = config["tile"]
    tile_lat, tile_lon = int(tile["lat"]), int(tile["lon"])
    output.mkdir(parents=True, exist_ok=True)
    objects_dir = output / "objects"
    earth_dir = output / "Earth nav data" / f"{tile_lat // 10 * 10:+03d}{tile_lon // 10 * 10:+04d}"
    objects_dir.mkdir(exist_ok=True)
    earth_dir.mkdir(parents=True, exist_ok=True)

    png_path = objects_dir / f"{TEXTURE_NAME}.png"
    _draw_facade_texture(png_path)
    texture_name = png_path.name
    if nvcompress:
        dds_path = objects_dir / f"{TEXTURE_NAME}.dds"
        _compress_texture(nvcompress, png_path, dds_path)
        texture_name = dds_path.name

    mesh_specs = {
        "scene_johoku_astro_tower": [
            LODSection(0.0, 600.0, build_astro_tower_mesh("near")),
            LODSection(600.0, 3000.0, build_astro_tower_mesh("mid")),
            LODSection(3000.0, 18000.0, build_astro_tower_mesh("far")),
        ],
        "scene_johoku_west_star": [
            LODSection(0.0, 500.0, build_star_mesh("near")),
            LODSection(500.0, 2500.0, build_star_mesh("mid")),
            LODSection(2500.0, 18000.0, build_star_mesh("far")),
        ],
        "scene_johoku_east_star": [
            LODSection(0.0, 500.0, build_star_mesh("near")),
            LODSection(500.0, 2500.0, build_star_mesh("mid")),
            LODSection(2500.0, 18000.0, build_star_mesh("far")),
        ],
        "scene_johoku_podium": [
            LODSection(0.0, 1000.0, build_podium_mesh("near")),
            LODSection(1000.0, 4000.0, build_podium_mesh("mid")),
            LODSection(4000.0, 18000.0, build_podium_mesh("far")),
        ],
    }
    for name, sections in mesh_specs.items():
        (objects_dir / f"{name}.obj").write_text(
            obj8_lod_text(sections, texture_name), encoding="utf-8"
        )

    text_dsf = earth_dir / f"{tile_lat:+03d}{tile_lon:+04d}.txt"
    _write_dsf(text_dsf, config)
    if dsftool:
        binary_dsf = text_dsf.with_suffix(".dsf")
        result = subprocess.run(
            [str(dsftool), "-text2dsf", str(text_dsf), str(binary_dsf)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"DSFTool failed: {result.stdout}\n{result.stderr}")
        text_dsf.unlink()

    if blend_output:
        _create_blend(
            blend_output,
            [
                ("THE_SCENE_JOHOKU_ASTRO_TOWER", build_astro_tower_mesh("near"), (0.0, 0.0, 0.0)),
                ("THE_SCENE_JOHOKU_WEST_STAR", build_star_mesh("near"), (-70.0, 0.0, 0.0)),
                ("THE_SCENE_JOHOKU_EAST_STAR", build_star_mesh("near"), (70.0, 0.0, 0.0)),
                ("THE_SCENE_JOHOKU_PODIUM", build_podium_mesh("near"), (0.0, 0.0, 0.0)),
            ],
            png_path,
        )

    report = {
        "schema_version": 1,
        "generator": "tools/create_scene_johoku.py",
        "name": "The Scene Johoku",
        "tile": {"lat": tile_lat, "lon": tile_lon},
        "objects": [f"objects/{name}.obj" for name in mesh_specs],
        "texture": f"objects/{texture_name}",
        "texture_source": "self-owned procedural facade atlas; no logos or copied imagery",
        "landmark_spec": {
            "astro_tower_height_m": 160.0,
            "astro_tower_floors": 45,
            "star_building_height_m": 31.0,
            "star_buildings": 2,
        },
        "lod_ranges_m": [[0, 500], [500, 2500], [2500, 18000]],
        "coordinate_note": "Initial coordinate from the user-provided Google Earth reference; verify footprint and heading in X-Plane",
        "verification_required": [
            "verify Astro Tower and Star building footprint alignment in X-Plane",
            "compare against default scenery before adding any exclusion rectangle",
            "measure FPS at the same camera position before and after the overlay",
        ],
    }
    (output / "generation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _script_argv(argv: list[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" in values:
        values = values[values.index("--") + 1 :]
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("tools/scene_johoku_config.json"))
    parser.add_argument("--dsftool", type=Path)
    parser.add_argument("--nvcompress", type=Path)
    parser.add_argument("--blend-output", type=Path)
    args = parser.parse_args(_script_argv(argv))
    if args.blend_output and bpy is None:
        parser.error("--blend-output requires Blender")
    report = generate_package(args.output, args.config, args.dsftool, args.nvcompress, args.blend_output)
    print(f"generated The Scene Johoku package with {len(report['objects'])} objects in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
