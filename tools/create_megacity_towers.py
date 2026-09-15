#!/usr/bin/env python3
"""Create a lightweight procedural MEGA CITY TOWERS X-Plane overlay.

The asset is intentionally landmark-oriented: the two towers share a small
procedural family, use OBJ8 LOD sections, and have no interior geometry.  The
facade PNG is self-owned and can be replaced by a ComfyUI-generated texture
without changing the geometry or DSF placement.

Examples::

    .venv/bin/python tools/create_megacity_towers.py \
        --output /tmp/zz_MegaCityTowers_+34+135 \
        --config tools/megacity_towers_config.json \
        --dsftool Utils/mac/DSFTool \
        --nvcompress Utils/mac/nvcompress

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python tools/create_megacity_towers.py -- \
        --output /tmp/zz_MegaCityTowers_+34+135 \
        --config tools/megacity_towers_config.json \
        --blend-output /tmp/zz_MegaCityTowers_+34+135/megacity-towers.blend

The default west coordinate is explicitly provisional.  Update the JSON
configuration after checking the building footprint in a map or simulator.
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

try:  # Blender is available only when invoked by Blender.
    import bpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Blender-only branch.
    bpy = None


TOWER_HEIGHT_M = 143.8
PODIUM_HEIGHT_M = 12.0
TOWER_WIDTH_M = 48.0
TOWER_DEPTH_M = 34.0
BODY_HEIGHT_M = 124.8
CROWN_HEIGHT_M = 7.0
TEXTURE_NAME = "mega_city_towers_facade"

FRONT_UV = ((0.02, 0.04), (0.02, 0.68), (0.98, 0.68), (0.98, 0.04))
SIDE_UV = ((0.02, 0.04), (0.02, 0.68), (0.20, 0.68), (0.20, 0.04))
BACK_UV = ((0.78, 0.04), (0.78, 0.68), (0.98, 0.68), (0.98, 0.04))
ROOF_UV = ((0.04, 0.72), (0.04, 0.98), (0.96, 0.98), (0.96, 0.72))
BOTTOM_UV = ((0.0, 0.0), (0.0, 0.04), (1.0, 0.04), (1.0, 0.0))


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
    """Add a closed box with +Y up and a facade-oriented UV layout."""
    hx, hz = width / 2.0, depth / 2.0
    bottom, upper = y, y + height
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
    quad = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    mesh.add_face((corners["bl"], corners["br"], corners["fr"], corners["fl"]), BOTTOM_UV)
    mesh.add_face((corners["bl"], corners["fl"], corners["ul"], corners["tl"]), SIDE_UV)
    mesh.add_face((corners["br"], corners["tr"], corners["ur"], corners["fr"]), SIDE_UV)
    mesh.add_face((corners["fr"], corners["ur"], corners["ul"], corners["fl"]), FRONT_UV)
    mesh.add_face((corners["bl"], corners["tl"], corners["tr"], corners["br"]), BACK_UV)
    if top:
        mesh.add_face((corners["ul"], corners["ur"], corners["tr"], corners["tl"]), ROOF_UV)


def _tower_dimensions(kind: str) -> tuple[float, float, int]:
    if kind == "east":
        return TOWER_WIDTH_M, TOWER_DEPTH_M, 41
    if kind == "west":
        return TOWER_WIDTH_M, TOWER_DEPTH_M, 40
    raise ValueError(f"unsupported tower kind: {kind}")


def build_tower_mesh(kind: str, detail: str = "near") -> MeshData:
    """Build a low-poly tower silhouette with three detail budgets."""
    width, depth, floors = _tower_dimensions(kind)
    mesh = MeshData()

    if detail == "far":
        add_box(mesh, width, TOWER_HEIGHT_M, depth)
        return mesh

    add_box(mesh, width, BODY_HEIGHT_M, depth, y=PODIUM_HEIGHT_M)
    add_box(mesh, width + 2.0, 1.0, depth + 2.0, y=PODIUM_HEIGHT_M - 0.5)

    if detail == "near":
        band_step = 1
        balcony_step = 1
    elif detail == "mid":
        band_step = 3
        balcony_step = 3
    else:
        raise ValueError(f"unsupported tower detail: {detail}")

    floor_height = BODY_HEIGHT_M / floors
    for floor in range(1, floors + 1):
        y = PODIUM_HEIGHT_M + floor * floor_height
        if floor % band_step == 0:
            add_box(mesh, width + 1.1, 0.14, depth + 1.1, y=y - 0.08)
        if floor % balcony_step == 0:
            balcony_width = width * 0.86
            add_box(mesh, balcony_width, 0.12, 1.15, y=y - 0.16, z=depth / 2.0 + 0.55)
            add_box(mesh, balcony_width, 0.12, 1.15, y=y - 0.16, z=-depth / 2.0 - 0.55)

    if detail == "near":
        # Vertical lines are a defining feature of the real facade and are
        # cheaper than modeling individual windows.
        for x in (-20.0, -10.0, 0.0, 10.0, 20.0):
            for z in (-depth / 2.0 - 0.28, depth / 2.0 + 0.28):
                add_box(mesh, 0.85, BODY_HEIGHT_M, 0.55, x=x, y=PODIUM_HEIGHT_M, z=z)
        for z in (-14.0, 14.0):
            add_box(mesh, 0.55, BODY_HEIGHT_M, 0.85, y=PODIUM_HEIGHT_M, z=z)

    crown_y = PODIUM_HEIGHT_M + BODY_HEIGHT_M
    add_box(mesh, width + 2.0, CROWN_HEIGHT_M, depth + 2.0, y=crown_y)
    add_box(mesh, width * 0.42, 2.4, depth * 0.34, y=TOWER_HEIGHT_M - 2.4, z=-2.0 if kind == "east" else 2.0)
    if detail == "near":
        add_box(mesh, 4.0, 1.8, 7.0, x=-10.0, y=TOWER_HEIGHT_M - 1.8, z=0.0)
        add_box(mesh, 4.0, 1.8, 7.0, x=10.0, y=TOWER_HEIGHT_M - 1.8, z=0.0)
    return mesh


def build_podium_mesh(detail: str = "near") -> MeshData:
    mesh = MeshData()
    if detail == "far":
        add_box(mesh, 150.0, PODIUM_HEIGHT_M, 70.0)
        return mesh
    add_box(mesh, 150.0, PODIUM_HEIGHT_M, 70.0)
    add_box(mesh, 128.0, 2.0, 52.0, y=PODIUM_HEIGHT_M)
    if detail == "near":
        add_box(mesh, 30.0, 5.0, 18.0, y=PODIUM_HEIGHT_M + 2.0)
        add_box(mesh, 8.0, 3.5, 60.0, y=PODIUM_HEIGHT_M + 2.0)
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


@dataclass(frozen=True)
class LODSection:
    near_m: float
    far_m: float
    mesh: MeshData


def obj8_lod_text(sections: Iterable[LODSection], texture_name: str) -> str:
    """Serialize OBJ8 vertices once and draw each LOD range separately."""
    all_vertices: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float]]] = []
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
        "GLOBAL_specular 0.25", "ATTR_shadow", "",
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
    """Write a self-owned facade atlas; ComfyUI can replace this file later."""
    if path.is_file():
        return
    if bpy is not None:
        image = bpy.data.images.new(path.stem, width=1024, height=1024)
        image.generated_color = (0.34, 0.42, 0.48, 1.0)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        return

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1024, 1024), (112, 126, 136))
    draw = ImageDraw.Draw(image)
    # Residential facade atlas: window rhythm, concrete mullions, and glass
    # balcony bands.  It is deliberately generic and contains no logos.
    for y in range(48, 670, 28):
        draw.rectangle((0, y - 2, 1023, y + 2), fill=(205, 211, 211))
        for x in range(18, 1024, 38):
            draw.rectangle((x, y + 4, x + 26, y + 20), fill=(44, 75, 92))
            draw.line((x + 2, y + 6, x + 24, y + 6), fill=(157, 190, 201), width=2)
            draw.line((x + 13, y + 5, x + 13, y + 19), fill=(97, 128, 139), width=1)
    for x in (90, 296, 512, 728, 934):
        draw.rectangle((x, 20, x + 8, 700), fill=(220, 222, 218))
    draw.rectangle((0, 705, 1023, 725), fill=(78, 85, 88))
    draw.rectangle((0, 730, 1023, 1023), fill=(163, 158, 143))
    for x in range(20, 1024, 66):
        draw.rectangle((x, 768, x + 44, 890), fill=(63, 76, 81))
        draw.rectangle((x + 5, 774, x + 39, 828), fill=(191, 207, 205))
    for y in range(746, 1000, 58):
        draw.line((0, y, 1023, y), fill=(213, 206, 187), width=3)
    draw.rectangle((0, 970, 1023, 1023), fill=(94, 101, 96))
    image.save(path)


def _compress_texture(nvcompress: Path, png_path: Path, dds_path: Path) -> None:
    result = subprocess.run(
        [str(nvcompress), "-bc3", "-fast", "-silent", str(png_path), str(dds_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcompress failed: {result.stdout}\n{result.stderr}")


def _create_blend(path: Path, objects: list[tuple[str, MeshData, tuple[float, float, float]]], texture_path: Path) -> None:
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
        raise ValueError("unsupported MEGA CITY TOWERS config schema")
    for key in ("tile", "east", "west", "podium"):
        if key not in data:
            raise ValueError(f"missing config section: {key}")
    return data


def _coordinate(config: dict[str, object], name: str) -> tuple[float, float, float]:
    record = config[name]
    if not isinstance(record, dict):
        raise ValueError(f"invalid config section: {name}")
    return float(record["lat"]), float(record["lon"]), float(record.get("heading", 0.0))


def _write_dsf(path: Path, config: dict[str, object]) -> None:
    tile = config["tile"]
    east_lat, east_lon, east_heading = _coordinate(config, "east")
    west_lat, west_lon, west_heading = _coordinate(config, "west")
    podium_lat = (east_lat + west_lat) / 2.0
    podium_lon = (east_lon + west_lon) / 2.0
    lines = [
        "PROPERTY sim/planet earth",
        "PROPERTY sim/overlay 1",
        f"PROPERTY sim/west {int(tile['lon'])}",
        f"PROPERTY sim/east {int(tile['lon']) + 1}",
        f"PROPERTY sim/south {int(tile['lat'])}",
        f"PROPERTY sim/north {int(tile['lat']) + 1}",
        "OBJECT_DEF objects/mega_city_towers_podium.obj",
        "OBJECT_DEF objects/mega_city_towers_west.obj",
        "OBJECT_DEF objects/mega_city_towers_east.obj",
        f"OBJECT 0 {podium_lon:.7f} {podium_lat:.7f} 0.00",
        f"OBJECT 1 {west_lon:.7f} {west_lat:.7f} {west_heading:.2f}",
        f"OBJECT 2 {east_lon:.7f} {east_lat:.7f} {east_heading:.2f}",
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
    dds_path = objects_dir / f"{TEXTURE_NAME}.dds"
    if nvcompress:
        _compress_texture(nvcompress, png_path, dds_path)
        texture_name = dds_path.name

    mesh_specs = {
        "mega_city_towers_west": [
            LODSection(0.0, 500.0, build_tower_mesh("west", "near")),
            LODSection(500.0, 2500.0, build_tower_mesh("west", "mid")),
            LODSection(2500.0, 15000.0, build_tower_mesh("west", "far")),
        ],
        "mega_city_towers_east": [
            LODSection(0.0, 500.0, build_tower_mesh("east", "near")),
            LODSection(500.0, 2500.0, build_tower_mesh("east", "mid")),
            LODSection(2500.0, 15000.0, build_tower_mesh("east", "far")),
        ],
        "mega_city_towers_podium": [
            LODSection(0.0, 1000.0, build_podium_mesh("near")),
            LODSection(1000.0, 4000.0, build_podium_mesh("mid")),
            LODSection(4000.0, 15000.0, build_podium_mesh("far")),
        ],
    }
    for name, sections in mesh_specs.items():
        (objects_dir / f"{name}.obj").write_text(
            obj8_lod_text(sections, texture_name), encoding="utf-8"
        )

    text_dsf = earth_dir / f"{tile_lat:+03d}{tile_lon:+04d}.txt"
    _write_dsf(text_dsf, config)
    binary_dsf = text_dsf.with_suffix(".dsf")
    if dsftool:
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
        east_lat, east_lon, _ = _coordinate(config, "east")
        west_lat, west_lon, _ = _coordinate(config, "west")
        meters_per_lon = 111320.0 * math.cos(math.radians((east_lat + west_lat) / 2.0))
        east_x = (east_lon - west_lon) * meters_per_lon / 2.0
        blend_objects = [
            ("MEGA_CITY_TOWERS_WEST", build_tower_mesh("west", "near"), (-east_x, 0.0, 0.0)),
            ("MEGA_CITY_TOWERS_EAST", build_tower_mesh("east", "near"), (east_x, 0.0, 0.0)),
            ("MEGA_CITY_TOWERS_PODIUM", build_podium_mesh("near"), (0.0, 0.0, 0.0)),
        ]
        _create_blend(blend_output, blend_objects, png_path)

    report = {
        "schema_version": 1,
        "generator": "tools/create_megacity_towers.py",
        "tile": {"lat": tile_lat, "lon": tile_lon},
        "objects": [
            "objects/mega_city_towers_west.obj",
            "objects/mega_city_towers_east.obj",
            "objects/mega_city_towers_podium.obj",
        ],
        "texture": f"objects/{texture_name}",
        "texture_source": "self-owned procedural facade atlas; replaceable by ComfyUI output",
        "lod_ranges_m": [[0, 500], [500, 2500], [2500, 15000]],
        "west_coordinate_note": config["west"].get("coordinate_source"),
        "east_coordinate_note": config["east"].get("coordinate_source"),
        "verification_required": [
            "verify west/east footprint alignment in X-Plane",
            "compare against default scenery before adding a tight exclusion",
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
    parser.add_argument("--config", type=Path, default=Path("tools/megacity_towers_config.json"))
    parser.add_argument("--dsftool", type=Path)
    parser.add_argument("--nvcompress", type=Path)
    parser.add_argument("--blend-output", type=Path)
    args = parser.parse_args(_script_argv(argv))
    if args.blend_output and bpy is None:
        parser.error("--blend-output requires Blender")
    report = generate_package(args.output, args.config, args.dsftool, args.nvcompress, args.blend_output)
    print(f"generated MEGA CITY TOWERS package with {len(report['objects'])} objects in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
