#!/usr/bin/env python3
"""Create self-owned low-poly OBJ/PNG assets for the first scenery PoC."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


ASSETS = {
    "jp_house_a.obj": ((9.0, 6.0, 6.0), (184, 150, 115)),
    "jp_apartment_a.obj": ((18.0, 10.0, 18.0), (150, 158, 170)),
    "jp_commercial_a.obj": ((22.0, 7.0, 16.0), (178, 178, 160)),
    "jp_industrial_a.obj": ((30.0, 8.0, 24.0), (125, 135, 142)),
}


def _texture(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(image)
    for x in range(16, 256, 32):
        for y in range(24, 256, 42):
            draw.rectangle((x, y, x + 12, y + 18), fill=(35, 50, 58))
    draw.line((0, 220, 256, 220), fill=(70, 70, 65), width=5)
    image.save(path)


def _obj(path: Path, dimensions: tuple[float, float, float], texture_name: str) -> None:
    width, height, depth = dimensions
    x, z = width / 2, depth / 2
    faces = [
        ((0, -1, 0), [(-x, 0, -z), (x, 0, -z), (x, 0, z), (-x, 0, z)]),
        ((0, 1, 0), [(-x, height, z), (x, height, z), (x, height, -z), (-x, height, -z)]),
        ((0, 0, -1), [(-x, 0, -z), (-x, height, -z), (x, height, -z), (x, 0, -z)]),
        ((1, 0, 0), [(x, 0, -z), (x, height, -z), (x, height, z), (x, 0, z)]),
        ((0, 0, 1), [(x, 0, z), (x, height, z), (-x, height, z), (-x, 0, z)]),
        ((-1, 0, 0), [(-x, 0, z), (-x, height, z), (-x, height, -z), (-x, 0, -z)]),
    ]
    vertices = []
    triangles = []
    for normal, points in faces:
        start = len(vertices)
        vertices.extend((point, normal, (u, v)) for point, (u, v) in zip(points, ((0, 0), (0, 1), (1, 1), (1, 0))))
        triangles.extend((start, start + 1, start + 2, start, start + 2, start + 3))
    lines = ["A", "800", "OBJ", "", f"TEXTURE {texture_name}", "GLOBAL_specular 0.35", "", "POINT_COUNTS 24 0 0 36"]
    for (px, py, pz), (nx, ny, nz), (u, v) in vertices:
        lines.append(f"VT {px:.3f} {py:.3f} {pz:.3f} {nx:.1f} {ny:.1f} {nz:.1f} {u:.3f} {v:.3f}")
    lines.extend(f"IDX {index}" for index in triangles)
    lines.append("TRIS 0 36")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    for obj_name, (dimensions, color) in ASSETS.items():
        texture_name = obj_name.replace(".obj", ".png")
        _texture(output / texture_name, color)
        _obj(output / obj_name, dimensions, texture_name)
    print(f"created {len(ASSETS)} demo assets in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
