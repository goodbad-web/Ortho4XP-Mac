#!/usr/bin/env python3
"""Generate an X-Plane building overlay from OSM XML or GeoJSON."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import warnings
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from shapely.geometry import Polygon, shape


DEFAULT_ASSETS = {
    "house": "objects/jp_house_a.obj",
    "apartments": "objects/jp_apartment_a.obj",
    "commercial": "objects/jp_commercial_a.obj",
    "industrial": "objects/jp_industrial_a.obj",
}

VARIANT_STEMS = {
    "house": "house",
    "apartments": "apartment",
    "commercial": "commercial",
    "industrial": "industrial",
}
SIZE_BUCKETS = ("small", "medium", "large")
HEIGHT_BUCKETS = ("low", "mid", "high")


def variant_asset(category: str, size: str, height: str) -> str:
    return f"objects/jp_{VARIANT_STEMS[category]}_{size}_{height}.obj"


@dataclass(frozen=True)
class Building:
    polygon: Polygon
    lon: float
    lat: float
    width_m: float
    depth_m: float
    heading: float
    height_m: float
    category: str
    source_id: str


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip().replace(",", ".")
        digits = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
        return float(digits) if digits else None
    except (TypeError, ValueError):
        return None


def category_from_tags(tags: dict[str, object]) -> str:
    value = " ".join(
        str(tags.get(key, "")).lower() for key in ("building", "building:use", "use")
    )
    if any(word in value for word in ("industrial", "warehouse", "factory")):
        return "industrial"
    if any(word in value for word in ("commercial", "retail", "office", "shop")):
        return "commercial"
    if any(word in value for word in ("apartments", "residential", "dormitory", "hotel")):
        return "apartments"
    return "house"


def height_from_tags(tags: dict[str, object]) -> float:
    direct = _number(tags.get("height"))
    if direct and 1.5 <= direct <= 180:
        return direct
    levels = _number(tags.get("building:levels"))
    if levels and 1 <= levels <= 60:
        return max(2.8, min(180.0, levels * 2.8 + 1.0))
    return {
        "industrial": 7.0,
        "commercial": 5.5,
        "apartments": 12.0,
        "house": 6.5,
    }[category_from_tags(tags)]


def _local_polygon(coords: Iterable[tuple[float, float]], tile_lat: int, tile_lon: int) -> Polygon:
    scale_x = 111320.0 * math.cos(math.radians(tile_lat + 0.5))
    scale_y = 110540.0
    points = [((lon - tile_lon) * scale_x, (lat - tile_lat) * scale_y) for lon, lat in coords]
    polygon = Polygon(points)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _building_from_polygon(
    polygon: Polygon, tags: dict[str, object], source_id: str, tile_lat: int, tile_lon: int
) -> Building | None:
    if polygon.is_empty or polygon.area < 4 or polygon.geom_type != "Polygon":
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*oriented_envelope.*", category=RuntimeWarning)
        rectangle = polygon.minimum_rotated_rectangle
    corners = list(rectangle.exterior.coords)[:-1]
    edges = [
        (math.hypot(corners[(i + 1) % 4][0] - corners[i][0], corners[(i + 1) % 4][1] - corners[i][1]), i)
        for i in range(4)
    ]
    length, index = max(edges)
    start, end = corners[index], corners[(index + 1) % 4]
    heading = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360
    other = min(edge[0] for edge in edges)
    centroid = polygon.centroid
    scale_x = 111320.0 * math.cos(math.radians(tile_lat + 0.5))
    scale_y = 110540.0
    return Building(
        polygon=polygon,
        lon=tile_lon + centroid.x / scale_x,
        lat=tile_lat + centroid.y / scale_y,
        width_m=max(2.0, float(length)),
        depth_m=max(2.0, float(other)),
        heading=heading,
        height_m=height_from_tags(tags),
        category=category_from_tags(tags),
        source_id=source_id,
    )


def _tags(element: ET.Element) -> dict[str, str]:
    return {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in element.findall("tag")}


def load_osm(path: Path, tile_lat: int, tile_lon: int) -> list[Building]:
    root = ET.parse(path).getroot()
    nodes = {
        node.attrib["id"]: (float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
    }
    result: list[Building] = []
    for way in root.findall("way"):
        tags = _tags(way)
        if "building" not in tags:
            continue
        points = [nodes[ref.attrib["ref"]] for ref in way.findall("nd") if ref.attrib.get("ref") in nodes]
        if len(points) < 4 or points[0] != points[-1]:
            continue
        building = _building_from_polygon(
            _local_polygon(points, tile_lat, tile_lon), tags, way.attrib.get("id", "way"), tile_lat, tile_lon
        )
        if building:
            result.append(building)
    return result


def _geojson_features(path: Path) -> Iterator[tuple[object, dict[str, object], str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", []) if data.get("type") == "FeatureCollection" else [data]
    for index, feature in enumerate(features):
        geometry = feature.get("geometry")
        if geometry:
            yield geometry, feature.get("properties", {}) or {}, str(feature.get("id", index))


def load_geojson(path: Path, tile_lat: int, tile_lon: int) -> list[Building]:
    result: list[Building] = []
    for geojson, properties, source_id in _geojson_features(path):
        geometry = shape(geojson)
        polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for item in polygons:
            if item.geom_type != "Polygon":
                continue
            points = [(lon, lat) for lon, lat in item.exterior.coords]
            building = _building_from_polygon(
                _local_polygon(points, tile_lat, tile_lon), properties, source_id, tile_lat, tile_lon
            )
            if building:
                result.append(building)
    return result


def load_overpass_json(path: Path, tile_lat: int, tile_lon: int) -> list[Building]:
    """Load Overpass ``[out:json]; ...; out geom`` building ways."""
    data = json.loads(path.read_text(encoding="utf-8"))
    result: list[Building] = []
    for element in data.get("elements", []):
        if element.get("type") != "way" or "building" not in element.get("tags", {}):
            continue
        points = [(node["lon"], node["lat"]) for node in element.get("geometry", [])]
        if len(points) < 4 or points[0] != points[-1]:
            continue
        building = _building_from_polygon(
            _local_polygon(points, tile_lat, tile_lon),
            element.get("tags", {}),
            str(element.get("id", "way")),
            tile_lat,
            tile_lon,
        )
        if building:
            result.append(building)
    return result


def _asset_for(category: str, assets: dict[str, str]) -> str:
    return assets.get(category, DEFAULT_ASSETS[category])


def _size_bucket(building: Building) -> str:
    longest = max(building.width_m, building.depth_m)
    if longest < 10:
        return "small"
    if longest < 24:
        return "medium"
    return "large"


def _height_bucket(building: Building) -> str:
    if building.height_m <= 7:
        return "low"
    if building.height_m <= 14:
        return "mid"
    return "high"


def asset_for_building(
    building: Building, assets: dict[str, str], available_assets: set[str] | None = None
) -> str:
    variant = variant_asset(building.category, _size_bucket(building), _height_bucket(building))
    if available_assets is None or variant in available_assets:
        return variant
    return _asset_for(building.category, assets)


def write_library(path: Path, tile_lat: int, tile_lon: int, assets: dict[str, str], mode: str) -> None:
    lines = [
        "A", "1200", "LIBRARY", "",
        f"REGION_DEFINE ortho4xp_ai_buildings_{tile_lat}_{tile_lon}",
        f"REGION_RECT {tile_lon} {tile_lat} {tile_lon} {tile_lat}",
        f"REGION ortho4xp_ai_buildings_{tile_lat}_{tile_lon}",
    ]
    command = "EXPORT_EXTEND" if mode == "extend" else "EXPORT"
    virtual_paths = {
        "house": "simheaven/houses/house_15x20x2.obj",
        "apartments": "simheaven/residential/residential_15x25x6.obj",
        "commercial": "simheaven/commercial/commercial_15x20x3.obj",
        "industrial": "simheaven/industrial/industrial_15x20x2.obj",
    }
    lines.extend(f"{command} {virtual_paths[category]} {assets[category]}" for category in virtual_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text_dsf(
    path: Path,
    tile_lat: int,
    tile_lon: int,
    buildings: list[Building],
    assets: dict[str, str],
    exclude_rect: tuple[int, int, int, int] | None,
    available_assets: set[str] | None = None,
) -> None:
    definitions = sorted({asset_for_building(building, assets, available_assets) for building in buildings})
    definition_index = {asset: index for index, asset in enumerate(definitions)}
    lines = [
        "PROPERTY sim/planet earth", "PROPERTY sim/overlay 1",
        f"PROPERTY sim/west {tile_lon}", f"PROPERTY sim/east {tile_lon + 1}",
        f"PROPERTY sim/south {tile_lat}", f"PROPERTY sim/north {tile_lat + 1}",
    ]
    if exclude_rect:
        west, south, east, north = exclude_rect
        lines.append(f"PROPERTY sim/exclude_objects {west}/{south}/{east}/{north}")
    lines.extend(f"OBJECT_DEF {asset}" for asset in definitions)
    for building in buildings:
        asset = asset_for_building(building, assets, available_assets)
        lines.append(
            f"OBJECT {definition_index[asset]} {building.lon:.7f} {building.lat:.7f} {building.heading:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _convert_dsftool(dsftool: Path, text_dsf: Path, binary_dsf: Path) -> None:
    result = subprocess.run([str(dsftool), "-text2dsf", str(text_dsf), str(binary_dsf)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"DSFTool failed: {result.stdout}\n{result.stderr}")


def definitions_for_report(
    buildings: list[Building], assets: dict[str, str], available_assets: set[str] | None
) -> set[str]:
    return {asset_for_building(building, assets, available_assets) for building in buildings}


def build_package(
    output: Path, tile_lat: int, tile_lon: int, buildings: list[Building],
    assets: dict[str, str], mode: str, dsftool: Path | None,
    exclude_rect: tuple[int, int, int, int] | None, asset_root: Path | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    earth_dir = output / "Earth nav data" / f"{tile_lat // 10 * 10:+03d}{tile_lon // 10 * 10:+04d}"
    earth_dir.mkdir(parents=True, exist_ok=True)
    (output / "objects").mkdir(exist_ok=True)
    missing_assets = []
    if asset_root:
        for relative_path in sorted(set(assets.values())):
            source = asset_root / Path(relative_path).name
            if not source.is_file():
                missing_assets.append(str(source))
        if missing_assets:
            raise FileNotFoundError("asset-root is missing: " + ", ".join(missing_assets))
        if not missing_assets:
            shutil.copytree(asset_root, output / "objects", dirs_exist_ok=True)
    else:
        missing_assets = [str(output / relative_path) for relative_path in sorted(set(assets.values()))]
    write_library(output / "library.txt", tile_lat, tile_lon, assets, mode)
    available_assets = {
        str(path.relative_to(output)) for path in (output / "objects").rglob("*.obj")
    }
    text_dsf = earth_dir / f"{tile_lat:+03d}{tile_lon:+04d}.txt"
    write_text_dsf(text_dsf, tile_lat, tile_lon, buildings, assets, exclude_rect, available_assets or None)
    if dsftool:
        _convert_dsftool(dsftool, text_dsf, text_dsf.with_suffix(".dsf"))
        text_dsf.unlink()
    report = {
        "tile": {"lat": tile_lat, "lon": tile_lon},
        "building_count": len(buildings),
        "categories": {category: sum(item.category == category for item in buildings) for category in sorted({item.category for item in buildings})},
        "height_source_policy": "height, building:levels, category default",
        "exclusion": "explicit rectangle" if exclude_rect else "none; compare for duplicate buildings",
        "missing_assets": [
            str(output / asset)
            for asset in sorted(set(definitions_for_report(buildings, assets, available_assets or None)))
            if not (output / asset).is_file()
        ],
    }
    (output / "generation-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _tile(value: str) -> int:
    parsed = int(value)
    if parsed < -90 or parsed > 89:
        raise argparse.ArgumentTypeError("tile latitude must be between -90 and 89")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=_tile, required=True)
    parser.add_argument("--lon", type=int, required=True)
    parser.add_argument("--osm", type=Path)
    parser.add_argument("--geojson", type=Path, action="append")
    parser.add_argument("--overpass-json", type=Path, help="Overpass JSON from a way[building] query with out geom")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, help="directory containing the four self-owned OBJ files")
    parser.add_argument("--dsftool", type=Path)
    parser.add_argument("--mode", choices=("replace", "extend"), default="replace")
    parser.add_argument("--exclude-rect", nargs=4, type=int, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    args = parser.parse_args(argv)
    if not args.osm and not args.geojson and not args.overpass_json:
        parser.error("--osm, --geojson, or --overpass-json is required")
    if args.lon < -180 or args.lon > 179:
        parser.error("tile longitude must be between -180 and 179")
    buildings = load_osm(args.osm, args.lat, args.lon) if args.osm else []
    for path in args.geojson or []:
        buildings.extend(load_geojson(path, args.lat, args.lon))
    if args.overpass_json:
        buildings.extend(load_overpass_json(args.overpass_json, args.lat, args.lon))
    exclude = tuple(args.exclude_rect) if args.exclude_rect else None
    build_package(args.output, args.lat, args.lon, buildings, dict(DEFAULT_ASSETS), args.mode, args.dsftool, exclude, args.asset_root)
    print(f"generated {len(buildings)} buildings in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
