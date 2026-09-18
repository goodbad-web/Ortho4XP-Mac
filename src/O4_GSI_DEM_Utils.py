"""GSI numerical elevation model import, cataloguing, and conversion.

The GSI download service distributes DEM data as ZIP archives containing
JPGIS/GML files.  This module deliberately keeps the archive management and
the raster conversion in one place so the command-line and Tk interfaces use
identical rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional
import xml.etree.ElementTree as ET

import numpy as np

try:
    from osgeo import gdal, osr

    HAS_GDAL = True
    gdal.UseExceptions()
    osr.UseExceptions()
except Exception:  # pragma: no cover - exercised on installations without GDAL
    gdal = None
    osr = None
    HAS_GDAL = False


PRODUCTS = ("DEM1A", "DEM5A", "DEM5B", "DEM5C", "DEM10A", "DEM10B")
PRODUCT_SCORE = {
    "DEM10B": 0,
    "DEM10A": 1,
    "DEM5C": 2,
    "DEM5B": 3,
    "DEM5A": 4,
    "DEM1A": 5,
}
PRODUCT_ARCSEC = {
    "DEM1A": 0.04,
    "DEM5A": 0.20,
    "DEM5B": 0.20,
    "DEM5C": 0.20,
    "DEM10A": 0.40,
    "DEM10B": 0.40,
}
RESOLUTION_NAMES = {
    "DEM1A": "1m",
    "DEM5A": "5m",
    "DEM5B": "5m",
    "DEM5C": "5m",
    "DEM10A": "10m",
    "DEM10B": "10m",
}
RESOLUTION_ARCSEC = {"1m": 0.04, "5m": 0.20, "10m": 0.40}
CATALOG_VERSION = 1
MANIFEST_VERSION = 1
NO_DATA = -9999.0
HGT_NO_DATA = -32768

ProgressCallback = Callable[[str, int, int, str], None]


class GSIError(RuntimeError):
    """Expected input, CRS, dependency, or output error."""


class GSICancelled(GSIError):
    """Raised when a cooperative cancellation request is observed."""


@dataclass(frozen=True)
class GSIPaths:
    root: Path
    input_dir: Path
    output_dir: Path
    catalog_path: Path


@dataclass(frozen=True)
class GSIRegion:
    label: str
    south: float
    west: float
    north: float
    east: float
    mesh_code: Optional[str] = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.south, self.west, self.north, self.east)


@dataclass
class GSIOptions:
    input_dir: Path
    output_dir: Path
    mesh_codes: tuple[str, ...] = ()
    bbox: Optional[tuple[float, float, float, float]] = None
    resolution: str = "auto"
    make_vrt: bool = False
    hgt_tiles: tuple[str, ...] = ()
    overwrite: bool = False
    source_crs: Optional[str] = None


@dataclass
class GSIImportResult:
    imported: int = 0
    skipped_duplicates: int = 0
    quarantined: int = 0
    failed: int = 0
    catalog_path: Optional[str] = None
    zip_count: int = 0
    product_counts: dict[str, int] = field(default_factory=dict)
    date_counts: dict[str, int] = field(default_factory=dict)
    cancelled: bool = False
    messages: list[str] = field(default_factory=list)


@dataclass
class GSIScanResult:
    entries: list[dict]
    ready: int
    duplicates: int
    quarantined: int
    invalid: int
    modified: int = 0
    missing: int = 0
    catalog_path: Optional[str] = None
    zip_count: int = 0
    product_counts: dict[str, int] = field(default_factory=dict)
    date_counts: dict[str, int] = field(default_factory=dict)
    selection_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class GSIBuildResult:
    outputs: list[str] = field(default_factory=list)
    vrt: Optional[str] = None
    manifest: Optional[str] = None
    failures: list[dict] = field(default_factory=list)
    cancelled: bool = False
    recommended_custom_dem: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return bool(self.outputs) and not self.failures and not self.cancelled


@dataclass
class GSIBlock:
    product: str
    date: str
    mesh_code: str
    south: float
    west: float
    north: float
    east: float
    values: np.ndarray
    source_path: Path
    xml_name: str
    source_crs: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.south, self.west, self.north, self.east)


def default_paths(project_root: Optional[Path] = None) -> GSIPaths:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    gsi_root = root / "Elevation_data" / "GSI"
    return GSIPaths(
        root=root,
        input_dir=gsi_root / "input",
        output_dir=gsi_root / "output",
        catalog_path=gsi_root / "input" / "catalog.json",
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(root: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _local_name(element.tag) == name]


def _first_text(root: ET.Element, name: str) -> Optional[str]:
    for element in _children(root, name):
        if element.text and element.text.strip():
            return element.text.strip()
    return None


def _date_from_name_or_xml(name: str, xml_dates: Iterable[str]) -> Optional[str]:
    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", name)
    if match:
        return match.group(1)
    for value in xml_dates:
        match = re.search(r"(20\d{2})[-/]?(\d{2})[-/]?(\d{2})", value)
        if match:
            return "".join(match.groups())
    return None


def _product_from_text(*values: Optional[str]) -> Optional[str]:
    joined = " ".join(value or "" for value in values).upper()
    for product in PRODUCTS:
        if product in joined:
            return product
    type_text = " ".join(value or "" for value in values)
    if "1m" in type_text:
        return "DEM1A"
    if "5m" in type_text:
        return "DEM5A"
    if "10m" in type_text:
        return "DEM10B"
    return None


def _parse_metadata(xml_bytes: bytes, archive_name: str = "") -> Optional[dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    product = _product_from_text(
        archive_name,
        _first_text(root, "type"),
        _first_text(root, "description"),
    )
    if product not in PRODUCTS:
        return None

    mesh_codes = sorted(
        {
            value.strip()
            for element in _children(root, "mesh")
            for value in [(element.text or "").strip()]
            if value and value.isdigit()
        }
    )
    dates = [
        element.text.strip()
        for element in _children(root, "timePosition")
        if element.text and element.text.strip()
    ]
    date = _date_from_name_or_xml(archive_name, dates)
    envelope = next(iter(_children(root, "Envelope")), None)
    srs_name = envelope.attrib.get("srsName", "") if envelope is not None else ""
    lower = _first_text(root, "lowerCorner")
    upper = _first_text(root, "upperCorner")
    high = _first_text(root, "high")
    if not mesh_codes or not date or not lower or not upper or not high:
        return None

    try:
        south, west = map(float, lower.split()[:2])
        north, east = map(float, upper.split()[:2])
        high_x, high_y = map(int, high.split()[:2])
    except (TypeError, ValueError):
        return None
    if north <= south or east <= west or high_x < 0 or high_y < 0:
        return None

    start = _first_text(root, "startPoint") or "0 0"
    try:
        start_x, start_y = map(int, start.split()[:2])
    except (TypeError, ValueError):
        start_x, start_y = 0, 0

    return {
        "root": root,
        "product": product,
        "date": date,
        "mesh_codes": mesh_codes,
        "south": south,
        "west": west,
        "north": north,
        "east": east,
        "width": high_x + 1,
        "height": high_y + 1,
        "start_x": start_x,
        "start_y": start_y,
        "srs_name": srs_name,
    }


def _parse_values(metadata: dict) -> np.ndarray:
    tuple_element = next(iter(_children(metadata["root"], "tupleList")), None)
    if tuple_element is None or not tuple_element.text:
        raise GSIError("GSI XML has no tupleList")

    values: list[float] = []
    for raw_line in tuple_element.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "," in line:
            kind, value_text = line.rsplit(",", 1)
        else:
            kind, value_text = "", line
        try:
            value = float(value_text.strip())
        except ValueError:
            value = np.nan
        if kind.strip() in ("データなし", "データ無し") or value <= -9990:
            value = np.nan
        values.append(value)

    expected = metadata["width"] * metadata["height"]
    if not values:
        raise GSIError("GSI XML has an empty tupleList")
    values_array = np.asarray(values, dtype=np.float32)
    result = np.full(expected, np.nan, dtype=np.float32)
    start = metadata["start_y"] * metadata["width"] + metadata["start_x"]
    if start < 0 or start >= expected:
        raise GSIError("GSI XML startPoint is outside the grid")
    count = min(values_array.size, expected - start)
    result[start : start + count] = values_array[:count]
    return result.reshape((metadata["height"], metadata["width"]))


def _require_gdal() -> None:
    if not HAS_GDAL:
        raise GSIError("GDAL/OSR is required for GSI raster conversion")


def _spatial_reference(value: str):
    _require_gdal()
    spatial_ref = osr.SpatialReference()
    try:
        spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception:
        pass
    try:
        error = spatial_ref.SetFromUserInput(value)
    except Exception as exc:
        raise GSIError(f"Could not resolve CRS {value!r}: {exc}") from exc
    if error != 0:
        raise GSIError(
            f"Could not resolve CRS {value!r}. Install a PROJ database with "
            "the required CRS definition or pass --source-crs explicitly."
        )
    return spatial_ref


def _resolve_source_crs(srs_name: str, override: Optional[str] = None):
    value = (override or srs_name or "").lower()
    if "jgd2024" in value or "jgd_2024" in value:
        # PROJ versions used by current GDAL identify the geographic JGD2024
        # CRS as ESRI:104221 even when the GSI XML uses fguuid:jgd2024.bl.
        return _spatial_reference("ESRI:104221"), "JGD2024"
    if "jgd2011" in value or "jgd_2011" in value or "6668" in value:
        return _spatial_reference("EPSG:6668"), "JGD2011"
    if "jgd2000" in value or "jgd_2000" in value or "4612" in value:
        # Older official DEM10B archives use the JGD2000 GML identifier.
        return _spatial_reference("EPSG:4612"), "JGD2000"
    if "4326" in value or "wgs84" in value or "wgs 84" in value:
        return _spatial_reference("EPSG:4326"), "WGS84"
    if not override:
        raise GSIError(
            f"Unknown GSI CRS {srs_name!r}; use --source-crs only when the "
            "source coordinate system is known."
        )
    return _spatial_reference(override), override


def _transform_bounds(
    bounds: tuple[float, float, float, float], source_ref
) -> tuple[float, float, float, float]:
    target_ref = _spatial_reference("EPSG:4326")
    transform = osr.CoordinateTransformation(source_ref, target_ref)
    south, west, north, east = bounds
    points = [
        transform.TransformPoint(west, south),
        transform.TransformPoint(west, north),
        transform.TransformPoint(east, south),
        transform.TransformPoint(east, north),
    ]
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return min(lats), min(lons), max(lats), max(lons)


def parse_gsi_xml(
    xml_bytes: bytes,
    source_path: Path,
    xml_name: str,
    source_crs: Optional[str] = None,
) -> GSIBlock:
    metadata = _parse_metadata(xml_bytes, source_path.name)
    if metadata is None:
        raise GSIError(f"Unsupported or incomplete GSI XML: {xml_name}")
    source_ref, source_label = _resolve_source_crs(
        metadata["srs_name"], source_crs
    )
    bounds = _transform_bounds(
        (
            metadata["south"],
            metadata["west"],
            metadata["north"],
            metadata["east"],
        ),
        source_ref,
    )
    return GSIBlock(
        product=metadata["product"],
        date=metadata["date"],
        mesh_code=metadata["mesh_codes"][0],
        south=bounds[0],
        west=bounds[1],
        north=bounds[2],
        east=bounds[3],
        values=_parse_values(metadata),
        source_path=source_path,
        xml_name=xml_name,
        source_crs=source_label,
    )


def _archive_metadata(path: Path) -> dict:
    if not zipfile.is_zipfile(path):
        raise GSIError("not a ZIP archive")
    meshes: set[str] = set()
    products: set[str] = set()
    dates: set[str] = set()
    xml_count = 0
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        for name in names:
            xml_count += 1
            metadata = _parse_metadata(archive.read(name), path.name)
            if metadata is None:
                continue
            products.add(metadata["product"])
            dates.add(metadata["date"])
            meshes.update(metadata["mesh_codes"])
    if xml_count == 0:
        raise GSIError("ZIP contains no XML files")
    if len(products) != 1:
        raise GSIError("ZIP does not contain exactly one supported DEM product")
    if not meshes:
        raise GSIError("ZIP contains no GSI mesh code")
    product = next(iter(products))
    date = sorted(dates)[-1] if dates else None
    if date is None:
        raise GSIError("ZIP contains no creation date")
    return {
        "product": product,
        "date": date,
        "mesh_codes": sorted(meshes),
        "xml_count": xml_count,
    }


def _sha256(path: Path, callback: Optional[ProgressCallback] = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_archives(root: Path, include_quarantine: bool = False) -> list[Path]:
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.rglob("*.zip")):
        if not include_quarantine and "_quarantine" in path.parts:
            continue
        if not any(part.startswith(".") for part in path.relative_to(root).parts):
            result.append(path)
    return result


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_catalog(path: Path) -> dict:
    if not path.is_file():
        return {"version": CATALOG_VERSION, "updated_at": None, "entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GSIError(f"Could not read catalog {path}: {exc}") from exc
    if payload.get("version") != CATALOG_VERSION or not isinstance(
        payload.get("entries"), list
    ):
        raise GSIError(f"Unsupported catalog format: {path}")
    return payload


def _catalog_entry(
    path: Path,
    input_dir: Path,
    digest: str,
    metadata: dict,
    status: str = "ready",
    reason: Optional[str] = None,
    source_path: Optional[Path] = None,
) -> dict:
    entry = {
        "path": str(path.relative_to(input_dir)),
        "sha256": digest,
        "size": path.stat().st_size,
        "product": metadata.get("product"),
        "date": metadata.get("date"),
        "mesh_codes": metadata.get("mesh_codes", []),
        "xml_count": metadata.get("xml_count", 0),
        "status": status,
        "source_path": str(source_path) if source_path else None,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if reason:
        entry["reason"] = reason
    return entry


def scan_gsi_input(
    input_dir: Path,
    write_catalog: bool = True,
    progress: Optional[ProgressCallback] = None,
    cancel_event=None,
    source_paths: Optional[dict[str, str]] = None,
    quarantine_reasons: Optional[dict[str, str]] = None,
) -> GSIScanResult:
    input_dir = Path(input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = input_dir / "catalog.json"
    previous_entries = {
        str(entry.get("path")): entry
        for entry in load_catalog(catalog_path).get("entries", [])
        if entry.get("path")
    }
    source_paths = source_paths or {}
    quarantine_reasons = quarantine_reasons or {}
    paths = _iter_archives(input_dir)
    entries: list[dict] = []
    seen_hashes: dict[str, str] = {}
    invalid = 0

    for index, path in enumerate(paths, 1):
        _check_cancel(cancel_event)
        _report(progress, "scan", index, len(paths), path.name)
        digest = None
        relative = str(path.relative_to(input_dir))
        previous = previous_entries.get(relative)
        changed = False
        try:
            digest = _sha256(path)
            changed = bool(
                previous
                and (
                    previous.get("sha256") != digest
                    or previous.get("size") != path.stat().st_size
                )
            )
            metadata = _archive_metadata(path)
            source_path = source_paths.get(relative)
            if not source_path and previous:
                source_path = previous.get("source_path")
            if digest in seen_hashes:
                entry = _catalog_entry(
                    path,
                    input_dir,
                    digest,
                    metadata,
                    status="duplicate",
                    reason=f"same content as {seen_hashes[digest]}",
                    source_path=Path(source_path) if source_path else None,
                )
            else:
                seen_hashes[digest] = relative
                entry = _catalog_entry(
                    path,
                    input_dir,
                    digest,
                    metadata,
                    source_path=Path(source_path) if source_path else None,
                )
            if changed:
                entry["status"] = "modified"
                entry["reason"] = "file changed since the previous catalog scan"
        except Exception as exc:
            if not previous or not changed:
                invalid += 1
            entry = {
                "path": relative,
                "sha256": digest,
                "size": path.stat().st_size,
                "product": None,
                "date": None,
                "mesh_codes": [],
                "xml_count": 0,
                "status": "modified" if previous and changed else "invalid",
                "reason": (
                    "file changed since the previous catalog scan; " + str(exc)
                    if previous and changed
                    else str(exc)
                ),
                "source_path": (
                    source_paths.get(relative)
                    or (previous.get("source_path") if previous else None)
                ),
                "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        entries.append(entry)

    current_paths = {entry["path"] for entry in entries}
    for relative, previous in previous_entries.items():
        if relative in current_paths or str(relative).startswith("_quarantine/"):
            continue
        if previous.get("status") not in {"ready", "duplicate", "modified", "missing"}:
            continue
        entries.append(
            {
                "path": relative,
                "sha256": previous.get("sha256"),
                "size": previous.get("size"),
                "product": previous.get("product"),
                "date": previous.get("date"),
                "mesh_codes": previous.get("mesh_codes", []),
                "xml_count": previous.get("xml_count", 0),
                "status": "missing",
                "reason": "catalog input is missing",
                "source_path": previous.get("source_path"),
                "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )

    latest_dates: dict[tuple[str, str], str] = {}
    for entry in entries:
        if entry.get("status") != "ready":
            continue
        product = entry.get("product")
        date = entry.get("date")
        for mesh_code in entry.get("mesh_codes", []):
            if product and date:
                key = (str(product), str(mesh_code))
                latest_dates[key] = max(date, latest_dates.get(key, ""))
    for entry in entries:
        status = entry.get("status")
        if status == "ready":
            entry["selection"] = (
                "candidate"
                if any(
                    latest_dates.get((str(entry.get("product")), str(mesh_code)))
                    == entry.get("date")
                    for mesh_code in entry.get("mesh_codes", [])
                )
                else "unused"
            )
        elif status == "duplicate":
            entry["selection"] = "duplicate"
        else:
            entry["selection"] = status

    quarantined_paths = _iter_archives(input_dir, include_quarantine=True)
    for path in quarantined_paths:
        if "_quarantine" not in path.parts:
            continue
        relative = str(path.relative_to(input_dir))
        previous_quarantine = previous_entries.get(relative)
        if any(entry["path"] == relative for entry in entries):
            continue
        try:
            digest = _sha256(path)
        except OSError:
            digest = None
        entries.append(
            {
                "path": relative,
                "sha256": digest,
                "size": path.stat().st_size,
                "product": None,
                "date": None,
                "mesh_codes": [],
                "xml_count": 0,
                "status": "quarantined",
                "selection": "quarantined",
                "reason": (
                    quarantine_reasons.get(relative)
                    or (previous_quarantine.get("reason") if previous_quarantine else None)
                    or "quarantine archive; reason unavailable"
                ),
                "source_path": (
                    source_paths.get(relative)
                    or (previous_quarantine.get("source_path") if previous_quarantine else None)
                ),
                "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )

    if write_catalog:
        _atomic_json_write(
            catalog_path,
            {
                "version": CATALOG_VERSION,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "entries": entries,
            },
        )
    return GSIScanResult(
        entries=entries,
        ready=sum(entry["status"] == "ready" for entry in entries),
        duplicates=sum(entry["status"] == "duplicate" for entry in entries),
        quarantined=sum(entry["status"] == "quarantined" for entry in entries),
        invalid=invalid,
        modified=sum(entry["status"] == "modified" for entry in entries),
        missing=sum(entry["status"] == "missing" for entry in entries),
        catalog_path=str(catalog_path) if write_catalog else None,
        zip_count=len(entries),
        product_counts=_count_field(entries, "product", {"ready", "duplicate"}),
        date_counts=_count_field(entries, "date", {"ready", "duplicate"}),
        selection_counts=_count_field(entries, "selection"),
    )


def _count_field(
    entries: Iterable[dict], field_name: str, statuses: Optional[set[str]] = None
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        if statuses is not None and entry.get("status") not in statuses:
            continue
        value = entry.get(field_name)
        if value:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _unique_destination(directory: Path, filename: str, digest: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    if _sha256(destination) == digest:
        return destination
    stem = destination.stem
    return directory / f"{stem}_{digest[:8]}{destination.suffix}"


def _quarantine_destination(directory: Path, filename: str, digest: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    if _sha256(destination) == digest:
        return destination
    return directory / f"{Path(filename).stem}_{digest[:8]}{Path(filename).suffix}"


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    try:
        shutil.copy2(source, temp_name)
        os.replace(temp_name, destination)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def import_gsi_archives(
    source_dir: Path,
    input_dir: Path,
    progress: Optional[ProgressCallback] = None,
    cancel_event=None,
) -> GSIImportResult:
    source_dir = Path(source_dir).resolve()
    input_dir = Path(input_dir).resolve()
    if not source_dir.is_dir():
        raise GSIError(f"Import source directory does not exist: {source_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = [path for path in _iter_archives(source_dir, include_quarantine=True)]
    result = GSIImportResult()
    source_paths: dict[str, str] = {}
    quarantine_reasons: dict[str, str] = {}
    existing_hashes = {
        entry.get("sha256")
        for entry in load_catalog(input_dir / "catalog.json").get("entries", [])
        if entry.get("sha256")
    }

    try:
        for index, source_path in enumerate(paths, 1):
            _check_cancel(cancel_event)
            _report(progress, "import", index, len(paths), source_path.name)
            digest = _sha256(source_path)
            if digest in existing_hashes:
                result.skipped_duplicates += 1
                continue
            try:
                metadata = _archive_metadata(source_path)
                destination_dir = input_dir / metadata["product"] / metadata["date"]
                destination = _unique_destination(
                    destination_dir, source_path.name, digest
                )
                if not destination.exists():
                    _copy_atomic(source_path, destination)
                existing_hashes.add(digest)
                source_paths[str(destination.relative_to(input_dir))] = str(source_path)
                result.imported += 1
            except Exception as exc:
                quarantine_dir = input_dir / "_quarantine"
                destination = _quarantine_destination(
                    quarantine_dir, source_path.name, digest
                )
                if not destination.exists():
                    _copy_atomic(source_path, destination)
                relative = str(destination.relative_to(input_dir))
                source_paths[relative] = str(source_path)
                quarantine_reasons[relative] = str(exc)
                result.quarantined += 1
                result.messages.append(f"{source_path.name}: {exc}")
    except GSICancelled:
        result.cancelled = True

    scan = scan_gsi_input(
        input_dir,
        write_catalog=True,
        progress=progress,
        cancel_event=None,
        source_paths=source_paths,
        quarantine_reasons=quarantine_reasons,
    )
    result.catalog_path = scan.catalog_path
    result.zip_count = scan.zip_count
    result.product_counts = scan.product_counts
    result.date_counts = scan.date_counts
    return result


def mesh_code_bounds(mesh_code: str) -> tuple[float, float, float, float]:
    code = str(mesh_code).strip()
    if not re.fullmatch(r"\d{8}", code):
        raise GSIError(f"Invalid third-level Japanese mesh code: {mesh_code}")
    first_lat = int(code[0:2]) / 1.5
    first_lon = 100.0 + int(code[2:4])
    south = first_lat + int(code[4]) * 5.0 / 60.0
    west = first_lon + int(code[5]) * 7.5 / 60.0
    south += int(code[6]) * 30.0 / 3600.0
    west += int(code[7]) * 45.0 / 3600.0
    return (south, west, south + 30.0 / 3600.0, west + 45.0 / 3600.0)


def _region_for_mesh(code: str) -> GSIRegion:
    south, west, north, east = mesh_code_bounds(code)
    return GSIRegion(code, south, west, north, east, mesh_code=code)


def _validate_region(region: GSIRegion) -> None:
    if not (-90 <= region.south < region.north <= 90):
        raise GSIError(f"Invalid latitude bounds for {region.label}")
    if not (-180 <= region.west < region.east <= 180):
        raise GSIError(f"Invalid longitude bounds for {region.label}")


def regions_from_options(options: GSIOptions) -> list[GSIRegion]:
    regions: list[GSIRegion] = []
    for code in options.mesh_codes:
        regions.append(_region_for_mesh(code))
    if options.bbox is not None:
        south, west, north, east = options.bbox
        region = GSIRegion(
            _bbox_label(options.bbox), south, west, north, east, mesh_code=None
        )
        regions.append(region)
    for region in regions:
        _validate_region(region)
    return regions


def _bbox_label(bbox: tuple[float, float, float, float]) -> str:
    return "bbox_" + "_".join(f"{value:.6f}".replace("-", "m").replace(".", "p") for value in bbox)


def _overlaps(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    south_a, west_a, north_a, east_a = first
    south_b, west_b, north_b, east_b = second
    return not (
        north_a <= south_b
        or south_a >= north_b
        or east_a <= west_b
        or west_a >= east_b
    )


def _record_may_contain(entry: dict, region: GSIRegion) -> bool:
    if not region.mesh_code:
        return True
    requested = region.mesh_code
    for value in entry.get("mesh_codes", []):
        value = str(value)
        if requested.startswith(value) or value.startswith(requested[:6]):
            return True
    return False


def _archive_blocks(
    path: Path,
    region: GSIRegion,
    source_crs: Optional[str],
    cancel_event=None,
) -> list[GSIBlock]:
    blocks: list[GSIBlock] = []
    with zipfile.ZipFile(path) as archive:
        for xml_name in archive.namelist():
            _check_cancel(cancel_event)
            if not xml_name.lower().endswith(".xml"):
                continue
            xml_bytes = archive.read(xml_name)
            metadata = _parse_metadata(xml_bytes, path.name)
            if metadata is None:
                continue
            source_bounds = (
                metadata["south"],
                metadata["west"],
                metadata["north"],
                metadata["east"],
            )
            # Bounds are geographic in the GSI GML.  The CRS transform is
            # repeated by parse_gsi_xml only for blocks that intersect.
            if not _overlaps(source_bounds, region.bounds):
                continue
            block = parse_gsi_xml(xml_bytes, path, xml_name, source_crs)
            if _overlaps(block.bounds, region.bounds):
                blocks.append(block)
    return blocks


def _candidate_entries(scan: GSIScanResult, region: GSIRegion) -> list[dict]:
    return [
        entry
        for entry in scan.entries
        if entry.get("status") == "ready" and _record_may_contain(entry, region)
    ]


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GSICancelled("cancelled")


def _report(
    callback: Optional[ProgressCallback],
    stage: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(stage, completed, total, message)


def _target_resolution(blocks: list[GSIBlock], requested: str) -> tuple[str, float]:
    if not blocks:
        raise GSIError("No GSI DEM blocks overlap the requested region")
    if requested == "1arcsec":
        return "1arcsec", 1.0
    if requested == "auto":
        arcsec = min(PRODUCT_ARCSEC[block.product] for block in blocks)
        name = "1m" if arcsec == 0.04 else "5m" if arcsec == 0.20 else "10m"
        return name, arcsec
    if requested not in RESOLUTION_ARCSEC:
        raise GSIError(f"Unsupported output resolution: {requested}")
    requested_arcsec = RESOLUTION_ARCSEC[requested]
    matching = [
        block
        for block in blocks
        if PRODUCT_ARCSEC[block.product] <= requested_arcsec + 1e-9
    ]
    if not matching:
        raise GSIError(
            f"No source at or above requested {requested} resolution; "
            "low-resolution DEMs are not upsampled as a fake 1m source."
        )
    return requested, requested_arcsec


def _insert_block(
    output: np.ndarray,
    block: GSIBlock,
    region: GSIRegion,
    resolution_deg: float,
    point_grid: bool = False,
) -> int:
    height, width = output.shape
    src_height, src_width = block.values.shape
    if src_width <= 0 or src_height <= 0:
        return 0
    source_south, source_west, source_north, source_east = block.bounds
    dlon = (source_east - source_west) / src_width
    dlat = (source_north - source_south) / src_height
    if dlon <= 0 or dlat <= 0:
        return 0

    if point_grid:
        col_start = max(0, int(np.floor((source_west - region.west) / resolution_deg)))
        col_end = min(width - 1, int(np.ceil((source_east - region.west) / resolution_deg)))
        row_start = max(0, int(np.floor((region.north - source_north) / resolution_deg)))
        row_end = min(height - 1, int(np.ceil((region.north - source_south) / resolution_deg)))
    else:
        col_start = max(0, int(np.floor((source_west - region.west) / resolution_deg)))
        col_end = min(width - 1, int(np.ceil((source_east - region.west) / resolution_deg)) - 1)
        row_start = max(0, int(np.floor((region.north - source_north) / resolution_deg)))
        row_end = min(height - 1, int(np.ceil((region.north - source_south) / resolution_deg)) - 1)
    if col_end < col_start or row_end < row_start:
        return 0

    cols = np.arange(col_start, col_end + 1, dtype=np.int32)
    rows = np.arange(row_start, row_end + 1, dtype=np.int32)
    if point_grid:
        target_lons = region.west + cols.astype(np.float64) * resolution_deg
        target_lats = region.north - rows.astype(np.float64) * resolution_deg
    else:
        target_lons = region.west + (cols.astype(np.float64) + 0.5) * resolution_deg
        target_lats = region.north - (rows.astype(np.float64) + 0.5) * resolution_deg

    x_float = (target_lons - source_west) / dlon - 0.5
    y_float = (source_north - target_lats) / dlat - 0.5
    valid_x = (x_float >= -0.5) & (x_float <= src_width - 0.5)
    valid_y = (y_float >= -0.5) & (y_float <= src_height - 0.5)
    x_index = np.clip(np.rint(x_float).astype(np.int32), 0, src_width - 1)
    y_index = np.clip(np.rint(y_float).astype(np.int32), 0, src_height - 1)
    sampled = block.values[np.ix_(y_index, x_index)]
    good = valid_y[:, None] & valid_x[None, :] & np.isfinite(sampled)
    if not np.any(good):
        return 0
    destination = output[np.ix_(rows, cols)]
    destination[good] = sampled[good]
    output[np.ix_(rows, cols)] = destination
    return int(np.count_nonzero(good))


def _grid_for_region(
    region: GSIRegion, arcsec: float, point_grid: bool = False
) -> tuple[np.ndarray, float]:
    resolution_deg = arcsec / 3600.0
    if point_grid:
        width = int(round((region.east - region.west) / resolution_deg)) + 1
        height = int(round((region.north - region.south) / resolution_deg)) + 1
    else:
        width = max(1, int(round((region.east - region.west) / resolution_deg)))
        height = max(1, int(round((region.north - region.south) / resolution_deg)))
    return np.full((height, width), np.nan, dtype=np.float32), resolution_deg


def _atomic_output_path(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise GSIError(f"Output already exists; pass --overwrite: {path}")
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_geotiff(
    path: Path,
    raster: np.ndarray,
    region: GSIRegion,
    resolution_deg: float,
    overwrite: bool = False,
) -> None:
    _require_gdal()
    path = Path(path)
    temp_path = _atomic_output_path(path, overwrite)
    if temp_path.exists():
        temp_path.unlink()
    try:
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            str(temp_path),
            raster.shape[1],
            raster.shape[0],
            1,
            gdal.GDT_Float32,
            options=[
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "COMPRESS=DEFLATE",
                "PREDICTOR=3",
                "ZLEVEL=6",
                "BIGTIFF=IF_SAFER",
                "NUM_THREADS=ALL_CPUS",
            ],
        )
        if dataset is None:
            raise GSIError(f"Could not create GeoTIFF: {temp_path}")
        dataset.SetGeoTransform(
            (region.west, resolution_deg, 0.0, region.north, 0.0, -resolution_deg)
        )
        spatial_ref = _spatial_reference("EPSG:4326")
        dataset.SetProjection(spatial_ref.ExportToWkt())
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(NO_DATA)
        for row_start in range(0, raster.shape[0], 512):
            row_end = min(raster.shape[0], row_start + 512)
            values = np.where(
                np.isfinite(raster[row_start:row_end]),
                raster[row_start:row_end],
                NO_DATA,
            ).astype(np.float32)
            band.WriteArray(values, xoff=0, yoff=row_start)
        band.FlushCache()
        dataset.FlushCache()
        dataset = None

        check = gdal.Open(str(temp_path), gdal.GA_ReadOnly)
        if check is None:
            raise GSIError(f"GDAL could not reopen GeoTIFF: {temp_path}")
        if (check.RasterXSize, check.RasterYSize) != (raster.shape[1], raster.shape[0]):
            raise GSIError("Generated GeoTIFF dimensions do not match the raster")
        check = None
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_hgt(
    path: Path,
    raster: np.ndarray,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    temp_path = _atomic_output_path(path, overwrite)
    try:
        values = np.where(np.isfinite(raster), np.rint(raster), HGT_NO_DATA)
        values = np.clip(values, -32768, 32767).astype(">i2")
        values.tofile(temp_path)
        expected = raster.shape[0] * raster.shape[1] * 2
        if temp_path.stat().st_size != expected:
            raise GSIError(f"Invalid HGT size: {temp_path.stat().st_size} != {expected}")
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _build_vrt(path: Path, sources: list[Path], overwrite: bool) -> None:
    _require_gdal()
    if not sources:
        raise GSIError("Cannot create VRT without GeoTIFF sources")
    temp_path = _atomic_output_path(path, overwrite)
    try:
        options = gdal.BuildVRTOptions(
            srcNodata=NO_DATA,
            VRTNodata=NO_DATA,
            resampleAlg="nearest",
        )
        dataset = gdal.BuildVRT(str(temp_path), [str(source) for source in sources], options=options)
        if dataset is None:
            raise GSIError(f"Could not create VRT: {path}")
        dataset.FlushCache()
        dataset = None
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _write_manifest(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise GSIError(f"Output already exists; pass --overwrite: {path}")
    _atomic_json_write(path, payload)


def _hgt_region(tile: str) -> GSIRegion:
    match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", tile.upper().strip())
    if not match:
        raise GSIError(f"Invalid HGT tile name: {tile}")
    lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
    lon = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
    return GSIRegion(tile.upper(), lat, lon, lat + 1, lon + 1)


def _relative_input(path: Path, input_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(input_dir.resolve()))
    except ValueError:
        return str(path)


def _build_region(
    region: GSIRegion,
    entries: list[dict],
    input_dir: Path,
    options: GSIOptions,
    progress: Optional[ProgressCallback],
    cancel_event,
    target_resolution: Optional[str] = None,
    point_grid: bool = False,
) -> tuple[Path, dict]:
    blocks: list[GSIBlock] = []
    candidate_total = len(entries)
    for index, entry in enumerate(entries, 1):
        _check_cancel(cancel_event)
        archive_path = input_dir / entry["path"]
        if not archive_path.is_file():
            raise GSIError(f"Catalog input is missing: {archive_path}")
        _report(progress, "build", index, candidate_total, f"{region.label}: {archive_path.name}")
        blocks.extend(
            _archive_blocks(archive_path, region, options.source_crs, cancel_event)
        )
    if not blocks:
        raise GSIError(f"No GSI blocks found for {region.label}")

    resolution_name, arcsec = _target_resolution(
        blocks, target_resolution or options.resolution
    )
    raster, resolution_deg = _grid_for_region(region, arcsec, point_grid=point_grid)
    source_counts: dict[str, int] = {product: 0 for product in PRODUCTS}
    ordered_blocks = sorted(
        blocks,
        key=lambda block: (
            PRODUCT_SCORE[block.product],
            block.date,
            str(block.source_path),
            block.xml_name,
        ),
    )
    for block in ordered_blocks:
        _check_cancel(cancel_event)
        inserted = _insert_block(
            raster, block, region, resolution_deg, point_grid=point_grid
        )
        source_counts[block.product] += inserted

    valid = np.isfinite(raster)
    output_name = f"{region.label}_GSI_{resolution_name}.tif"
    output_path = options.output_dir / output_name
    if point_grid:
        output_path = options.output_dir / "hgt" / f"{region.label}.hgt"
        write_hgt(output_path, raster, overwrite=options.overwrite)
    else:
        write_geotiff(
            output_path,
            raster,
            region,
            resolution_deg,
            overwrite=options.overwrite,
        )
    return output_path, {
        "region": asdict(region),
        "resolution": resolution_name,
        "source_resolution_arcsec": arcsec,
        "width": int(raster.shape[1]),
        "height": int(raster.shape[0]),
        "valid_cells": int(valid.sum()),
        "missing_cells": int((~valid).sum()),
        "coverage_percent": float(valid.mean() * 100.0),
        "source_cell_counts": source_counts,
        "inputs": sorted(
            {
                _relative_input(block.source_path, input_dir)
                for block in blocks
            }
        ),
        "crs": "EPSG:4326",
        "nodata": HGT_NO_DATA if point_grid else NO_DATA,
    }


def build_gsi_dem(
    options: GSIOptions,
    progress: Optional[ProgressCallback] = None,
    cancel_event=None,
) -> GSIBuildResult:
    options.input_dir = Path(options.input_dir).resolve()
    options.output_dir = Path(options.output_dir).resolve()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    regions = regions_from_options(options)
    if not regions and not options.hgt_tiles:
        raise GSIError("Specify at least one --mesh-code, --bbox, or --hgt-tile")
    try:
        scan = scan_gsi_input(
            options.input_dir,
            write_catalog=True,
            progress=progress,
            cancel_event=cancel_event,
        )
    except GSICancelled:
        return GSIBuildResult(cancelled=True)
    if scan.modified or scan.missing:
        changed = scan.modified + scan.missing
        raise GSIError(
            f"GSI catalog integrity check failed for {changed} input archive(s); "
            "run scan after restoring or re-importing the original files."
        )
    if not scan.ready:
        raise GSIError(f"No ready GSI ZIP files found in {options.input_dir}")

    result = GSIBuildResult()
    manifest_results: list[dict] = []
    output_paths: list[Path] = []

    jobs: list[tuple[GSIRegion, bool, Optional[str]]] = [
        (region, False, None) for region in regions
    ]
    jobs.extend((_hgt_region(tile), True, "1arcsec") for tile in options.hgt_tiles)
    for index, (region, point_grid, forced_resolution) in enumerate(jobs, 1):
        _check_cancel(cancel_event)
        try:
            entries = _candidate_entries(scan, region)
            path, metadata = _build_region(
                region,
                entries,
                options.input_dir,
                options,
                progress,
                cancel_event,
                target_resolution=forced_resolution,
                point_grid=point_grid,
            )
            output_paths.append(path)
            result.outputs.append(str(path))
            manifest_results.append(metadata)
            _report(progress, "complete", index, len(jobs), str(path))
        except GSICancelled:
            result.cancelled = True
            break
        except Exception as exc:
            result.failures.append({"region": region.label, "error": str(exc)})

    tif_outputs = [path for path in output_paths if path.suffix.lower() == ".tif"]
    if options.make_vrt and tif_outputs:
        try:
            vrt_path = options.output_dir / "gsi_dem.vrt"
            _build_vrt(vrt_path, tif_outputs, options.overwrite)
            result.vrt = str(vrt_path)
        except Exception as exc:
            result.failures.append({"stage": "vrt", "error": str(exc)})

    manifest_path = options.output_dir / "gsi_dem_manifest.json"
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_dir": str(options.input_dir),
        "output_dir": str(options.output_dir),
        "options": {
            "mesh_codes": list(options.mesh_codes),
            "bbox": list(options.bbox) if options.bbox else None,
            "resolution": options.resolution,
            "make_vrt": options.make_vrt,
            "hgt_tiles": list(options.hgt_tiles),
            "source_crs": options.source_crs,
        },
        "results": manifest_results,
        "outputs": result.outputs,
        "vrt": result.vrt,
        "failures": result.failures,
        "cancelled": result.cancelled,
    }
    try:
        _write_manifest(manifest_path, manifest, options.overwrite)
        result.manifest = str(manifest_path)
    except Exception as exc:
        result.failures.append({"stage": "manifest", "error": str(exc)})

    if result.vrt:
        result.recommended_custom_dem = result.vrt
    elif len(result.outputs) == 1 and result.outputs[0].lower().endswith(".tif"):
        result.recommended_custom_dem = result.outputs[0]
    return result
