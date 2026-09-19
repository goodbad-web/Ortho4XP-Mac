"""GSI numerical elevation model import, cataloguing, and conversion.

The GSI download service distributes DEM data as ZIP archives containing
JPGIS/GML files.  This module deliberately keeps the archive management and
the raster conversion in one place so the command-line and Tk interfaces use
identical rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
STORAGE_FORMATS = ("compact_int16", "float32_legacy")
DEFAULT_STORAGE_FORMAT = "compact_int16"
COMPACT_SCALE = 0.25
COMPACT_OFFSET = 0.0
COMPACT_NO_DATA = -32768
CATALOG_VERSION = 1
MANIFEST_VERSION = 2
GSI_CACHE_VERSION = 1
GSI_CACHE_DIRNAME = ".gsi_cache"
GSI_MAX_ARCHIVE_WORKERS = 10
GSI_MAX_SCAN_WORKERS = 8
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
    storage_format: str = DEFAULT_STORAGE_FORMAT


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


def _is_auxiliary_xml_member(name: str) -> bool:
    """Ignore macOS archive metadata, not actual GSI data XML."""
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    return "__MACOSX" in parts or bool(parts and parts[-1].startswith("._"))


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


def _validate_metadata_payload(metadata: dict) -> None:
    """Validate the data payload without retaining a second raster array."""
    tuple_element = next(iter(_children(metadata["root"], "tupleList")), None)
    if tuple_element is None or not tuple_element.text:
        raise GSIError("GSI XML has no tupleList")

    value_count = 0
    for raw_line in tuple_element.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        value_count += 1
        if "," in line:
            kind, value_text = line.rsplit(",", 1)
        else:
            kind, value_text = "", line
        try:
            float(value_text.strip())
        except ValueError as exc:
            if kind.strip() not in ("データなし", "データ無し") and (
                value_text.strip() not in ("データなし", "データ無し")
            ):
                raise GSIError(f"GSI XML has an invalid elevation tuple: {line}") from exc
    if value_count == 0:
        raise GSIError("GSI XML has an empty tupleList")

    expected = metadata["width"] * metadata["height"]
    start = metadata["start_y"] * metadata["width"] + metadata["start_x"]
    if start < 0 or start >= expected:
        raise GSIError("GSI XML startPoint is outside the grid")
    if value_count > expected - start:
        raise GSIError("GSI XML tupleList exceeds the declared grid")


def _parse_values(metadata: dict, validate: bool = False) -> np.ndarray:
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
        except ValueError as exc:
            if validate and kind.strip() not in ("データなし", "データ無し") and (
                value_text.strip() not in ("データなし", "データ無し")
            ):
                raise GSIError(
                    f"GSI XML has an invalid elevation tuple: {line}"
                ) from exc
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
    if validate and len(values) > expected - start:
        raise GSIError("GSI XML tupleList exceeds the declared grid")
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
    metadata: Optional[dict] = None,
    values: Optional[np.ndarray] = None,
) -> GSIBlock:
    metadata = metadata or _parse_metadata(xml_bytes, source_path.name)
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
        values=values if values is not None else _parse_values(metadata),
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
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".xml") and not _is_auxiliary_xml_member(name)
        ]
        for name in names:
            xml_count += 1
            metadata = _parse_metadata(archive.read(name), path.name)
            if metadata is None:
                raise GSIError(f"ZIP contains unsupported or incomplete XML: {name}")
            _validate_metadata_payload(metadata)
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
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != CATALOG_VERSION
        or not isinstance(entries, list)
        or any(not isinstance(entry, dict) for entry in entries)
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
    stat_result=None,
) -> dict:
    stat_result = stat_result or path.stat()
    entry = {
        "path": str(path.relative_to(input_dir)),
        "sha256": digest,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
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


def _catalog_entry_reusable(previous: Optional[dict], stat_result) -> bool:
    """Return whether an incremental scan can trust a previous entry.

    The normal catalog path has an exact nanosecond mtime.  Older catalogs did
    not record it, so use their per-entry checked_at timestamp as a one-way
    compatibility fallback.  This function is used only by the GUI's
    incremental scan; the CLI scan remains strict and always rereads bytes.
    """
    if not previous or not previous.get("sha256"):
        return False
    if previous.get("size") != stat_result.st_size:
        return False
    previous_mtime = previous.get("mtime_ns")
    if previous_mtime is not None:
        try:
            return int(previous_mtime) == stat_result.st_mtime_ns
        except (TypeError, ValueError):
            return False
    checked_at = previous.get("checked_at")
    if not checked_at:
        return False
    try:
        checked_ns = int(datetime.fromisoformat(str(checked_at)).timestamp() * 1e9)
    except (TypeError, ValueError, OverflowError):
        return False
    return stat_result.st_mtime_ns <= checked_ns


def _scan_archive_worker(path_string: str) -> dict:
    """Hash and validate one archive in an isolated process."""
    path = Path(path_string)
    stat_result = path.stat()
    digest = None
    try:
        digest = _sha256(path)
        metadata = _archive_metadata(path)
        return {
            "ok": True,
            "digest": digest,
            "metadata": metadata,
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }
    except Exception as exc:
        return {
            "ok": False,
            "digest": digest,
            "error": str(exc),
            "size": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        }


def _scan_archive_records(
    pending: list[tuple[Path, object]],
    cancel_event,
    progress: Optional[ProgressCallback] = None,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
) -> dict[str, dict]:
    """Scan uncached archives concurrently while retaining serial fallback."""
    if not pending:
        return {}

    total = progress_total if progress_total is not None else progress_offset + len(pending)
    records: dict[str, dict] = {}

    def store_record(path: Path, record: dict) -> None:
        records[str(path)] = record
        _report(
            progress,
            "scan",
            progress_offset + len(records),
            total,
            path.name,
        )

    def serial_records() -> dict[str, dict]:
        for path, _stat_result in pending:
            _check_cancel(cancel_event)
            if str(path) in records:
                continue
            store_record(path, _scan_archive_worker(str(path)))
        return records

    worker_count = _gsi_scan_worker_count(len(pending))
    if worker_count <= 1:
        return serial_records()

    executor = None
    futures = {}
    try:
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
        futures = {
            executor.submit(_scan_archive_worker, str(path)): path
            for path, _stat_result in pending
        }
        for future in as_completed(futures):
            _check_cancel(cancel_event)
            path = futures[future]
            store_record(path, future.result())
        executor.shutdown(wait=True)
        executor = None
        return records
    except GSICancelled:
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    except (BrokenProcessPool, OSError, RuntimeError, ValueError):
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        return serial_records()
    except Exception:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise


def _apply_catalog_integrity(
    entry: dict, previous: Optional[dict], digest: Optional[str], size: int
) -> bool:
    """Keep modified entries blocked until their expected bytes return."""
    if not previous or not digest:
        return False

    previous_status = previous.get("status")
    previous_digest = previous.get("sha256")
    previous_size = previous.get("size")
    expected_digest = previous.get("expected_sha256") or previous_digest
    expected_size = previous.get("expected_size")
    if expected_size is None:
        expected_size = previous_size

    if previous_status == "modified":
        if expected_digest == digest and (
            expected_size is None or expected_size == size
        ):
            entry.pop("expected_sha256", None)
            entry.pop("expected_size", None)
            return False
        entry["status"] = "modified"
        entry["reason"] = "file changed since the previous catalog scan"
        if expected_digest:
            entry["expected_sha256"] = expected_digest
        if expected_size is not None:
            entry["expected_size"] = expected_size
        return True

    if previous_status == "missing":
        if expected_digest == digest and (
            expected_size is None or expected_size == size
        ):
            entry.pop("expected_sha256", None)
            entry.pop("expected_size", None)
            return False
        entry["status"] = "modified"
        entry["reason"] = "file changed since the previous catalog scan"
        if previous_digest:
            entry["expected_sha256"] = previous_digest
        if previous_size is not None:
            entry["expected_size"] = previous_size
        return True

    if previous_status in {"ready", "duplicate"} and (
        previous_digest != digest or previous_size != size
    ):
        entry["status"] = "modified"
        entry["reason"] = "file changed since the previous catalog scan"
        if previous_digest:
            entry["expected_sha256"] = previous_digest
        if previous_size is not None:
            entry["expected_size"] = previous_size
        return True
    return False


def scan_gsi_input(
    input_dir: Path,
    write_catalog: bool = True,
    progress: Optional[ProgressCallback] = None,
    cancel_event=None,
    source_paths: Optional[dict[str, str]] = None,
    quarantine_reasons: Optional[dict[str, str]] = None,
    incremental: bool = False,
) -> GSIScanResult:
    input_dir = Path(input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = input_dir / "catalog.json"
    try:
        previous_catalog = load_catalog(catalog_path)
    except GSIError:
        previous_catalog = {"entries": []}
    previous_entries = {
        str(entry.get("path")): entry
        for entry in previous_catalog.get("entries", [])
        if entry.get("path")
    }
    source_paths = source_paths or {}
    quarantine_reasons = quarantine_reasons or {}
    paths = _iter_archives(input_dir)
    entries: list[dict] = []
    seen_hashes: dict[str, str] = {}
    invalid = 0
    cached_entries: dict[str, dict] = {}
    pending: list[tuple[Path, object]] = []

    for index, path in enumerate(paths, 1):
        _check_cancel(cancel_event)
        stat_result = path.stat()
        relative = str(path.relative_to(input_dir))
        previous = previous_entries.get(relative)
        if incremental and _catalog_entry_reusable(previous, stat_result):
            entry = dict(previous)
            entry["size"] = stat_result.st_size
            entry["mtime_ns"] = stat_result.st_mtime_ns
            cached_entries[relative] = entry
        else:
            pending.append((path, stat_result))

    cached_count = len(cached_entries)
    if cached_count:
        _report(progress, "scan", cached_count, len(paths), "cached archives")
    archive_records = _scan_archive_records(
        pending,
        cancel_event,
        progress=progress,
        progress_offset=cached_count,
        progress_total=len(paths),
    )

    for index, path in enumerate(paths, 1):
        _check_cancel(cancel_event)
        stat_result = path.stat()
        digest = None
        relative = str(path.relative_to(input_dir))
        previous = previous_entries.get(relative)
        changed = False
        if relative in cached_entries:
            entry = cached_entries[relative]
            digest = entry.get("sha256")
            if entry.get("status") == "invalid":
                invalid += 1
            if entry.get("status") in {"ready", "duplicate"}:
                if digest in seen_hashes:
                    entry["status"] = "duplicate"
                    entry["reason"] = f"same content as {seen_hashes[digest]}"
                else:
                    seen_hashes[digest] = relative
                    entry["status"] = "ready"
                    entry.pop("reason", None)
            entries.append(entry)
            continue
        try:
            record = archive_records[str(path)]
            digest = record.get("digest")
            changed = bool(
                previous
                and (
                    previous.get("sha256") != digest
                    or previous.get("size") != stat_result.st_size
                )
            )
            if not record.get("ok"):
                raise GSIError(record.get("error") or "archive scan failed")
            metadata = record["metadata"]
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
                    stat_result=stat_result,
                )
            else:
                seen_hashes[digest] = relative
                entry = _catalog_entry(
                    path,
                    input_dir,
                    digest,
                    metadata,
                    source_path=Path(source_path) if source_path else None,
                    stat_result=stat_result,
                )
            _apply_catalog_integrity(entry, previous, digest, stat_result.st_size)
        except Exception as exc:
            if not previous or not changed:
                invalid += 1
            entry = {
                "path": relative,
                "sha256": digest,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
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
            _apply_catalog_integrity(entry, previous, digest, entry["size"])
        entries.append(entry)

    current_paths = {entry["path"] for entry in entries}
    for relative, previous in previous_entries.items():
        if relative in current_paths or str(relative).startswith("_quarantine/"):
            continue
        if previous.get("status") not in {"ready", "duplicate", "modified", "missing"}:
            continue
        missing_entry = {
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
        if previous.get("expected_sha256"):
            missing_entry["expected_sha256"] = previous["expected_sha256"]
        if previous.get("expected_size") is not None:
            missing_entry["expected_size"] = previous["expected_size"]
        entries.append(missing_entry)

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
        stat_result = path.stat()
        entries.append(
            {
                "path": relative,
                "sha256": digest,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
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


def _catalog_archive_state(
    entries: Iterable[dict], input_dir: Path, digest: str
) -> tuple[str, Optional[Path]]:
    """Return whether a digest exists, or which catalog path can be repaired."""
    repair_path: Optional[Path] = None
    for entry in entries:
        expected_digest = entry.get("expected_sha256") or entry.get("sha256")
        if expected_digest != digest or not entry.get("path"):
            continue
        managed_path = input_dir / str(entry["path"])
        if "_quarantine" in managed_path.relative_to(input_dir).parts:
            if managed_path.is_file():
                try:
                    if _sha256(managed_path) == digest:
                        return "existing", managed_path
                except OSError:
                    pass
            continue
        if managed_path.is_file():
            try:
                if _sha256(managed_path) == digest:
                    return "existing", managed_path
            except OSError:
                pass
            repair_path = repair_path or managed_path
        else:
            repair_path = repair_path or managed_path
    if repair_path is not None:
        return "repair", repair_path
    return "new", None


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


def _restore_catalog_archive(
    source: Path, destination: Path, input_dir: Path, digest: str
) -> Optional[Path]:
    """Restore a catalog path while preserving changed managed bytes."""
    preserved_path: Optional[Path] = None
    if destination.exists():
        current_digest = _sha256(destination)
        if current_digest == digest:
            return None
        quarantine_dir = input_dir / "_quarantine"
        preserved_path = _quarantine_destination(
            quarantine_dir, destination.name, current_digest
        )
        if not preserved_path.exists():
            _copy_atomic(destination, preserved_path)
    _copy_atomic(source, destination)
    return preserved_path


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
    catalog_entries = load_catalog(input_dir / "catalog.json").get("entries", [])
    registered_hashes: set[str] = set()

    try:
        for index, source_path in enumerate(paths, 1):
            _check_cancel(cancel_event)
            _report(progress, "import", index, len(paths), source_path.name)
            digest = _sha256(source_path)
            if digest in registered_hashes:
                result.skipped_duplicates += 1
                continue
            archive_state, repair_path = _catalog_archive_state(
                catalog_entries, input_dir, digest
            )
            if archive_state == "existing":
                result.skipped_duplicates += 1
                continue
            try:
                metadata = _archive_metadata(source_path)
                if archive_state == "repair" and repair_path is not None:
                    preserved_path = _restore_catalog_archive(
                        source_path, repair_path, input_dir, digest
                    )
                    if preserved_path is not None:
                        preserved_relative = str(
                            preserved_path.relative_to(input_dir)
                        )
                        source_paths[preserved_relative] = str(repair_path)
                        quarantine_reasons[preserved_relative] = (
                            "previous managed file preserved during re-import"
                        )
                    relative = str(repair_path.relative_to(input_dir))
                    source_paths[relative] = str(source_path)
                    registered_hashes.add(digest)
                    result.imported += 1
                    continue
                destination_dir = input_dir / metadata["product"] / metadata["date"]
                destination = _unique_destination(
                    destination_dir, source_path.name, digest
                )
                if not destination.exists():
                    _copy_atomic(source_path, destination)
                registered_hashes.add(digest)
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

    try:
        scan = scan_gsi_input(
            input_dir,
            write_catalog=True,
            progress=progress,
            cancel_event=cancel_event,
            source_paths=source_paths,
            quarantine_reasons=quarantine_reasons,
        )
    except GSICancelled:
        result.cancelled = True
        result.catalog_path = str(input_dir / "catalog.json")
        return result
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


def _catalog_mesh_bounds(mesh_code: str) -> Optional[tuple[float, float, float, float]]:
    code = str(mesh_code).strip()
    if len(code) >= 8 and code[:8].isdigit():
        return mesh_code_bounds(code[:8])
    if len(code) == 6 and code.isdigit():
        first_lat = int(code[0:2]) / 1.5
        first_lon = 100.0 + int(code[2:4])
        south = first_lat + int(code[4]) * 5.0 / 60.0
        west = first_lon + int(code[5]) * 7.5 / 60.0
        return (south, west, south + 5.0 / 60.0, west + 7.5 / 60.0)
    if len(code) == 4 and code.isdigit():
        south = int(code[0:2]) / 1.5
        west = 100.0 + int(code[2:4])
        return (south, west, south + 40.0 / 60.0, west + 1.0)
    return None


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
    if region.mesh_code:
        requested = region.mesh_code
        for value in entry.get("mesh_codes", []):
            value = str(value)
            if requested.startswith(value) or value.startswith(requested[:6]):
                return True
        return False

    mesh_codes = entry.get("mesh_codes", [])
    if not mesh_codes:
        return True
    for value in mesh_codes:
        mesh_bounds = _catalog_mesh_bounds(value)
        if mesh_bounds is None:
            # An unknown catalog code must not cause valid input to be omitted.
            return True
        if _overlaps(mesh_bounds, region.bounds):
            return True
    return False


def _metadata_summary(metadata: dict) -> dict:
    """Keep only small, XML-tree-independent metadata for a build cache."""
    return {
        key: metadata[key]
        for key in (
            "product",
            "date",
            "mesh_codes",
            "south",
            "west",
            "north",
            "east",
            "width",
            "height",
            "start_x",
            "start_y",
            "srs_name",
        )
    }


def _metadata_bounds_in_wgs84(
    metadata: dict, source_crs: Optional[str]
) -> tuple[float, float, float, float]:
    """Return cached metadata bounds in the same CRS as GSIRegion."""
    transformed_cache = metadata.setdefault("_bounds_wgs84", {})
    cache_key = source_crs or ""
    if cache_key not in transformed_cache:
        source_ref, _ = _resolve_source_crs(metadata["srs_name"], source_crs)
        transformed_cache[cache_key] = _transform_bounds(
            (
                metadata["south"],
                metadata["west"],
                metadata["north"],
                metadata["east"],
            ),
            source_ref,
        )
    return transformed_cache[cache_key]


def _gsi_cache_root(input_dir: Path) -> Path:
    return Path(input_dir) / GSI_CACHE_DIRNAME / f"v{GSI_CACHE_VERSION}"


def _archive_cache_dir(cache_root: Path, archive_digest: str) -> Path:
    return Path(cache_root) / "archives" / archive_digest


def _archive_cache_index_path(cache_root: Path, archive_digest: str) -> Path:
    return _archive_cache_dir(cache_root, archive_digest) / "index.json"


def _cache_member_payload_name(xml_name: str) -> str:
    return hashlib.sha256(xml_name.encode("utf-8")).hexdigest() + ".npy"


def _zip_xml_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [
        info
        for info in archive.infolist()
        if info.filename.lower().endswith(".xml")
        and not _is_auxiliary_xml_member(info.filename)
    ]


def _load_archive_cache(
    cache_root: Optional[Path],
    archive_digest: Optional[str],
    infos: Iterable[zipfile.ZipInfo],
) -> Optional[dict[str, dict]]:
    if not cache_root or not archive_digest:
        return None
    try:
        payload = json.loads(
            _archive_cache_index_path(cache_root, archive_digest).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != GSI_CACHE_VERSION
        or payload.get("archive_sha256") != archive_digest
        or not isinstance(payload.get("members"), dict)
    ):
        return None
    expected_names = {info.filename for info in infos}
    members = payload["members"]
    if set(members) != expected_names:
        return None
    for info in infos:
        record = members.get(info.filename)
        if not isinstance(record, dict):
            return None
        try:
            valid_fingerprint = (
                int(record.get("crc", -1)) == int(info.CRC)
                and int(record.get("file_size", -1)) == int(info.file_size)
                and int(record.get("compress_size", -1)) == int(info.compress_size)
            )
        except (TypeError, ValueError):
            valid_fingerprint = False
        if not valid_fingerprint or not isinstance(record.get("metadata"), dict):
            return None
    return members


def _atomic_npy_write(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        with Path(temp_name).open("wb") as stream:
            np.save(stream, np.asarray(values, dtype=np.float32), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _load_cached_values(
    cache_root: Optional[Path], archive_digest: Optional[str], record: dict
) -> Optional[np.ndarray]:
    if not cache_root or not archive_digest:
        return None
    payload_name = record.get("payload")
    metadata = record.get("metadata")
    if not isinstance(payload_name, str) or not isinstance(metadata, dict):
        return None
    payload_path = _archive_cache_dir(cache_root, archive_digest) / payload_name
    try:
        values = np.load(payload_path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    try:
        expected_shape = (
            int(metadata.get("height", -1)),
            int(metadata.get("width", -1)),
        )
    except (TypeError, ValueError):
        return None
    if values.dtype != np.float32 or values.shape != expected_shape:
        return None
    return np.asarray(values, dtype=np.float32, order="C")


def _write_archive_cache(
    cache_root: Optional[Path],
    archive_digest: Optional[str],
    members: dict[str, dict],
) -> None:
    if not cache_root or not archive_digest:
        return
    _atomic_json_write(
        _archive_cache_index_path(cache_root, archive_digest),
        {
            "version": GSI_CACHE_VERSION,
            "archive_sha256": archive_digest,
            "members": members,
        },
    )


def _archive_block_from_values(
    path: Path,
    xml_name: str,
    metadata: dict,
    values: np.ndarray,
    source_crs: Optional[str],
) -> GSIBlock:
    return parse_gsi_xml(
        b"",
        path,
        xml_name,
        source_crs,
        metadata=metadata,
        values=values,
    )


def _archive_blocks(
    path: Path,
    region: GSIRegion,
    source_crs: Optional[str],
    cancel_event=None,
    metadata_cache: Optional[dict[str, tuple[tuple[str, dict], ...]]] = None,
    payload_cache_root: Optional[Path] = None,
    archive_digest: Optional[str] = None,
) -> list[GSIBlock]:
    return _archive_blocks_fast(
        path,
        region,
        source_crs,
        cancel_event=cancel_event,
        metadata_cache=metadata_cache,
        payload_cache_root=payload_cache_root,
        archive_digest=archive_digest,
    )


def _archive_blocks_fast(
    path: Path,
    region: GSIRegion,
    source_crs: Optional[str],
    cancel_event=None,
    metadata_cache: Optional[dict[str, tuple[tuple[str, dict], ...]]] = None,
    payload_cache_root: Optional[Path] = None,
    archive_digest: Optional[str] = None,
) -> list[GSIBlock]:
    blocks: list[GSIBlock] = []
    cache_key = str(Path(path).resolve())
    indexed_members = metadata_cache.get(cache_key) if metadata_cache is not None else None

    with zipfile.ZipFile(path) as archive:
        infos = _zip_xml_infos(archive)
        cache_members = _load_archive_cache(
            payload_cache_root, archive_digest, infos
        )

        if indexed_members is None and cache_members is not None:
            indexed_members = tuple(
                (info.filename, cache_members[info.filename]["metadata"])
                for info in infos
            )

        if indexed_members is None:
            indexed: list[tuple[str, dict]] = []
            cache_records: dict[str, dict] = {}
            for info in infos:
                _check_cancel(cancel_event)
                xml_name = info.filename
                xml_bytes = archive.read(info)
                metadata = _parse_metadata(xml_bytes, path.name)
                if metadata is None:
                    raise GSIError(
                        f"ZIP contains unsupported or incomplete XML: {xml_name}"
                    )
                summary = _metadata_summary(metadata)
                indexed.append((xml_name, summary))
                record = {
                    "crc": int(info.CRC),
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "metadata": summary,
                }
                source_bounds = _metadata_bounds_in_wgs84(summary, source_crs)
                if _overlaps(source_bounds, region.bounds):
                    values = _parse_values(metadata, validate=True)
                    if payload_cache_root and archive_digest:
                        payload_name = f"values/{_cache_member_payload_name(xml_name)}"
                        _atomic_npy_write(
                            _archive_cache_dir(payload_cache_root, archive_digest)
                            / payload_name,
                            values,
                        )
                        record["payload"] = payload_name
                    block = _archive_block_from_values(
                        path, xml_name, summary, values, source_crs
                    )
                    if _overlaps(block.bounds, region.bounds):
                        blocks.append(block)
                cache_records[xml_name] = record
            indexed_members = tuple(indexed)
            if metadata_cache is not None:
                metadata_cache[cache_key] = indexed_members
            _write_archive_cache(payload_cache_root, archive_digest, cache_records)
            return blocks

        if metadata_cache is not None:
            metadata_cache[cache_key] = indexed_members

        for xml_name, summary in indexed_members:
            _check_cancel(cancel_event)
            source_bounds = _metadata_bounds_in_wgs84(summary, source_crs)
            if not _overlaps(source_bounds, region.bounds):
                continue
            record = cache_members.get(xml_name) if cache_members else None
            values = _load_cached_values(
                payload_cache_root, archive_digest, record or {}
            )
            if values is None:
                xml_bytes = archive.read(xml_name)
                metadata = _parse_metadata(xml_bytes, path.name)
                if metadata is None:
                    raise GSIError(
                        f"Unsupported or incomplete GSI XML: {xml_name}"
                    )
                values = _parse_values(metadata, validate=True)
                summary = _metadata_summary(metadata)
                if record is not None and payload_cache_root and archive_digest:
                    payload_name = f"values/{_cache_member_payload_name(xml_name)}"
                    _atomic_npy_write(
                        _archive_cache_dir(payload_cache_root, archive_digest)
                        / payload_name,
                        values,
                    )
                    record["payload"] = payload_name
                    _write_archive_cache(
                        payload_cache_root, archive_digest, cache_members
                    )
            block = _archive_block_from_values(
                path, xml_name, summary, values, source_crs
            )
            if _overlaps(block.bounds, region.bounds):
                blocks.append(block)
    return blocks


def _archive_blocks_worker(
    path: str,
    region: GSIRegion,
    source_crs: Optional[str],
    payload_cache_root: Optional[str],
    archive_digest: Optional[str],
) -> list[GSIBlock]:
    return _archive_blocks_fast(
        Path(path),
        region,
        source_crs,
        payload_cache_root=Path(payload_cache_root) if payload_cache_root else None,
        archive_digest=archive_digest,
    )


def _candidate_entries(scan: GSIScanResult, region: GSIRegion) -> list[dict]:
    return [
        entry
        for entry in scan.entries
        if entry.get("status") == "ready" and _record_may_contain(entry, region)
    ]


def _scan_result_from_catalog(input_dir: Path) -> GSIScanResult:
    catalog_path = Path(input_dir) / "catalog.json"
    payload = load_catalog(catalog_path)
    entries = list(payload.get("entries", []))
    return GSIScanResult(
        entries=entries,
        ready=sum(entry.get("status") == "ready" for entry in entries),
        duplicates=sum(entry.get("status") == "duplicate" for entry in entries),
        quarantined=sum(entry.get("status") == "quarantined" for entry in entries),
        invalid=sum(entry.get("status") == "invalid" for entry in entries),
        modified=sum(entry.get("status") == "modified" for entry in entries),
        missing=sum(entry.get("status") == "missing" for entry in entries),
        catalog_path=str(catalog_path),
        zip_count=len(entries),
        product_counts=_count_field(entries, "product", {"ready", "duplicate"}),
        date_counts=_count_field(entries, "date", {"ready", "duplicate"}),
        selection_counts=_count_field(entries, "selection"),
    )


def _catalog_archive_paths(entries: Iterable[dict]) -> set[str]:
    return {
        str(entry.get("path"))
        for entry in entries
        if entry.get("path") and "_quarantine" not in Path(str(entry["path"])).parts
    }


def _catalog_matches_input(input_dir: Path, entries: Iterable[dict]) -> bool:
    actual = {
        str(path.relative_to(input_dir))
        for path in _iter_archives(input_dir)
    }
    return actual == _catalog_archive_paths(entries)


def _load_build_scan(
    input_dir: Path,
    progress: Optional[ProgressCallback],
    cancel_event,
) -> GSIScanResult:
    """Load a catalog cheaply, falling back to the strict scan when needed."""
    catalog_path = Path(input_dir) / "catalog.json"
    if not catalog_path.is_file():
        return scan_gsi_input(
            input_dir,
            write_catalog=True,
            progress=progress,
            cancel_event=cancel_event,
        )
    try:
        scan = _scan_result_from_catalog(input_dir)
        if not _catalog_matches_input(Path(input_dir), scan.entries):
            return scan_gsi_input(
                input_dir,
                write_catalog=True,
                progress=progress,
                cancel_event=cancel_event,
            )
        return scan
    except GSICancelled:
        raise
    except GSIError:
        return scan_gsi_input(
            input_dir,
            write_catalog=True,
            progress=progress,
            cancel_event=cancel_event,
        )


def _verify_build_entries(
    entries: Iterable[dict],
    input_dir: Path,
    progress: Optional[ProgressCallback],
    cancel_event,
) -> None:
    unique_entries = {
        str(entry["path"]): entry
        for entry in entries
        if entry.get("path")
    }
    total = len(unique_entries)
    for index, entry in enumerate(unique_entries.values(), 1):
        _check_cancel(cancel_event)
        relative = str(entry["path"])
        path = Path(input_dir) / relative
        if entry.get("status") != "ready":
            raise GSIError(
                f"Catalog input is not ready for build: {relative} "
                f"({entry.get('status')})"
            )
        if not path.is_file():
            raise GSIError(f"Catalog input is missing: {path}")
        size = path.stat().st_size
        expected_size = entry.get("size")
        if expected_size is not None and int(expected_size) != size:
            raise GSIError(f"Catalog input size changed: {path}")
        _report(progress, "verify", index, total, path.name)
        digest = _sha256(path)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise GSIError(f"Catalog input digest changed: {path}")


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

    if not point_grid and np.isclose(dlon, resolution_deg) and np.isclose(
        dlat, resolution_deg
    ):
        source_col = (source_west - region.west) / resolution_deg
        source_row = (region.north - source_north) / resolution_deg
        source_col0 = int(round(source_col))
        source_row0 = int(round(source_row))
        if np.isclose(source_col, source_col0) and np.isclose(
            source_row, source_row0
        ):
            output_col0 = max(0, source_col0)
            output_col1 = min(width, source_col0 + src_width)
            output_row0 = max(0, source_row0)
            output_row1 = min(height, source_row0 + src_height)
            if output_col1 <= output_col0 or output_row1 <= output_row0:
                return 0
            source_col_slice = slice(
                output_col0 - source_col0,
                output_col1 - source_col0,
            )
            source_row_slice = slice(
                output_row0 - source_row0,
                output_row1 - source_row0,
            )
            sampled = block.values[source_row_slice, source_col_slice]
            good = np.isfinite(sampled)
            if not np.any(good):
                return 0
            destination = output[output_row0:output_row1, output_col0:output_col1]
            destination[good] = sampled[good]
            output[output_row0:output_row1, output_col0:output_col1] = destination
            return int(np.count_nonzero(good))

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
    storage_format: str = DEFAULT_STORAGE_FORMAT,
) -> dict:
    _require_gdal()
    if storage_format not in STORAGE_FORMATS:
        raise GSIError(f"Unsupported GSI storage format: {storage_format}")

    compact = storage_format == "compact_int16"
    valid = np.isfinite(raster)
    if compact:
        encoded = np.rint((raster[valid] - COMPACT_OFFSET) / COMPACT_SCALE)
        if encoded.size and (
            encoded.min() < np.iinfo(np.int16).min + 1
            or encoded.max() > np.iinfo(np.int16).max
        ):
            minimum = float(raster[valid].min())
            maximum = float(raster[valid].max())
            raise GSIError(
                "DEM values exceed compact_int16 range "
                f"[-8191.75, 8191.75] m: min={minimum}, max={maximum}"
            )
        data_type = gdal.GDT_Int16
        nodata = COMPACT_NO_DATA
        predictor = 2
        compression_candidates = ("ZSTD", "DEFLATE")
    else:
        data_type = gdal.GDT_Float32
        nodata = NO_DATA
        predictor = 3
        compression_candidates = ("DEFLATE",)

    path = Path(path)
    temp_path = _atomic_output_path(path, overwrite)
    if temp_path.exists():
        temp_path.unlink()
    fallback_reason = None
    try:
        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise GSIError("GDAL GTiff driver is unavailable")
        for compression in compression_candidates:
            try:
                options = [
                    "TILED=YES",
                    "BLOCKXSIZE=512",
                    "BLOCKYSIZE=512",
                    f"COMPRESS={compression}",
                    f"PREDICTOR={predictor}",
                    "ZLEVEL=6",
                    "ZSTD_LEVEL=9",
                    "BIGTIFF=IF_SAFER",
                    "NUM_THREADS=ALL_CPUS",
                ]
                dataset = driver.Create(
                    str(temp_path),
                    raster.shape[1],
                    raster.shape[0],
                    1,
                    data_type,
                    options=options,
                )
                if dataset is None:
                    raise GSIError(f"Could not create GeoTIFF: {temp_path}")
                dataset.SetGeoTransform(
                    (region.west, resolution_deg, 0.0, region.north, 0.0, -resolution_deg)
                )
                spatial_ref = _spatial_reference("EPSG:4326")
                dataset.SetProjection(spatial_ref.ExportToWkt())
                band = dataset.GetRasterBand(1)
                band.SetNoDataValue(nodata)
                if compact:
                    band.SetScale(COMPACT_SCALE)
                    band.SetOffset(COMPACT_OFFSET)
                for row_start in range(0, raster.shape[0], 512):
                    row_end = min(raster.shape[0], row_start + 512)
                    if compact:
                        values = np.full(
                            (row_end - row_start, raster.shape[1]),
                            COMPACT_NO_DATA,
                            dtype=np.int16,
                        )
                        rows = raster[row_start:row_end]
                        row_valid = np.isfinite(rows)
                        values[row_valid] = np.rint(
                            (rows[row_valid] - COMPACT_OFFSET) / COMPACT_SCALE
                        ).astype(np.int16)
                    else:
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
                actual_compression = check.GetMetadataItem(
                    "COMPRESSION", "IMAGE_STRUCTURE"
                )
                if actual_compression != compression:
                    check = None
                    temp_path.unlink(missing_ok=True)
                    if compression == "ZSTD":
                        fallback_reason = (
                            f"GDAL did not produce ZSTD (reported {actual_compression!r})"
                        )
                        continue
                    raise GSIError(
                        f"GeoTIFF compression verification failed: "
                        f"requested={compression}, actual={actual_compression}"
                    )
                check = None
                os.replace(temp_path, path)
                return {
                    "storage_format": storage_format,
                    "data_type": "Int16" if compact else "Float32",
                    "scale": COMPACT_SCALE if compact else 1.0,
                    "offset": COMPACT_OFFSET if compact else 0.0,
                    "nodata": nodata,
                    "compression": compression,
                    "predictor": predictor,
                    "quantization_max_error": 0.125 if compact else 0.0,
                    "compression_fallback_reason": fallback_reason,
                }
            except Exception as error:
                if temp_path.exists():
                    temp_path.unlink()
                if compression == "ZSTD" and compact:
                    fallback_reason = str(error)
                    continue
                raise
        raise GSIError("Could not create a verified compressed GeoTIFF")
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


def _raster_contract(path: Path) -> dict:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise GSIError(f"Could not open VRT source: {path}")
    band = dataset.GetRasterBand(1)
    scale = band.GetScale()
    offset = band.GetOffset()
    spatial_ref = dataset.GetSpatialRef()
    authority_name = (
        spatial_ref.GetAuthorityName(None) if spatial_ref is not None else None
    )
    authority_code = (
        spatial_ref.GetAuthorityCode(None) if spatial_ref is not None else None
    )
    crs = (
        f"{authority_name}:{authority_code}"
        if authority_name and authority_code
        else None
    )
    contract = {
        "data_type": gdal.GetDataTypeName(band.DataType),
        "scale": 1.0 if scale is None else float(scale),
        "offset": 0.0 if offset is None else float(offset),
        "nodata": band.GetNoDataValue(),
        "crs": crs,
    }
    dataset = None
    return contract


def _build_vrt(
    path: Path,
    sources: list[Path],
    overwrite: bool,
    output_bounds: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    _require_gdal()
    if not sources:
        raise GSIError("Cannot create VRT without GeoTIFF sources")
    contracts = [_raster_contract(source) for source in sources]
    first_contract = contracts[0]
    if any(contract != first_contract for contract in contracts[1:]):
        raise GSIError(
            "Cannot create VRT from GeoTIFFs with different data type, "
            "scale, offset, or NoData contracts"
        )
    if first_contract["nodata"] is None:
        raise GSIError("VRT sources must advertise a NoData value")
    temp_path = _atomic_output_path(path, overwrite)
    try:
        options = gdal.BuildVRTOptions(
            srcNodata=first_contract["nodata"],
            VRTNodata=first_contract["nodata"],
            resampleAlg="nearest",
            resolution="highest" if output_bounds else None,
            outputBounds=output_bounds,
        )
        dataset = gdal.BuildVRT(str(temp_path), [str(source) for source in sources], options=options)
        if dataset is None:
            raise GSIError(f"Could not create VRT: {path}")
        dataset.FlushCache()
        dataset = None
        check = gdal.Open(str(temp_path), gdal.GA_ReadOnly)
        if check is None:
            raise GSIError(f"Could not reopen generated VRT: {temp_path}")
        band = check.GetRasterBand(1)
        actual_contract = {
            "data_type": gdal.GetDataTypeName(band.DataType),
            "scale": 1.0 if band.GetScale() is None else float(band.GetScale()),
            "offset": 0.0 if band.GetOffset() is None else float(band.GetOffset()),
            "nodata": band.GetNoDataValue(),
            "crs": (
                f"{check.GetSpatialRef().GetAuthorityName(None)}:"
                f"{check.GetSpatialRef().GetAuthorityCode(None)}"
                if check.GetSpatialRef() is not None
                and check.GetSpatialRef().GetAuthorityName(None)
                and check.GetSpatialRef().GetAuthorityCode(None)
                else None
            ),
        }
        check = None
        if actual_contract != first_contract:
            raise GSIError(
                "Generated VRT does not preserve the common raster contract"
            )
        os.replace(temp_path, path)
        return first_contract
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


def _gsi_archive_worker_count(candidate_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    return min(
        max(1, candidate_count),
        GSI_MAX_ARCHIVE_WORKERS,
        max(1, cpu_count),
    )


def _gsi_scan_worker_count(pending_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    return min(
        max(1, pending_count),
        GSI_MAX_SCAN_WORKERS,
        max(1, cpu_count // 2),
    )


def _archive_blocks_parallel(
    entries: list[dict],
    region: GSIRegion,
    input_dir: Path,
    source_crs: Optional[str],
    payload_cache_root: Path,
    cancel_event,
    progress: Optional[ProgressCallback] = None,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
) -> list[GSIBlock]:
    total = progress_total if progress_total is not None else len(entries)
    worker_count = _gsi_archive_worker_count(len(entries))
    if worker_count <= 1:
        blocks: list[GSIBlock] = []
        for index, entry in enumerate(entries, 1):
            _check_cancel(cancel_event)
            blocks.extend(
                _archive_blocks(
                    input_dir / entry["path"],
                    region,
                    source_crs,
                    cancel_event=cancel_event,
                    payload_cache_root=payload_cache_root,
                    archive_digest=entry.get("sha256"),
                )
            )
            _report(
                progress,
                "build",
                progress_offset + index,
                total,
                f"{region.label}: {Path(entry['path']).name}",
            )
        return blocks

    context = multiprocessing.get_context("spawn")
    futures = {}
    executor = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        )
        for entry in entries:
            _check_cancel(cancel_event)
            future = executor.submit(
                _archive_blocks_worker,
                str(input_dir / entry["path"]),
                region,
                source_crs,
                str(payload_cache_root),
                entry.get("sha256"),
            )
            futures[future] = entry
        blocks = []
        for index, future in enumerate(as_completed(futures), 1):
            _check_cancel(cancel_event)
            blocks.extend(future.result())
            _report(
                progress,
                "build",
                progress_offset + index,
                total,
                f"{region.label}: {Path(futures[future]['path']).name}",
            )
        executor.shutdown(wait=True)
        executor = None
        return blocks
    except GSICancelled:
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    except BrokenProcessPool:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        # A process pool is an optimization only. Retry serially so a spawn
        # or native-runtime limitation never changes the GSI result contract.
        blocks = []
        for index, entry in enumerate(entries, 1):
            _check_cancel(cancel_event)
            blocks.extend(
                _archive_blocks(
                    input_dir / entry["path"],
                    region,
                    source_crs,
                    cancel_event=cancel_event,
                    payload_cache_root=payload_cache_root,
                    archive_digest=entry.get("sha256"),
                )
            )
            _report(
                progress,
                "build",
                progress_offset + index,
                total,
                f"{region.label}: {Path(entry['path']).name}",
            )
        return blocks
    except Exception:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise


def _build_region(
    region: GSIRegion,
    entries: list[dict],
    input_dir: Path,
    options: GSIOptions,
    progress: Optional[ProgressCallback],
    cancel_event,
    metadata_cache: Optional[dict[str, tuple[tuple[str, dict], ...]]] = None,
    payload_cache_root: Optional[Path] = None,
    target_resolution: Optional[str] = None,
    point_grid: bool = False,
    progress_offset: int = 0,
    progress_total: Optional[int] = None,
) -> tuple[Path, dict]:
    blocks: list[GSIBlock] = []
    candidate_total = len(entries)
    for entry in entries:
        archive_path = input_dir / entry["path"]
        if not archive_path.is_file():
            raise GSIError(f"Catalog input is missing: {archive_path}")
    report_total = progress_total if progress_total is not None else candidate_total
    if candidate_total > 1 and payload_cache_root is not None:
        blocks.extend(
            _archive_blocks_parallel(
                entries,
                region,
                input_dir,
                options.source_crs,
                payload_cache_root,
                cancel_event,
                progress=progress,
                progress_offset=progress_offset,
                progress_total=report_total,
            )
        )
    else:
        for index, entry in enumerate(entries, 1):
            _check_cancel(cancel_event)
            archive_path = input_dir / entry["path"]
            blocks.extend(
                _archive_blocks(
                    archive_path,
                    region,
                    options.source_crs,
                    cancel_event,
                    metadata_cache=metadata_cache,
                    payload_cache_root=payload_cache_root,
                    archive_digest=entry.get("sha256"),
                )
            )
            _report(
                progress,
                "build",
                progress_offset + index,
                report_total,
                f"{region.label}: {archive_path.name}",
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
    output_metadata = {
        "storage_format": "hgt_legacy",
        "data_type": "Int16",
        "scale": 1.0,
        "offset": 0.0,
        "nodata": HGT_NO_DATA,
        "compression": "none",
        "predictor": None,
        "quantization_max_error": 0.5,
        "compression_fallback_reason": None,
    }
    if point_grid:
        output_path = options.output_dir / "hgt" / f"{region.label}.hgt"
        write_hgt(output_path, raster, overwrite=options.overwrite)
    else:
        output_metadata = write_geotiff(
            output_path,
            raster,
            region,
            resolution_deg,
            overwrite=options.overwrite,
            storage_format=options.storage_format,
        )
    return output_path, {
        "region": asdict(region),
        "output": str(output_path),
        "resolution": resolution_name,
        "source_date": max(
            (str(block.date) for block in blocks if block.date),
            default=None,
        ),
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
        **output_metadata,
    }


def build_gsi_dem(
    options: GSIOptions,
    progress: Optional[ProgressCallback] = None,
    cancel_event=None,
) -> GSIBuildResult:
    if options.storage_format not in STORAGE_FORMATS:
        raise GSIError(f"Unsupported GSI storage format: {options.storage_format}")
    options.input_dir = Path(options.input_dir).resolve()
    options.output_dir = Path(options.output_dir).resolve()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    regions = regions_from_options(options)
    if not regions and not options.hgt_tiles:
        raise GSIError("Specify at least one --mesh-code, --bbox, or --hgt-tile")
    try:
        scan = _load_build_scan(
            options.input_dir,
            progress=progress,
            cancel_event=cancel_event,
        )
    except GSICancelled:
        return GSIBuildResult(cancelled=True)
    if not scan.ready:
        raise GSIError(f"No ready GSI ZIP files found in {options.input_dir}")

    result = GSIBuildResult()
    manifest_results: list[dict] = []
    output_paths: list[Path] = []
    vrt_contract = None

    jobs: list[tuple[GSIRegion, bool, Optional[str]]] = [
        (region, False, None) for region in regions
    ]
    jobs.extend((_hgt_region(tile), True, "1arcsec") for tile in options.hgt_tiles)
    job_entries: list[list[dict]] = []
    all_candidates: list[dict] = []
    for region, _point_grid, _forced_resolution in jobs:
        candidates = _candidate_entries(scan, region)
        job_entries.append(candidates)
        all_candidates.extend(candidates)

    verify_total = len(
        {
            str(entry["path"]): entry
            for entry in all_candidates
            if entry.get("path")
        }
    )
    archive_total = sum(len(entries) for entries in job_entries)
    overall_total = verify_total + archive_total + len(jobs) + 1

    def report_overall(completed: int, message: str) -> None:
        _report(progress, "build", completed, overall_total, message)

    def report_verify(_stage: str, completed: int, _total: int, message: str) -> None:
        report_overall(completed, f"Verify: {message}")

    try:
        _verify_build_entries(
            all_candidates,
            options.input_dir,
            report_verify,
            cancel_event,
        )
    except GSICancelled:
        return GSIBuildResult(cancelled=True)

    metadata_cache: dict[str, tuple[tuple[str, dict], ...]] = {}
    payload_cache_root = _gsi_cache_root(options.input_dir)
    overall_done = verify_total
    for index, (region, point_grid, forced_resolution) in enumerate(jobs, 1):
        _check_cancel(cancel_event)
        entries = job_entries[index - 1]
        archive_offset = overall_done

        def report_region(
            _stage: str,
            completed: int,
            _total: int,
            message: str,
        ) -> None:
            report_overall(archive_offset + completed, f"Build: {message}")

        try:
            path, metadata = _build_region(
                region,
                entries,
                options.input_dir,
                options,
                report_region,
                cancel_event,
                metadata_cache=metadata_cache,
                payload_cache_root=payload_cache_root,
                target_resolution=forced_resolution,
                point_grid=point_grid,
                progress_offset=0,
                progress_total=len(entries),
            )
            output_paths.append(path)
            result.outputs.append(str(path))
            manifest_results.append(metadata)
            overall_done = archive_offset + len(entries) + 1
            report_overall(overall_done, f"Complete: {path}")
        except GSICancelled:
            result.cancelled = True
            break
        except Exception as exc:
            result.failures.append({"region": region.label, "error": str(exc)})
            overall_done = archive_offset + len(entries) + 1
            report_overall(overall_done, f"Failed: {region.label}")

    tif_outputs = [path for path in output_paths if path.suffix.lower() == ".tif"]
    if options.make_vrt and tif_outputs:
        try:
            vrt_path = options.output_dir / "gsi_dem.vrt"
            vrt_bounds = None
            if regions:
                vrt_bounds = (
                    math.floor(min(region.west for region in regions)),
                    math.floor(min(region.south for region in regions)),
                    math.ceil(max(region.east for region in regions)),
                    math.ceil(max(region.north for region in regions)),
                )
            vrt_contract = _build_vrt(
                vrt_path,
                tif_outputs,
                options.overwrite,
                output_bounds=vrt_bounds,
            )
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
            "storage_format": options.storage_format,
        },
        "results": manifest_results,
        "outputs": result.outputs,
        "vrt": result.vrt,
        "vrt_contract": vrt_contract,
        "failures": result.failures,
        "cancelled": result.cancelled,
    }
    try:
        _write_manifest(manifest_path, manifest, options.overwrite)
        result.manifest = str(manifest_path)
    except Exception as exc:
        result.failures.append({"stage": "manifest", "error": str(exc)})

    if not result.cancelled:
        report_overall(overall_total, "Finalized")

    if result.vrt:
        result.recommended_custom_dem = result.vrt
    elif len(result.outputs) == 1 and result.outputs[0].lower().endswith(".tif"):
        result.recommended_custom_dem = result.outputs[0]
    return result
