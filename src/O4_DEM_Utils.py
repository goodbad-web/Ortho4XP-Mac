import os
import io
import json
import time
import requests
import zipfile
import itertools
import shutil
import tempfile
import uuid
import re
from pathlib import Path
from math import sqrt
import array
import numpy

try:
    from osgeo import gdal
    has_gdal = True
    gdal.UseExceptions()
except:
    has_gdal = False
from PIL import Image
import O4_UI_Utils as UI
import O4_File_Names as FNAMES


def _resident_dem_smoothing(server, raster, mask_array, kernel):
    """Try the resident ASHelper Metal raster operation for one DEM window."""
    if server is None or getattr(server, "gpu_disabled", False):
        return None
    try:
        from O4_ASHelper_Server import raster_batch

        os.makedirs(FNAMES.Tmp_dir, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix=".dem-smooth-", dir=FNAMES.Tmp_dir)
        input_path = os.path.join(workdir, "input.raw")
        mask_path = os.path.join(workdir, "mask.raw")
        output_path = os.path.join(workdir, "output.raw")
        numpy.asarray(raster, dtype=numpy.float32, order="C").tofile(input_path)
        numpy.asarray(mask_array * 255.0, dtype=numpy.uint8, order="C").tofile(mask_path)
        height, width = raster.shape
        response = raster_batch(
            server,
            "dem_smooth_batch",
            [
                {
                    "id": "dem-smooth",
                    "input": input_path,
                    "mask": mask_path,
                    "output": output_path,
                    "width": int(width),
                    "height": int(height),
                    "stride": int(width * numpy.dtype(numpy.float32).itemsize),
                    "mask_stride": int(width),
                    "kernel": [float(value) for value in kernel],
                }
            ],
        )
        result = (response.get("results") or [{}])[0]
        if not result.get("ok") or not os.path.isfile(output_path):
            return None
        output = numpy.fromfile(output_path, dtype=numpy.float32)
        if output.size != height * width:
            return None
        return output.reshape((height, width)).copy()
    except Exception as error:
        UI.vprint(2, "Metal DEM smoothing fallback due to error:", error)
        return None
    finally:
        if "workdir" in locals():
            shutil.rmtree(workdir, ignore_errors=True)

available_sources = (
    "View",
    "Viewfinderpanoramas (J. de Ferranti) - mostly worldwide",
    "SRTM",
    "SRTMv3 (from OpenTopography) - NOW REQUIRES MANUAL DOWNLOAD",
    "NED1",
    'NED 1" (from USGS) - USA, Canada, Mexico',
    "NED1/3",
    'NED 1/3" (from USGS) - USA',
    "ALOS",
    "ALOS 3W30 (from OpenTopography) - NOW REQUIRES MANUAL DOWNLOAD",
)

global_sources = ("View", "SRTM", "ALOS")
DEM_MEMORY_FRACTION = 0.8


class DEMError(RuntimeError):
    """Raised when an explicitly selected DEM cannot be used safely."""


def _physical_memory_bytes():
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0
    if pages <= 0 or page_size <= 0:
        return 0
    return pages * page_size


def _raster_memory_estimate(
    width, height, scale, offset, fill_nodata, raw_bytes_per_cell=0
):
    """Estimate the peak NumPy working set for a GDAL DEM read."""
    cells = int(width) * int(height)
    float_bytes = cells * numpy.dtype(numpy.float32).itemsize
    mask_bytes = cells
    scale_bytes = float_bytes if scale != 1.0 or offset != 0.0 else 0
    fill_bytes = 4 * float_bytes if fill_nodata else 0
    source_bytes = cells * int(raw_bytes_per_cell)
    return source_bytes + float_bytes + mask_bytes + scale_bytes + fill_bytes


def _check_raster_memory(
    width,
    height,
    scale,
    offset,
    fill_nodata,
    file_name,
    raw_bytes_per_cell=0,
):
    physical = _physical_memory_bytes()
    if not physical:
        return
    estimated = _raster_memory_estimate(
        width,
        height,
        scale,
        offset,
        fill_nodata,
        raw_bytes_per_cell=raw_bytes_per_cell,
    )
    limit = int(physical * DEM_MEMORY_FRACTION)
    if estimated > limit:
        raise DEMError(
            "DEM raster is too large for the configured memory budget: "
            f"estimated={estimated / 1024**3:.1f} GiB, "
            f"limit={limit / 1024**3:.1f} GiB, file={file_name}. "
            "Reduce the bbox, use 5m/10m input, or disable NoData filling."
        )


def _raster_bounds(file_name):
    if not has_gdal:
        return None
    try:
        dataset = gdal.Open(str(file_name), gdal.GA_ReadOnly)
        if dataset is None:
            return None
        geo = dataset.GetGeoTransform()
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        dataset = None
        if len(geo) != 6 or geo[2] != 0 or geo[4] != 0:
            return None
        x_values = (geo[0], geo[0] + geo[1] * width)
        y_values = (geo[3], geo[3] + geo[5] * height)
        return (min(x_values), min(y_values), max(x_values), max(y_values))
    except Exception:
        return None


def _raster_intersects_tile(file_name, lat, lon):
    bounds = _raster_bounds(file_name)
    if bounds is None:
        return False
    west, south, east, north = bounds
    return not (
        east <= lon or west >= lon + 1.0 or north <= lat or south >= lat + 1.0
    )


def _raster_covers_tile(file_name, lat, lon):
    """Return whether a GDAL raster covers the complete one-degree tile."""
    bounds = _raster_bounds(file_name)
    if bounds is None:
        return False
    west, south, east, north = bounds
    epsilon = 1e-9
    return (
        west <= lon + epsilon
        and east >= lon + 1.0 - epsilon
        and south <= lat + epsilon
        and north >= lat + 1.0 - epsilon
    )


def _raster_resolution(file_name):
    """Return raster pixel area for deterministic GSI candidate ordering."""
    try:
        dataset = gdal.Open(str(file_name), gdal.GA_ReadOnly)
        if dataset is None:
            return float("inf")
        geo = dataset.GetGeoTransform()
        dataset = None
        if len(geo) != 6 or geo[1] == 0 or geo[5] == 0:
            return float("inf")
        return abs(geo[1] * geo[5])
    except Exception:
        return float("inf")


def _raster_contract(file_name):
    dataset = gdal.Open(str(file_name), gdal.GA_ReadOnly)
    if dataset is None:
        raise DEMError(f"Could not open GSI DEM raster: {file_name}")
    band = dataset.GetRasterBand(1)
    scale = band.GetScale()
    offset = band.GetOffset()
    spatial_ref = dataset.GetSpatialRef()
    epsg = spatial_ref.GetAuthorityCode(None) if spatial_ref is not None else None
    contract = (
        gdal.GetDataTypeName(band.DataType),
        1.0 if scale is None else float(scale),
        0.0 if offset is None else float(offset),
        band.GetNoDataValue(),
        epsg,
    )
    dataset = None
    return contract


def _validated_gsi_raster_contract(file_name, expected=None):
    try:
        contract = _raster_contract(file_name)
    except Exception as error:
        if isinstance(error, DEMError):
            raise
        raise DEMError(f"Could not inspect GSI DEM raster: {file_name}: {error}") from error
    if contract[3] is None:
        raise DEMError(f"GSI DEM raster has no NoData contract: {file_name}")
    if contract[4] not in (None, "4326", "4269"):
        raise DEMError(f"Unsupported GSI DEM CRS EPSG:{contract[4]}")
    if expected is not None and contract != expected:
        raise DEMError(
            "GSI DEM raster contract does not match the manifest sources: "
            f"{file_name}"
        )
    return contract


def _build_tile_aligned_vrt(sources, lat, lon):
    if not has_gdal:
        raise DEMError("GDAL is required to combine split GSI DEM files")
    sources = [str(source) for source in sources]
    contracts = [_raster_contract(source) for source in sources]
    first = contracts[0]
    if first[3] is None:
        raise DEMError("GSI DEM sources must advertise a NoData value")
    if first[4] not in (None, "4326", "4269"):
        raise DEMError(f"Unsupported GSI DEM CRS EPSG:{first[4]}")
    if any(contract != first for contract in contracts[1:]):
        raise DEMError(
            "Cannot combine GSI DEM files with different data type, "
            "scale, offset, NoData, or CRS contracts"
        )
    vrt_path = f"/vsimem/ortho4xp-gsi-{uuid.uuid4().hex}.vrt"
    options = gdal.BuildVRTOptions(
        srcNodata=first[3],
        VRTNodata=first[3],
        resampleAlg="nearest",
        resolution="highest",
        outputBounds=(lon, lat, lon + 1.0, lat + 1.0),
    )
    try:
        dataset = gdal.BuildVRT(vrt_path, sources, options=options)
        if dataset is None:
            raise DEMError("Could not create tile-aligned GSI VRT")
        dataset.FlushCache()
        dataset = None
        if not _raster_covers_tile(vrt_path, lat, lon):
            raise DEMError("Generated GSI VRT does not cover the requested tile")
        return vrt_path, lambda: gdal.Unlink(vrt_path)
    except Exception:
        gdal.Unlink(vrt_path)
        raise


def _manifest_path(source):
    candidates = []
    for root, dirs, files in os.walk(source):
        dirs.sort()
        for file_name in sorted(files):
            if file_name.lower() == "gsi_dem_manifest.json":
                candidates.append(Path(root) / file_name)
    return candidates[0] if candidates else None


def _manifest_output_paths(manifest_path, manifest):
    values = manifest.get("outputs")
    if not isinstance(values, list):
        raise DEMError(f"GSI manifest outputs must be a list: {manifest_path}")
    outputs = []
    for value in values:
        if not isinstance(value, str):
            raise DEMError(f"GSI manifest output path is invalid: {manifest_path}")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise DEMError(f"GSI manifest output is missing: {candidate}")
        if candidate.suffix.lower() in (".tif", ".tiff"):
            outputs.append(candidate)
    if not outputs:
        raise DEMError(f"GSI manifest has no readable GeoTIFF outputs: {manifest_path}")
    return outputs


def _is_gsi_dem_path(file_name):
    path = Path(file_name)
    lower_name = path.name.lower()
    if path.suffix.lower() == ".vrt":
        return True
    if "_gsi_" in lower_name or lower_name == "gsi_dem.vrt":
        return True
    return (path.parent / "gsi_dem_manifest.json").is_file()


def _manifest_result_date(result):
    for key in ("source_date", "date", "created_at", "timestamp", "generated_at"):
        value = result.get(key)
        if value:
            digits = "".join(character for character in str(value) if character.isdigit())
            if len(digits) >= 8:
                return int(digits[:14].ljust(14, "0"))
    for value in result.get("inputs", []):
        match = re.search(r"(?<!\d)(?:19|20)\d{6}(?!\d)", str(value))
        if match:
            return int(match.group(0) + "000000")
    return 0


def _resolve_custom_dem_file(source, lat, lon):
    """Return (path, cleanup, is_gsi) for a custom DEM directory."""
    source = Path(source)
    target_hgt = (FNAMES.hem_latlon(lat, lon) + ".hgt").lower()
    target_tif = (FNAMES.hem_latlon(lat, lon) + ".tif").lower()
    target_tiff = (FNAMES.hem_latlon(lat, lon) + ".tiff").lower()
    exact_candidates = []
    gsi_candidates = []
    for root, dirs, files in os.walk(source):
        dirs.sort()
        for file_name in sorted(files):
            lower_name = file_name.lower()
            path = Path(root) / file_name
            if lower_name in (target_hgt, target_tif, target_tiff):
                exact_candidates.append(path)
            if lower_name == "gsi_dem.vrt" or "_gsi_" in lower_name:
                if lower_name.endswith((".tif", ".tiff", ".vrt")):
                    gsi_candidates.append(path)
    if exact_candidates:
        return str(sorted(exact_candidates)[0]), None, True

    manifest_path = _manifest_path(source)
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise DEMError(
                f"Could not read GSI manifest: {manifest_path}: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise DEMError(f"GSI manifest must contain an object: {manifest_path}")
        outputs = _manifest_output_paths(manifest_path, manifest)
        source_contract = _validated_gsi_raster_contract(outputs[0])
        for output in outputs[1:]:
            _validated_gsi_raster_contract(output, source_contract)
        vrt_value = manifest.get("vrt")
        if vrt_value:
            if not isinstance(vrt_value, str):
                raise DEMError(f"GSI manifest VRT path is invalid: {manifest_path}")
            manifest_vrt = Path(vrt_value)
            if not manifest_vrt.is_absolute():
                manifest_vrt = manifest_path.parent / manifest_vrt
            manifest_vrt = manifest_vrt.resolve()
            if not manifest_vrt.is_file():
                raise DEMError(f"GSI manifest VRT is missing: {manifest_vrt}")
            _validated_gsi_raster_contract(manifest_vrt, source_contract)
            if _raster_intersects_tile(manifest_vrt, lat, lon):
                if _raster_covers_tile(manifest_vrt, lat, lon):
                    return str(manifest_vrt), None, True
        overlapping = [
            path for path in outputs if _raster_intersects_tile(path, lat, lon)
        ]
        if not overlapping:
            raise DEMError(
                f"GSI manifest has no output overlapping tile {lat:+d}{lon:+d}: "
                f"{manifest_path}"
            )
        results = manifest.get("results", [])
        if not isinstance(results, list) or any(
            not isinstance(result, dict) for result in results
        ):
            raise DEMError(f"GSI manifest results are invalid: {manifest_path}")
        result_by_path = {}
        for result in results:
            output_value = result.get("output")
            if isinstance(output_value, str):
                output_path = Path(output_value)
                if not output_path.is_absolute():
                    output_path = manifest_path.parent / output_path
                result_by_path[str(output_path.resolve())] = result
        for index, path in enumerate(outputs):
            if str(path) not in result_by_path and index < len(results):
                result_by_path[str(path)] = results[index]

        def priority(path):
            result = result_by_path.get(str(path), {})
            resolution = result.get("resolution", "")
            try:
                resolution_value = float(str(resolution).rstrip("m"))
            except ValueError:
                resolution_value = float("inf")
            date_value = _manifest_result_date(result)
            # GDAL BuildVRT gives the last overlapping source precedence.
            # Put lower-quality/older sources first so the preferred source is last.
            resolution_order = -resolution_value
            if resolution_value == float("inf"):
                resolution_order = float("-inf")
            return resolution_order, date_value, outputs.index(path), str(path)

        overlapping.sort(key=priority)
        if len(overlapping) == 1 and _raster_covers_tile(overlapping[0], lat, lon):
            return str(overlapping[0]), None, True
        return (*_build_tile_aligned_vrt(overlapping, lat, lon), True)

    if not gsi_candidates:
        return None, None, False
    overlapping = [
        path for path in gsi_candidates if _raster_intersects_tile(path, lat, lon)
    ]
    if not overlapping:
        raise DEMError(
            f"GSI DEM files do not overlap tile {lat:+d}{lon:+d} in {source}"
        )
    vrt_candidates = [
        path for path in overlapping if path.name.lower() == "gsi_dem.vrt"
    ]
    if vrt_candidates:
        vrt_path = sorted(vrt_candidates)[0]
        _validated_gsi_raster_contract(vrt_path)
        if _raster_covers_tile(vrt_path, lat, lon):
            return str(vrt_path), None, True
        return (*_build_tile_aligned_vrt([vrt_path], lat, lon), True)
    covering = [path for path in overlapping if _raster_covers_tile(path, lat, lon)]
    if len(overlapping) > 1 and not covering:
        raise DEMError(
            "Multiple partial GSI DEM files were found without gsi_dem.vrt. "
            "Create a VRT before using this directory."
        )
    if covering:
        return (
            str(min(covering, key=lambda path: (_raster_resolution(path), str(path)))),
            None,
            True,
        )
    return (*_build_tile_aligned_vrt(overlapping, lat, lon), True)


def _find_custom_dem_file(source, lat, lon):
    """Compatibility wrapper returning the resolved custom DEM path."""
    return _resolve_custom_dem_file(source, lat, lon)[0]

################################################################################
class DEM:
    def __init__(self, lat, lon, source="", fill_nodata=True, info_only=False):
        self.lat = lat
        self.lon = lon
        source = source.replace("{latlon}", FNAMES.hem_latlon(lat, lon))
        if ";" in source:
            self.alt = self.alt_composite
            self.alt_vec = self.alt_vec_composite
        else:
            self.alt = self.alt_nostrict
            self.alt_vec = self.alt_vec_nostrict
        self._preserve_nodata = False
        self.load_data(source, info_only, fill_nodata)
        if info_only:
            return
        if fill_nodata == "to zero":
            self.nodata_to_zero()
        elif fill_nodata and not self._preserve_nodata:
            if not fill_nodata_values_with_nearest_neighbor(
                self.alt_dem, self.nodata
            ):
                UI.vprint(
                    1,
                    "   INFO: Dataset contains too much no_data to be filled.",
                )
                self.nodata_to_zero()

        UI.vprint(
            1,
            "    * Min altitude:",
            self.alt_dem.min(),
            ", Max altitude:",
            self.alt_dem.max(),
            ", Mean:",
            self.alt_dem.mean(),
        )

    def load_data(self, source, info_only=False, fill_nodata=True):
        if not source:
            if os.path.exists(FNAMES.generic_tif(self.lat, self.lon)):
                source = FNAMES.generic_tif(self.lat, self.lon)
            else:
                source = available_sources[1]
        if ";" in source:
            source, local_sources = source.split(";")[0], source.split(";")[1:]
        else:
            local_sources = None
        if source in available_sources[1::2]:
            short_source = available_sources[
                available_sources.index(source) - 1
            ]
            if short_source in global_sources:
                (
                    self.epsg,
                    self.x0,
                    self.y0,
                    self.x1,
                    self.y1,
                    self.nodata,
                    self.nxdem,
                    self.nydem,
                    self.alt_dem,
                ) = build_combined_raster(
                    short_source, self.lat, self.lon, info_only
                )
            else:
                if ensure_elevation(short_source, self.lat, self.lon):
                    (
                        self.epsg,
                        self.x0,
                        self.y0,
                        self.x1,
                        self.y1,
                        self.nodata,
                        self.nxdem,
                        self.nydem,
                        self.alt_dem,
                    ) = read_elevation_from_file(
                        FNAMES.elevation_data(short_source, self.lat, self.lon),
                        self.lat,
                        self.lon,
                        info_only,
                        3601,
                        fill_nodata=fill_nodata,
                    )
                else:
                    (
                        self.epsg,
                        self.x0,
                        self.y0,
                        self.x1,
                        self.y1,
                        self.nodata,
                        self.nxdem,
                        self.nydem,
                        self.alt_dem,
                    ) = (
                        4326,
                        0,
                        0,
                        1,
                        1,
                        -32768,
                        3601,
                        3601,
                        numpy.zeros((3601, 3601), dtype=numpy.float32),
                    )
        else:
            cleanup = None
            strict_custom = False
            if os.path.isdir(source):
                file_name, cleanup, strict_custom = _resolve_custom_dem_file(
                    source, self.lat, self.lon
                )
                if file_name is not None:
                    self._preserve_nodata = bool(
                        strict_custom and str(file_name).lower().endswith(".vrt")
                    )
                    UI.vprint(1, "   INFO: Found matching custom DEM in directory:", file_name)
                else:
                    UI.vprint(1, "   INFO: No matching DEM found in", source, ", falling back to default.")
                    if os.path.exists(FNAMES.generic_tif(self.lat, self.lon)):
                        file_name = FNAMES.generic_tif(self.lat, self.lon)
                    else:
                        short_source = available_sources[0]
                        if ensure_elevation(short_source, self.lat, self.lon):
                            file_name = FNAMES.elevation_data(short_source, self.lat, self.lon)
                        else:
                            # If even fallback fails, don't pass the directory to read_elevation_from_file
                            # instead, set to empty so it gets zero altitude in read_elevation_from_file or here
                            file_name = "" 
            else:
                file_name = source
                strict_custom = _is_gsi_dem_path(file_name)
                self._preserve_nodata = bool(
                    strict_custom and str(file_name).lower().endswith(".vrt")
                )
            if not file_name:
                (
                    self.epsg,
                    self.x0,
                    self.y0,
                    self.x1,
                    self.y1,
                    self.nodata,
                    self.nxdem,
                    self.nydem,
                    self.alt_dem,
                ) = (
                    4326,
                    0,
                    0,
                    1,
                    1,
                    -32768,
                    3601,
                    3601,
                    numpy.zeros((3601, 3601), dtype=numpy.float32),
                )
            else:
                try:
                    (
                        self.epsg,
                        self.x0,
                        self.y0,
                        self.x1,
                        self.y1,
                        self.nodata,
                        self.nxdem,
                        self.nydem,
                        self.alt_dem,
                    ) = read_elevation_from_file(
                        file_name,
                        self.lat,
                        self.lon,
                        info_only,
                        strict=strict_custom,
                        fill_nodata=fill_nodata,
                    )
                finally:
                    if cleanup is not None:
                        cleanup()
        if not local_sources:
            return
        self.subdems = tuple()
        for local_source in local_sources:
            self.subdems += (
                DEM(self.lat, self.lon, local_source, False, info_only),
            )
            self.subdems[-1].alt = self.subdems[-1].alt_strict
            self.subdems[-1].alt_vec = self.subdems[-1].alt_vec_strict

    def nodata_to_zero(self):
        if (self.alt_dem == self.nodata).any():
            UI.vprint(1, "   INFO: Replacing nodata nodes with zero altitude.")
            self.alt_dem[self.alt_dem == self.nodata] = 0
        self.nodata = -32768
        return

    def write_to_file(self, filename):
        self.alt_dem.astype(numpy.float32).tofile(filename)
        return

    def create_normal_map(self, pixx, pixy):
        dx = numpy.zeros((self.nxdem, self.nydem))
        dy = numpy.zeros((self.nxdem, self.nydem))
        dx[:, 1:-1] = (self.alt_dem[:, 2:] - self.alt_dem[:, 0:-2]) / (2 * pixx)
        dx[:, 0] = (self.alt_dem[:, 1] - self.alt_dem[:, 0]) / (pixx)
        dx[:, -1] = (self.alt_dem[:, -1] - self.alt_dem[:, -2]) / (pixx)
        dy[1:-1, :] = (self.alt_dem[:-2, :] - self.alt_dem[2:, :]) / (2 * pixy)
        dy[0, :] = (self.alt_dem[0, :] - self.alt_dem[1, :]) / (pixy)
        dy[-1, :] = (self.alt_dem[-2, :] - self.alt_dem[-1, :]) / (pixy)
        del self.alt_dem
        norm = numpy.sqrt(1 + dx ** 2 + dy ** 2)
        dx = dx / norm
        dy = dy / norm
        del norm
        band_r = Image.fromarray(
            ((1 + dx) / 2 * 255).astype(numpy.uint8)
        ).resize((4096, 4096))
        del dx
        band_g = Image.fromarray(
            ((1 - dy) / 2 * 255).astype(numpy.uint8)
        ).resize((4096, 4096))
        del dy
        band_b = Image.fromarray(
            (numpy.ones((4096, 4096)) * 10).astype(numpy.uint8)
        )
        band_a = Image.fromarray(
            (numpy.ones((4096, 4096)) * 128).astype(numpy.uint8)
        )
        im = Image.merge("RGBA", (band_r, band_g, band_b, band_a))
        im.save("normal_map.png")

    def super_level_set(self, level, wgs84_bbox):
        (lonmin, lonmax, latmin, latmax) = wgs84_bbox
        xmin = lonmin - self.lon
        xmax = lonmax - self.lon
        ymin = latmin - self.lat
        ymax = latmax - self.lat
        if xmin < self.x0:
            xmin = self.x0
        if xmax > self.x1:
            xmax = self.x1
        if ymin < self.y0:
            ymin = self.y0
        if ymax > self.y1:
            ymax = self.y1
        pixx0 = round((xmin - self.x0) / (self.x1 - self.x0) * (self.nxdem - 1))
        pixx1 = round((xmax - self.x0) / (self.x1 - self.x0) * (self.nxdem - 1))
        pixy0 = round((self.y1 - ymax) / (self.y1 - self.y0) * (self.nydem - 1))
        pixy1 = round((self.y1 - ymin) / (self.y1 - self.y0) * (self.nydem - 1))
        return (
            (
                xmin + self.lon,
                xmax + self.lon,
                ymin + self.lat,
                ymax + self.lat,
            ),
            self.alt_dem[pixy0 : pixy1 + 1, pixx0 : pixx1 + 1] >= level,
        )

    def alt_nostrict(self, node):
        Nx = self.nxdem - 1
        Ny = self.nydem - 1
        x = node[0]
        y = node[1]
        x = max(x, self.x0)
        x = min(x, self.x1)
        y = max(y, self.y0)
        y = min(y, self.y1)
        px = (x - self.x0) / (self.x1 - self.x0) * Nx
        py = (y - self.y0) / (self.y1 - self.y0) * Ny
        nx = int(px)
        Nminusny = Ny - int(py)
        rx = px - nx
        ry = py + Nminusny - Ny
        t1 = self.alt_dem[Nminusny, nx]
        t2 = self.alt_dem[
            (Nminusny - 1) * (Nminusny >= 1),
            (nx + 1) * (nx < Nx) + Nx * (nx == Nx),
        ]
        t3 = self.alt_dem[Nminusny, (nx + 1) * (nx < Nx) + Nx * (nx == Nx)]
        t4 = self.alt_dem[(Nminusny - 1) * (Nminusny >= 1), nx]
        return ((1 - rx) * t1 + ry * t2 + (rx - ry) * t3) * (rx >= ry) + (
            (1 - ry) * t1 + rx * t2 + (ry - rx) * t4
        ) * (rx < ry)

    def alt_strict(self, node):
        x = node[0]
        y = node[1]
        return (
            self.nodata
            if (
                (x > self.x1) or (x < self.x0) or (y < self.y0) or (y > self.y1)
            )
            else self.alt_dem[
                int(
                    round(
                        (self.y1 - y) / (self.y1 - self.y0) * (self.nydem - 1)
                    )
                ),
                int(
                    round(
                        (x - self.x0) / (self.x1 - self.x0) * (self.nxdem - 1)
                    )
                ),
            ]
        )

    def alt_composite(self, node):
        for subdem in self.subdems[::-1]:
            tmp = subdem.alt_strict(node)
            if tmp != subdem.nodata:
                return tmp
        return self.alt_nostrict(node)

    def alt_vec_nostrict(self, way):
        Nx = self.nxdem - 1
        Ny = self.nydem - 1
        x, y = way[:, 0], way[:, 1]
        x = numpy.maximum.reduce([x, self.x0 * numpy.ones(x.shape)])
        x = numpy.minimum.reduce([x, self.x1 * numpy.ones(x.shape)])
        y = numpy.maximum.reduce([y, self.y0 * numpy.ones(y.shape)])
        y = numpy.minimum.reduce([y, self.y1 * numpy.ones(y.shape)])
        px = (x - self.x0) / (self.x1 - self.x0) * Nx
        py = (y - self.y0) / (self.y1 - self.y0) * Ny
        nx = px.astype(numpy.int64)
        Nminusny = Ny - py.astype(numpy.int64)
        rx = px - nx
        ry = py + Nminusny - Ny
        t1 = [self.alt_dem[i][j] for i, j in zip(Nminusny, nx)]
        t2 = [
            self.alt_dem[i][j]
            for i, j in zip(
                (Nminusny - 1) * (Nminusny >= 1),
                (nx + 1) * (nx < Nx) + Nx * (nx == Nx),
            )
        ]
        t3 = [
            self.alt_dem[i][j]
            for i, j in zip(Nminusny, (nx + 1) * (nx < Nx) + Nx * (nx == Nx))
        ]
        t4 = [
            self.alt_dem[i][j]
            for i, j in zip((Nminusny - 1) * (Nminusny >= 1), nx)
        ]
        return ((1 - rx) * t1 + ry * t2 + (rx - ry) * t3) * (rx >= ry) + (
            (1 - ry) * t1 + rx * t2 + (ry - rx) * t4
        ) * (rx < ry)

    def alt_vec_strict(self, way):
        x, y = way[:, 0], way[:, 1]
        mask = (x >= self.x0) * (x <= self.x1) * (y >= self.y0) * (y <= self.y1)
        nx = numpy.round(
            (x - self.x0) / (self.x1 - self.x0) * (self.nxdem - 1)
        ).astype(numpy.int64)
        Nminusny = numpy.round(
            (self.y1 - y) / (self.y1 - self.y0) * (self.nydem - 1)
        ).astype(numpy.int64)
        return numpy.array(
            [
                self.alt_dem[i][j] if k else self.nodata
                for i, j, k in zip(Nminusny, nx, mask)
            ]
        )

    def alt_vec_composite(self, way):
        tmp = self.alt_vec_nostrict(way)
        for subdem in self.subdems:
            tmp2 = subdem.alt_vec_strict(way)
            tmp[tmp2 != subdem.nodata] = tmp2[tmp2 != subdem.nodata]
        return tmp

################################################################################
def build_combined_raster(source, lat, lon, info_only):
    world_tiles = numpy.array(
        Image.open(os.path.join(FNAMES.Utils_dir, "world_tiles.png"))
    )
    if source in ("View", "SRTM"):
        base = 3601
        overlap = 1
        beyond = 36
        x0 = y0 = -0.01
        x1 = y1 = 1.01
        epsg = 4326
        nodata = -32768
        nxdem = nydem = base + 2 * beyond  # = 3673
    elif source == ("ALOS"):
        base = 3600
        overlap = 0
        beyond = 36
        eps = 1 / 7200
        x0 = y0 = -0.01 + eps
        x1 = y1 = 1.01 - eps
        epsg = 4326
        nodata = -32768
        nxdem = nydem = base + 2 * beyond  # = 3672
    if info_only:
        return (epsg, x0, y0, x1, y1, nodata, nxdem, nydem, None)
    alt_dem = numpy.zeros((nydem, nxdem), dtype=numpy.float32)
    for (lat0, lon0) in itertools.product(
        (lat, lat - 1, lat + 1), (lon, lon - 1, lon + 1)
    ):
        verbose = True if (lat0 == lat and lon0 == lon) else False
        x = (180 + lon0) % 360
        y = 89 - lat0
        if not world_tiles[y, x]:
            tmparray = numpy.zeros((base, base), dtype=numpy.float32)
        elif ensure_elevation(source, lat0, (lon0 + 180) % 360 - 180, verbose):
            tmparray = read_elevation_from_file(
                FNAMES.elevation_data(source, lat0, (lon0 + 180) % 360 - 180),
                lat0,
                (lon0 + 180) % 360 - 180,
                info_only,
                base,
            )[-1]
        else:
            tmparray = numpy.zeros((base, base), dtype=numpy.float32)
        by = beyond
        ov = overlap
        if lat0 == lat and lon0 == lon:
            alt_dem[by:-by, by:-by] = tmparray
        elif lat0 == lat and lon0 == lon - 1:
            alt_dem[by:-by, :by] = (
                tmparray[:, -by - ov : -ov] if ov else tmparray[:, -by:]
            )
        elif lat0 == lat and lon0 == lon + 1:
            alt_dem[by:-by, -by:] = (
                tmparray[:, ov : ov + by] if ov else tmparray[:, :by]
            )
        elif lat0 == lat + 1 and lon0 == lon:
            alt_dem[:by, by:-by] = (
                tmparray[-ov - by : -ov, :] if ov else tmparray[-by:, :]
            )
        elif lat0 == lat - 1 and lon0 == lon:
            alt_dem[-by:, by:-by] = (
                tmparray[ov : ov + by, :] if ov else tmparray[:by, :]
            )
        elif lat0 == lat + 1 and lon0 == lon - 1:
            alt_dem[:by, :by] = (
                tmparray[-ov - by : -ov, -ov - by : -ov]
                if ov
                else tmparray[-by:, -by:]
            )
        elif lat0 == lat + 1 and lon0 == lon + 1:
            alt_dem[:by, -by:] = (
                tmparray[-ov - by : -ov, ov : ov + by]
                if ov
                else tmparray[-by:, :by]
            )
        elif lat0 == lat - 1 and lon0 == lon - 1:
            alt_dem[-by:, :by] = (
                tmparray[ov : ov + by, -ov - by : -ov]
                if ov
                else tmparray[:by, -by:]
            )
        elif lat0 == lat - 1 and lon0 == lon + 1:
            alt_dem[-by:, -by:] = (
                tmparray[ov : ov + by, ov : ov + by]
                if ov
                else tmparray[:by, :by]
            )
    return (epsg, x0, y0, x1, y1, nodata, nxdem, nydem, alt_dem)

################################################################################
def read_elevation_from_file(
    file_name,
    lat,
    lon,
    info_only=False,
    base_if_error=3601,
    strict=False,
    fill_nodata=True,
):
    alt_dem = None
    if file_name[-4:].lower() == ".hgt":
        x0 = y0 = 0
        x1 = y1 = 1
        epsg = 4326
        nodata = -32768
        try:
            if not os.path.isfile(file_name):
                raise FileNotFoundError
            nxdem = nydem = int(round(sqrt(os.path.getsize(file_name) / 2)))
            _check_raster_memory(
                nxdem,
                nydem,
                1.0,
                0.0,
                bool(fill_nodata) and fill_nodata != "to zero",
                file_name,
                raw_bytes_per_cell=2,
            )
            if not info_only:
                alt_dem = (
                    numpy.fromfile(file_name, numpy.dtype(">i2"))
                    .astype(numpy.float32)
                    .reshape((nydem, nxdem))
                )
            if nxdem == 1201:
                nxdem = nydem = 3601
                if not info_only:
                    fill_nodata_values_with_nearest_neighbor(alt_dem, nodata)
                    alt_dem = upsample(alt_dem)
        except DEMError:
            raise
        except Exception as e:
            if strict:
                raise DEMError(f"Could not read custom DEM {file_name}: {e}") from e
            print(e)
            UI.lvprint(
                1,
                "    ERROR: in reading elevation from",
                file_name,
                "-> replaced with zero altitude.",
            )
            nxdem = nydem = base_if_error
            if not info_only:
                alt_dem = numpy.zeros(
                    (base_if_error, base_if_error), dtype=numpy.float32
                )

    elif file_name[-4:].lower() == ".raw":
        try:
            if not os.path.isfile(file_name):
                raise FileNotFoundError
            nxdem = nydem = int(round(sqrt(os.path.getsize(file_name) / 2)))
            _check_raster_memory(
                nxdem,
                nydem,
                1.0,
                0.0,
                bool(fill_nodata) and fill_nodata != "to zero",
                file_name,
                raw_bytes_per_cell=2,
            )
            f = open(file_name, "rb")
            alt = array.array("h")
            alt.fromfile(f, nxdem * nydem)
            f.close()
            if not info_only:
                alt_dem = numpy.asarray(alt, dtype=numpy.float32).reshape(
                    (nxdem, nydem)
                )[::-1]
        except DEMError:
            raise
        except Exception as error:
            if strict:
                raise DEMError(f"Could not read custom DEM {file_name}: {error}") from error
            UI.lvprint(
                1,
                "    ERROR: in reading elevation from",
                file_name,
                "-> replaced with zero altitude.",
            )
            nxdem = nydem = base_if_error
            if not info_only:
                alt_dem = numpy.zeros(
                    (base_if_error, base_if_error), dtype=numpy.float32
                )
        x0 = y0 = 0
        x1 = y1 = 1
        epsg = 4326
        nodata = -32768
    elif has_gdal:
        try:
            ds = gdal.Open(file_name)
            rs = ds.GetRasterBand(1)
            raw_nodata = rs.GetNoDataValue()
            scale = rs.GetScale()
            offset = rs.GetOffset()
            scale = 1.0 if scale is None else numpy.float32(scale)
            offset = 0.0 if offset is None else numpy.float32(offset)
            _check_raster_memory(
                ds.RasterXSize,
                ds.RasterYSize,
                scale,
                offset,
                bool(fill_nodata) and fill_nodata != "to zero",
                file_name,
            )
            if not info_only:
                alt_dem = rs.ReadAsArray(buf_type=gdal.GDT_Float32)
                nodata_mask = (
                    alt_dem == numpy.float32(raw_nodata)
                    if raw_nodata is not None
                    else numpy.zeros(alt_dem.shape, dtype=bool)
                )
                if scale != 1.0 or offset != 0.0:
                    alt_dem *= scale
                    alt_dem += offset
                if raw_nodata is not None:
                    alt_dem[nodata_mask] = -32768
            (nxdem, nydem) = (ds.RasterXSize, ds.RasterYSize)
            nodata = raw_nodata
            if raw_nodata is None:
                UI.vprint(
                    1,
                    "    WARNING: raster DEM does not advertise its no_data ",
                    "value, assuming -32768.",
                )
                nodata = -32768
            else:
                nodata = -32768
            try:
                epsg = int(ds.GetProjection().split('"')[-2])
            except:
                UI.vprint(
                    1,
                    "    WARNING: raster DEM does not advertise its EPSG ",
                    "code, assuming 4326.",
                )
                epsg = 4326
            if epsg not in (
                4326,
                4269,
            ):  
            # let's be blind about 4269 which might be sufficiently close to 
            # 4326 for our purposes
                UI.lvprint(
                    1,
                    "    WARNING: unsupported EPSG code ",
                    epsg,
                    ". Only EPSG:4326 is supported, result is likely to ",
                    "be non sense.",
                )
            geo = ds.GetGeoTransform()
            # We are assuming AREA_OR_POINT is area here
            x0 = geo[0] + 0.5 * geo[1] - lon
            y1 = geo[3] + 0.5 * geo[5] - lat
            x1 = x0 + (nxdem - 1) * geo[1]
            y0 = y1 + (nydem - 1) * geo[5]
        except DEMError:
            raise
        except Exception as error:
            if strict:
                raise DEMError(f"Could not read custom DEM {file_name}: {error}") from error
            UI.lvprint(
                1,
                "   ERROR: in reading ",
                file_name,
                "-> replaced with zero altitude.",
            )
            nxdem = nydem = base_if_error
            if not info_only:
                alt_dem = numpy.zeros(
                    (base_if_error, base_if_error), dtype=numpy.float32
                )
            x0 = y0 = 0
            x1 = y1 = 1
            epsg = 4326
            nodata = -32768
    elif not has_gdal:
        if strict:
            raise DEMError(f"GDAL is required to read custom DEM {file_name}")
        UI.lvprint(
            1,
            "   WARNING: unsupported raster (install Gdal):",
            file_name,
            "-> replaced with zero altitude.",
        )
        nxdem = nydem = base_if_error
        if not info_only:
            alt_dem = numpy.zeros(
                (base_if_error, base_if_error), dtype=numpy.float32
            )
        x0 = y0 = 0
        x1 = y1 = 1
        epsg = 4326
        nodata = -32768
    return (epsg, x0, y0, x1, y1, nodata, nxdem, nydem, alt_dem)


##############################################################################

##############################################################################
def ensure_elevation(source, lat, lon, verbose=True):
    if source == "View":
        # Viewfinderpanorama grouping of files and resolutions is a 
        # bit complicated...
        if (lat, lon) in (
            (44, 5),
            (45, 5),
            (46, 5),
            (43, 6),
            (44, 6),
            (45, 6),
            (46, 6),
            (47, 6),
            (43, 7),
            (44, 7),
            (45, 7),
            (46, 7),
            (47, 7),
            (45, 8),
            (46, 8),
            (47, 8),
            (45, 9),
            (46, 9),
            (47, 9),
            (45, 10),
            (46, 10),
            (47, 10),
            (45, 11),
            (46, 11),
            (47, 11),
            (45, 12),
            (46, 12),
            (47, 12),
            (46, 13),
            (47, 13),
            (46, 14),
            (47, 14),
            (46, 15),
            (47, 15),
        ):
            resol = 1
            url = (
                "http://viewfinderpanoramas.org/dem1/"
                + os.path.basename(FNAMES.base_file_name(lat, lon)).lower()
                + ".zip"
            )
        else:
            deferranti_nbr = 31 + lon // 6
            if deferranti_nbr < 10:
                deferranti_nbr = "0" + str(deferranti_nbr)
            else:
                deferranti_nbr = str(deferranti_nbr)
            alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            deferranti_letter = (
                alphabet[lat // 4] if lat >= 0 else alphabet[(-1 - lat) // 4]
            )
            if lat < 0:
                deferranti_letter = "S" + deferranti_letter
            if deferranti_letter + deferranti_nbr in (
                "O31",
                "P31",
                "N32",
                "O32",
                "P32",
                "Q32",
                "N33",
                "O33",
                "P33",
                "Q33",
                "R33",
                "O34",
                "P34",
                "Q34",
                "R34",
                "O35",
                "P35",
                "Q35",
                "R35",
                "P36",
                "Q36",
                "R36",
            ):
                resol = 1
            else:
                resol = 3
            url = (
                "http://viewfinderpanoramas.org/dem"
                + str(resol)
                + "/"
                + deferranti_letter
                + deferranti_nbr
                + ".zip"
            )
        if os.path.exists(FNAMES.viewfinderpanorama(lat, lon)) and (
            resol == 3
            or os.path.getsize(FNAMES.viewfinderpanorama(lat, lon)) >= 25934402
        ):
            UI.vprint(2, "   Recycling ", FNAMES.viewfinderpanorama(lat, lon))
            return 1
        UI.vprint(
            1,
            "    Downloading ",
            FNAMES.viewfinderpanorama(lat, lon),
            "from Viewfinderpanoramas (J. de Ferranti).",
        )
        r = http_request(url, source, verbose)
        if not r:
            return 0
        with zipfile.ZipFile(io.BytesIO(r.content), "r") as zip_ref:
            for f in zip_ref.filelist:
                fname = os.path.basename(f.filename)
                if not fname:
                    continue
                try:
                    lat0 = int(fname[1:3])
                    lon0 = int(fname[4:7])
                except:
                    UI.vprint(
                        2,
                        "      Archive contains the unknown file name",
                        fname,
                        "which is skipped.",
                    )
                    continue
                if ("S" in fname) or ("s" in fname):
                    lat0 *= -1
                if ("W" in fname) or ("w" in fname):
                    lon0 *= -1
                out_filename = FNAMES.viewfinderpanorama(lat0, lon0)
                # we don't wish to overwrite a 1" version by downloading 
                # the whole archive of a nearby 3" one
                if (
                    not os.path.exists(out_filename)
                    or os.path.getsize(out_filename) <= f.file_size
                ):
                    if not os.path.isdir(os.path.dirname(out_filename)):
                        os.makedirs(os.path.dirname(out_filename))
                    with open(out_filename, "wb") as out:
                        UI.vprint(2, "      Extracting", out_filename)
                        out.write(zip_ref.open(f, "r").read())
        return 1 if os.path.exists(FNAMES.viewfinderpanorama(lat, lon)) else 0
    elif source in ("SRTM", "ALOS"):
        if os.path.exists(FNAMES.elevation_data(source, lat, lon)):
            UI.vprint(
                2, "   Recycling ", FNAMES.elevation_data(source, lat, lon)
            )
            return 1
        UI.vprint(
            1,
            "    WARNING : This elevation source has no longer direct downloads !"
        )
        return 0
        # TODO : is there a way to get it back (worth it ?) 
        url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/"
        if source == "SRTM":
            url += "SRTM_GL1/SRTM_GL1_srtm/"
            if lat < -60 or lat >= 60:
                return 0
            if lat < 0:
                url += "South/"
            elif lat <= 29:
                url += "North/North_0_29/"
            else:
                url += "North/North_30_60/"
            url += os.path.basename(FNAMES.viewfinderpanorama(lat, lon))
        elif source == "ALOS":
            url += "AW3D30/AW3D30_alos/"
            if lat < 0:
                url += "South/"
            elif lat <= 45:
                url += "North/North_0_45/"
            else:
                url += "North/North_46_90/"
            tmp = os.path.basename(FNAMES.base_file_name(lat, lon))
            tmp = tmp[0] + "0" + tmp[1:] + "_AVE_DSM.tif"
            url += tmp
        r = http_request(url, source, verbose)
        if not r:
            return 0
        if not os.path.isdir(
            os.path.dirname(FNAMES.elevation_data(source, lat, lon))
        ):
            os.makedirs(
                os.path.dirname(FNAMES.elevation_data(source, lat, lon))
            )
        with open(FNAMES.elevation_data(source, lat, lon), "wb") as out:
            try:
                out.write(r.content)
            except:
                return 0
    elif source in ("NED1", "NED1/3"):
        if os.path.exists(FNAMES.elevation_data(source, lat, lon)):
            UI.vprint(
                2, "   Recycling ", FNAMES.elevation_data(source, lat, lon)
            )
            return 1
        UI.vprint(
            1,
            "    Downloading ",
            FNAMES.elevation_data(source, lat, lon),
            "from USGS.",
        )
        nbr = "1" if source == "NED1" else "13"
        url_base = (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/"
            + nbr + "/TIFF/current/"
        )
        tid = "n" if lat >= 0 else "s"
        tid = tid + str(abs(lat + 1)).zfill(2)
        tid = tid + "w" if lon < 0 else "e"
        tid = tid + str(abs(lon)).zfill(3)
        url_base = url_base + tid + "/"
        usgs_name = (
            "USGS_" + nbr + "_" + tid + ".tif"
        )
        url = url_base + usgs_name
        r = http_request(url, source, verbose)
        if not r:
            return 0
        if not os.path.isdir(
            os.path.dirname(FNAMES.elevation_data(source, lat, lon))
        ):
            os.makedirs(
                os.path.dirname(FNAMES.elevation_data(source, lat, lon))
            )
        with open(FNAMES.elevation_data(source, lat, lon), "wb") as out:
            try:
                out.write(r.content)
            except:
                return 0
    else:
        UI.vprint(1, "   ERROR: Unknown elevation source.")
        return 0
    return 1

################################################################################
def http_request(url, source, verbose=False):
    s = requests.Session()
    tentative = 0
    while True:
        try:
            r = s.get(url, timeout=10)
            status_code = str(r)
            if "[20" in status_code:
                return r
            elif "[40" in status_code or "[30" in status_code:
                if verbose:
                    UI.vprint(2, "    Server said 'Not Found'")
                return 0
            elif "[5" in status_code:
                if verbose:
                    UI.vprint(
                        2, "    Server said 'Internal Error'.", status_code
                    )
            else:
                if verbose:
                    UI.vprint(2, status_code)
        except Exception as e:
            if verbose:
                UI.vprint(2, e)
        tentative += 1
        if tentative == 6:
            return 0
        UI.vprint(
            1,
            "    ",
            source,
            "server may be down or busy, new tentative in",
            2 ** tentative,
            "sec...",
        )
        time.sleep(2 ** tentative)

################################################################################
def fill_nodata_values_with_nearest_neighbor(alt_dem, nodata):
    step = 0
    while (alt_dem == nodata).any():
        if not step:
            if numpy.sum(alt_dem == nodata) >= 100000:
                return 0
            UI.vprint(
                2,
                "    INFO: Elevation file contains voids, trying to fill ",
                "them recursively by nearest neighbour.",
            )
        else:
            UI.vprint(2, "    ", step)
        alt10 = numpy.roll(alt_dem, 1, axis=0)
        alt10[0] = alt_dem[0]
        alt20 = numpy.roll(alt_dem, -1, axis=0)
        alt20[-1] = alt_dem[-1]
        alt01 = numpy.roll(alt_dem, 1, axis=1)
        alt01[:, 0] = alt_dem[:, 0]
        alt02 = numpy.roll(alt_dem, -1, axis=1)
        alt02[:, -1] = alt_dem[:, -1]
        if (nodata < 0):
            atemp = numpy.maximum(alt10, alt20)
            atemp = numpy.maximum(atemp, alt01)
            atemp = numpy.maximum(atemp, alt02)
        else:
            atemp = numpy.minimum(alt10, alt20)
            atemp = numpy.minimum(atemp, alt01)
            atemp = numpy.minimum(atemp, alt02)
        alt_dem[alt_dem == nodata] = atemp[alt_dem == nodata]
        step += 1
        if step > 20:
            UI.vprint(
                1,
                "    WARNING: The raster contain holes that seem to big to ",
                "be filled... I'm filling the remainder with zero.",
            )
            alt_dem[alt_dem == nodata] = 0
            break
    if step:
        UI.vprint(2, "    Done.")
    return 1

################################################################################
def upsample(alt_dem):
    # only implemented from 1201 to 3601, might be worth upgrading it some day
    alt_dem_tmp = numpy.zeros((3601, 3601), dtype=numpy.float32)
    for i in range(1201):
        alt_dem_tmp[3 * i, ::3] = alt_dem[i]
        alt_dem_tmp[3 * i, 1::3] = (
            2 / 3 * alt_dem[i, :-1] + 1 / 3 * alt_dem[i, 1:]
        )
        alt_dem_tmp[3 * i, 2::3] = (
            1 / 3 * alt_dem[i, :-1] + 2 / 3 * alt_dem[i, 1:]
        )
        if i == 1200:
            break
        alt_dem_tmp[3 * i + 1, ::3] = (
            2 / 3 * alt_dem[i] + 1 / 3 * alt_dem[i + 1]
        )
        alt_dem_tmp[3 * i + 2, ::3] = (
            1 / 3 * alt_dem[i] + 2 / 3 * alt_dem[i + 1]
        )
        alt_dem_tmp[3 * i + 1, 1::3] = (
            4 / 9 * alt_dem[i][:-1]
            + 2 / 9 * alt_dem[i, 1:]
            + 2 / 9 * alt_dem[i + 1, :-1]
            + 1 / 9 * alt_dem[i + 1, 1:]
        )
        alt_dem_tmp[3 * i + 2, 1::3] = (
            2 / 9 * alt_dem[i][:-1]
            + 1 / 9 * alt_dem[i, 1:]
            + 4 / 9 * alt_dem[i + 1, :-1]
            + 2 / 9 * alt_dem[i + 1, 1:]
        )
        alt_dem_tmp[3 * i + 1, 2::3] = (
            2 / 9 * alt_dem[i][:-1]
            + 4 / 9 * alt_dem[i, 1:]
            + 1 / 9 * alt_dem[i + 1, :-1]
            + 2 / 9 * alt_dem[i + 1, 1:]
        )
        alt_dem_tmp[3 * i + 2, 2::3] = (
            1 / 9 * alt_dem[i][:-1]
            + 2 / 9 * alt_dem[i, 1:]
            + 2 / 9 * alt_dem[i + 1, :-1]
            + 4 / 9 * alt_dem[i + 1, 1:]
        )
    return alt_dem_tmp

################################################################################
def smoothen(
    raster,
    pix_width,
    mask_im,
    preserve_boundary=True,
    ashelper_server=None,
):
    if not pix_width:
        return raster
    if not mask_im:
        return raster
    tmp = numpy.array(raster)
    mask_array = numpy.array(mask_im, dtype=numpy.float32) / 255
    kernel = numpy.array(range(1, 2 * (pix_width + 1)))
    kernel[pix_width + 1 :] = range(pix_width, 0, -1)
    kernel = kernel / (pix_width + 1) ** 2
    tmp = tmp * mask_array
    tmpw = numpy.array(mask_array)
    use_gpu = getattr(UI, "use_gpu_for_dem_smoothing", False)
    resident_gpu_result = None
    if use_gpu and ashelper_server is not None:
        resident_gpu_result = _resident_dem_smoothing(
            ashelper_server,
            numpy.array(raster),
            mask_array,
            kernel,
        )
    if resident_gpu_result is not None:
        tmp = resident_gpu_result
        use_gpu = True
    elif use_gpu:
        try:
            import cv2
            gpu_tmp = cv2.UMat(tmp)
            gpu_tmpw = cv2.UMat(tmpw)
            k_float = kernel.astype(numpy.float32)
            gpu_tmp = cv2.sepFilter2D(gpu_tmp, -1, k_float, k_float, borderType=cv2.BORDER_CONSTANT)
            gpu_tmpw = cv2.sepFilter2D(gpu_tmpw, -1, k_float, k_float, borderType=cv2.BORDER_CONSTANT)
            tmp = gpu_tmp.get()
            tmpw = gpu_tmpw.get()
        except Exception as e:
            UI.vprint(2, f"GPU DEM smoothing fallback due to error: {str(e)}")
            use_gpu = False

    if not use_gpu:
        for i in range(0, len(tmp)):
            tmp[i] = numpy.convolve(tmp[i], kernel)[pix_width:-pix_width]
            tmpw[i] = numpy.convolve(tmpw[i], kernel)[pix_width:-pix_width]
        tmp = tmp.transpose()
        tmpw = tmpw.transpose()
        for i in range(0, len(tmp)):
            tmp[i] = numpy.convolve(tmp[i], kernel)[pix_width:-pix_width]
            tmpw[i] = numpy.convolve(tmpw[i], kernel)[pix_width:-pix_width]
        tmp = tmp.transpose()
        tmpw = tmpw.transpose()
    if resident_gpu_result is None:
        tmp[mask_array != 0] = (
            mask_array[mask_array != 0]
            * tmp[mask_array != 0]
            / tmpw[mask_array != 0]
            + (1 - mask_array[mask_array != 0]) * raster[mask_array != 0]
        )
    if preserve_boundary:
        for i in range(pix_width):
            tmp[i] = (
                i / pix_width * tmp[i] + (pix_width - i) / pix_width * raster[i]
            )
            tmp[-i - 1] = (
                i / pix_width * tmp[-i - 1]
                + (pix_width - i) / pix_width * raster[-i - 1]
            )
        for i in range(pix_width):
            tmp[:, i] = (
                i / pix_width * tmp[:, i]
                + (pix_width - i) / pix_width * raster[:, i]
            )
            tmp[:, -i - 1] = (
                i / pix_width * tmp[:, -i - 1]
                + (pix_width - i) / pix_width * raster[:, -i - 1]
            )
    return raster * (mask_array == 0) + tmp * (mask_array != 0)
