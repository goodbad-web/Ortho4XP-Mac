from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import re
import time
import zipfile

import numpy as np


# ============================================================
# Configuration
# ============================================================

INPUT = Path(
    "/Users/hiroshi/Downloads/20260917235514638-001"
)

OUTPUT = Path(
    "/Users/hiroshi/Developer/Ortho4XP-Mac/"
    "Elevation_data/JapanDEM5m/N34E132_GSI_5m.tif"
)

# Target tile: N34E132
LAT_S = 34.0
LAT_N = 35.0
LON_W = 132.0
LON_E = 133.0

# GSI DEM5A is approximately a 5 m class product. 0.2 arc-second
# gives an exact 18,000 x 18,000 EPSG:4326 raster over one degree.
RES_ARCSEC = 0.2
RES_DEG = RES_ARCSEC / 3600.0
WIDTH = int(round((LON_E - LON_W) / RES_DEG))
HEIGHT = int(round((LAT_N - LAT_S) / RES_DEG))

# DEM10B -> base, DEM5A -> override.
CPU_COUNT = max(1, os.cpu_count() or 1)
MAX_WORKERS = min(12, CPU_COUNT)
USE_PROCESSES = True

# Previous 1 arc-second mask inspection showed the remaining GSI no-data
# region for this tile to be sea. Keep this explicit so it is easy to disable.
FILL_REMAINING_NODATA_WITH_SEA_LEVEL = True
SEA_LEVEL_M = 0.0

# A small missing-data preview instead of a gigantic 18k x 18k PNG.
MASK_PREVIEW_STRIDE = 10
MISSING_PREVIEW = Path(
    "/Users/hiroshi/Developer/Ortho4XP-Mac/missing_dem_5m_preview.png"
)

# Full output grid, north-up:
# row 0 = north, col 0 = west.
out = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)


# ============================================================
# Fast GSI GML parser
# ============================================================

RE_LOWER = re.compile(
    rb"<gml:lowerCorner[^>]*>\s*([^<]+?)\s*</gml:lowerCorner>"
)
RE_UPPER = re.compile(
    rb"<gml:upperCorner[^>]*>\s*([^<]+?)\s*</gml:upperCorner>"
)
RE_HIGH = re.compile(
    rb"<gml:high[^>]*>\s*([^<]+?)\s*</gml:high>"
)
RE_START = re.compile(
    rb"<gml:startPoint[^>]*>\s*([^<]+?)\s*</gml:startPoint>"
)
RE_TUPLES = re.compile(
    rb"<gml:tupleList[^>]*>\s*(.*?)\s*</gml:tupleList>",
    re.DOTALL,
)


def read_xml(xml_bytes):
    """Read one GSI DEM XML and return its bounds plus float32 array."""
    m_lower = RE_LOWER.search(xml_bytes)
    m_upper = RE_UPPER.search(xml_bytes)
    m_high = RE_HIGH.search(xml_bytes)
    m_start = RE_START.search(xml_bytes)
    m_tuples = RE_TUPLES.search(xml_bytes)

    if not (m_lower and m_upper and m_high and m_tuples):
        return None

    lat0, lon0 = map(float, m_lower.group(1).split())
    lat1, lon1 = map(float, m_upper.group(1).split())
    hx, hy = map(int, m_high.group(1).split())

    nx = hx + 1
    ny = hy + 1

    if m_start:
        sx, sy = map(int, m_start.group(1).split())
    else:
        sx, sy = 0, 0

    arr = np.full((ny, nx), np.nan, dtype=np.float32)

    block = m_tuples.group(1).decode("utf-8", errors="ignore")
    values = []
    append = values.append

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        pos = line.rfind(",")
        if pos < 0:
            continue

        kind = line[:pos].strip()
        value_text = line[pos + 1 :].strip()

        try:
            z = float(value_text)
        except ValueError:
            z = np.nan

        if kind in ("データなし", "データ無し") or z <= -9990:
            z = np.nan

        append(z)

    if not values:
        return None

    values = np.asarray(values, dtype=np.float32)
    start_index = sy * nx + sx
    flat = arr.ravel()
    available = flat.size - start_index

    if available <= 0:
        return None

    n = min(values.size, available)
    flat[start_index : start_index + n] = values[:n]

    return lat0, lon0, lat1, lon1, arr


# ============================================================
# Resampling / insertion
# ============================================================


def insert_tile(lat0, lon0, lat1, lon1, src):
    """
    Resample one GSI DEM block onto the 0.2 arc-second target raster.

    Source values are treated as pixel-centre samples, matching the existing
    HGT converter logic. Nearest neighbour preserves the measured source
    elevations and avoids inventing intermediate terrain values.
    """
    ny, nx = src.shape
    if nx <= 0 or ny <= 0:
        return 0

    if (
        lat1 <= LAT_S
        or lat0 >= LAT_N
        or lon1 <= LON_W
        or lon0 >= LON_E
    ):
        return 0

    dlon = (lon1 - lon0) / nx
    dlat = (lat1 - lat0) / ny
    if dlon <= 0 or dlat <= 0:
        return 0

    # Output raster uses pixel areas. Pixel centres are:
    # lon = LON_W + (col + 0.5) * RES_DEG
    # lat = LAT_N - (row + 0.5) * RES_DEG
    c0 = max(0, int(np.floor((lon0 - LON_W) / RES_DEG)))
    c1 = min(WIDTH - 1, int(np.ceil((lon1 - LON_W) / RES_DEG)) - 1)
    r0 = max(0, int(np.floor((LAT_N - lat1) / RES_DEG)))
    r1 = min(HEIGHT - 1, int(np.ceil((LAT_N - lat0) / RES_DEG)) - 1)

    if c1 < c0 or r1 < r0:
        return 0

    cols = np.arange(c0, c1 + 1, dtype=np.int32)
    rows = np.arange(r0, r1 + 1, dtype=np.int32)

    target_lons = LON_W + (cols.astype(np.float64) + 0.5) * RES_DEG
    target_lats = LAT_N - (rows.astype(np.float64) + 0.5) * RES_DEG

    xi_float = (target_lons - lon0) / dlon - 0.5
    yi_float = (lat1 - target_lats) / dlat - 0.5

    valid_x = (xi_float >= -0.5) & (xi_float <= nx - 0.5)
    valid_y = (yi_float >= -0.5) & (yi_float <= ny - 0.5)

    xi = np.rint(xi_float).astype(np.int32)
    yi = np.rint(yi_float).astype(np.int32)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    sampled = src[np.ix_(yi, xi)]
    good = valid_y[:, None] & valid_x[None, :] & np.isfinite(sampled)

    if not np.any(good):
        return 0

    dst = out[np.ix_(rows, cols)]
    dst[good] = sampled[good]
    out[np.ix_(rows, cols)] = dst

    return int(np.count_nonzero(good))


# ============================================================
# DEM priority / ZIP parsing
# ============================================================


def dem_priority(path):
    name = path.name.upper()
    if "DEM10B" in name:
        return 0
    if "DEM5A" in name:
        return 1
    return -1


def process_zip(zpath):
    """Parse one ZIP. No writes to the global output array occur here."""
    results = []
    errors = []

    try:
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".xml"):
                    continue

                try:
                    result = read_xml(zf.read(name))
                except Exception as exc:
                    errors.append((name, str(exc)))
                    continue

                if result is None:
                    errors.append((name, "unsupported or incomplete XML"))
                    continue

                lat0, lon0, lat1, lon1, _ = result
                if (
                    lat1 <= LAT_S
                    or lat0 >= LAT_N
                    or lon1 <= LON_W
                    or lon0 >= LON_E
                ):
                    continue

                results.append(result)

    except zipfile.BadZipFile:
        return zpath, [], [(str(zpath), "BAD ZIP")]
    except Exception as exc:
        return zpath, [], [(str(zpath), str(exc))]

    return zpath, results, errors


def process_group(zip_files, kind):
    if not zip_files:
        print(f"\n{kind}: no ZIP files found.")
        return 0, 0

    executor_cls = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor

    print()
    print("=" * 70)
    print(
        f"{kind}: {len(zip_files)} ZIP files, "
        f"{MAX_WORKERS} {'proc' if USE_PROCESSES else 'workers'}"
    )
    print("=" * 70)

    xml_count = 0
    error_count = 0
    completed = 0
    started = time.perf_counter()

    with executor_cls(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(process_zip, path): path for path in zip_files
        }

        for future in as_completed(future_map):
            zpath = future_map[future]
            completed += 1

            try:
                _, results, errors = future.result()
            except Exception as exc:
                print(f"\nERROR {zpath.name}: {exc}")
                error_count += 1
                continue

            # Only the main process writes to `out`, so overlapping DEM tiles
            # cannot race each other.
            for lat0, lon0, lat1, lon1, arr in results:
                insert_tile(lat0, lon0, lat1, lon1, arr)
                xml_count += 1

            if errors:
                error_count += len(errors)
                for name, msg in errors[:3]:
                    print(f"\n  ERROR: {zpath.name} / {name}: {msg}")

            if completed % 10 == 0 or completed == len(zip_files):
                elapsed = time.perf_counter() - started
                print(
                    f"\r{kind}: {completed}/{len(zip_files)} ZIP"
                    f" | XML {xml_count} | {elapsed:.1f}s",
                    end="",
                    flush=True,
                )

    print()
    elapsed = time.perf_counter() - started
    print(
        f"{kind} finished: {xml_count} XML, "
        f"{error_count} errors, {elapsed:.2f}s"
    )
    return xml_count, error_count


# ============================================================
# Diagnostics
# ============================================================


def write_missing_png(mask, path):
    """Write a small grayscale PNG: black=DEM exists, white=missing."""
    import struct
    import zlib

    img = np.where(mask, 255, 0).astype(np.uint8)
    height, width = img.shape

    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(height))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0),
    )
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def grid_stats(label):
    valid = np.isfinite(out)
    valid_count = int(valid.sum())
    total = out.size
    missing = total - valid_count
    coverage = valid_count / total * 100.0

    print(f"\n{label}")
    print(f"  Coverage: {valid_count:,} / {total:,} ({coverage:.8f}%)")

    if valid_count:
        vals = out[valid]
        print(f"  Min:  {float(np.min(vals)):.3f} m")
        print(f"  Max:  {float(np.max(vals)):.3f} m")
        print(f"  Mean: {float(np.mean(vals)):.3f} m")

    return valid_count, missing, coverage


# ============================================================
# GeoTIFF output
# ============================================================


def write_geotiff(path):
    try:
        from osgeo import gdal, osr
    except ImportError as exc:
        raise SystemExit(
            "ERROR: GDAL Python bindings are required.\n"
            "Ortho4XP already uses GDAL for GeoTIFF custom DEMs, so run this "
            "script with the same Python environment that provides osgeo.gdal."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.tif")

    if tmp.exists():
        tmp.unlink()

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise SystemExit("ERROR: GDAL GTiff driver is unavailable.")

    ds = driver.Create(
        str(tmp),
        WIDTH,
        HEIGHT,
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

    if ds is None:
        raise SystemExit(f"ERROR: Could not create {tmp}")

    # Exact one-degree extent. GDAL geotransform is referenced to the
    # upper-left pixel EDGE; Ortho4XP will correctly derive pixel centres.
    ds.SetGeoTransform(
        (
            LON_W,
            RES_DEG,
            0.0,
            LAT_N,
            0.0,
            -RES_DEG,
        )
    )

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)

    # Write in strips so GDAL does not need a second full-size copy.
    block_rows = 512
    for yoff in range(0, HEIGHT, block_rows):
        rows = min(block_rows, HEIGHT - yoff)
        band.WriteArray(out[yoff : yoff + rows], xoff=0, yoff=yoff)

    band.FlushCache()
    ds.FlushCache()
    ds = None

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise SystemExit("ERROR: GeoTIFF temporary file was not written.")

    # Verify the file can be reopened and its spatial metadata is correct.
    check = gdal.Open(str(tmp), gdal.GA_ReadOnly)
    if check is None:
        raise SystemExit("ERROR: GDAL could not reopen the generated GeoTIFF.")

    if check.RasterXSize != WIDTH or check.RasterYSize != HEIGHT:
        raise SystemExit(
            f"ERROR: unexpected raster size: "
            f"{check.RasterXSize}x{check.RasterYSize}"
        )

    gt = check.GetGeoTransform()
    check = None

    expected_gt = (LON_W, RES_DEG, 0.0, LAT_N, 0.0, -RES_DEG)
    if any(abs(a - b) > 1e-12 for a, b in zip(gt, expected_gt)):
        raise SystemExit(f"ERROR: unexpected geotransform: {gt}")

    tmp.replace(path)


# ============================================================
# Main
# ============================================================


def main():
    started = time.perf_counter()

    print("Input :", INPUT)
    print("Output:", OUTPUT)
    print(
        f"Grid  : {WIDTH} x {HEIGHT} pixels, "
        f"{RES_ARCSEC:.1f} arc-sec (~5 m class), EPSG:4326"
    )
    print(f"RAM   : output array ~{out.nbytes / (1024**3):.2f} GiB")

    if not INPUT.exists():
        raise SystemExit(f"Input directory not found: {INPUT}")

    all_zip_files = sorted(INPUT.rglob("*.zip"))
    dem_zip_files = [p for p in all_zip_files if dem_priority(p) >= 0]
    dem10_files = [p for p in dem_zip_files if dem_priority(p) == 0]
    dem5a_files = [p for p in dem_zip_files if dem_priority(p) == 1]

    print()
    print("DEM10B ZIP files:", len(dem10_files))
    print("DEM5A  ZIP files:", len(dem5a_files))

    if not dem10_files:
        raise SystemExit("ERROR: No DEM10B ZIP files found.")
    if not dem5a_files:
        print("WARNING: No DEM5A ZIP files found; output will be DEM10B-derived.")

    # Stage 1: coarse/base DEM.
    count10, errors10 = process_group(dem10_files, "DEM10B")
    grid_stats("Coverage after DEM10B:")

    # Stage 2: higher-resolution override.
    count5a, errors5a = process_group(dem5a_files, "DEM5A")
    valid_count, missing, coverage = grid_stats("Coverage after DEM5A merge:")

    # Small diagnostic preview before any sea fill.
    preview_mask = np.isnan(out[::MASK_PREVIEW_STRIDE, ::MASK_PREVIEW_STRIDE])
    write_missing_png(preview_mask, MISSING_PREVIEW)
    print("Missing-data preview:", MISSING_PREVIEW)

    print()
    print("=" * 70)
    print("FINAL SOURCE STATISTICS")
    print("=" * 70)
    print("DEM10B XML used:", count10)
    print("DEM5A  XML used:", count5a)
    print("Total XML used :", count10 + count5a)
    print("Errors         :", errors10 + errors5a)
    print("Missing cells  :", f"{missing:,}")
    print("Coverage       :", f"{coverage:.8f}%")

    if errors10 + errors5a:
        print("WARNING: parser errors occurred; inspect the messages above.")

    if missing:
        if not FILL_REMAINING_NODATA_WITH_SEA_LEVEL:
            raise SystemExit(
                "Missing cells remain and automatic sea-level fill is disabled."
            )

        print()
        print(
            f"Filling {missing:,} remaining no-data cells with "
            f"sea level {SEA_LEVEL_M:.1f} m."
        )
        out[np.isnan(out)] = SEA_LEVEL_M

    if not np.all(np.isfinite(out)):
        raise SystemExit("ERROR: non-finite values remain after fill.")

    min_value = float(np.min(out))
    max_value = float(np.max(out))
    mean_value = float(np.mean(out))

    if min_value <= -32000 or max_value >= 32000:
        raise SystemExit(
            f"ERROR: implausible DEM range: min={min_value}, max={max_value}"
        )

    print()
    print(
        f"Final range: min={min_value:.3f} m, "
        f"max={max_value:.3f} m, mean={mean_value:.3f} m"
    )

    print("Writing GeoTIFF...")
    write_geotiff(OUTPUT)

    elapsed = time.perf_counter() - started

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print("Written:", OUTPUT)
    print("Raster :", f"{WIDTH} x {HEIGHT}")
    print("CRS    : EPSG:4326")
    print("Pixel  :", f"{RES_ARCSEC:.1f} arc-sec")
    print("Size   :", f"{OUTPUT.stat().st_size / (1024**3):.2f} GiB on disk")
    print("Elapsed:", f"{elapsed:.2f}s")
    print()
    print("Ortho4XP custom_dem:")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
