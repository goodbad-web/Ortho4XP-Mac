from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import zipfile
import re
import time
import os

import numpy as np


# ============================================================
# Configuration
# ============================================================

INPUT = Path(
    "/Users/hiroshi/Downloads/20260917235514638-001"
)

OUTPUT = Path(
    "/Users/hiroshi/Developer/Ortho4XP-Mac/"
    "Elevation_data/JapanDEM1-hgt/I53/I53/N34E132.hgt"
)

# Target HGT tile:
# N34E132 = latitude 34...35, longitude 132...133
LAT_S = 34.0
LAT_N = 35.0
LON_W = 132.0
LON_E = 133.0

# 1 arc-second HGT
SIZE = 3601
EXPECTED_BYTES = SIZE * SIZE * 2

# M5 Max / auto-tune
CPU_COUNT = max(1, (os.cpu_count() or 1))
# Upper bound to avoid oversubscription if desired
MAX_WORKERS = min(12, CPU_COUNT)

# Use processes for CPU-bound ZIP decompress + regex XML parse
USE_PROCESSES = True

# Output grid
# row 0 = north (35N)
# col 0 = west  (132E)
out = np.full(
    (SIZE, SIZE),
    np.nan,
    dtype=np.float32,
)


# ============================================================
# Fast GML parser
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
    """
    Read one GSI DEM XML.

    Returns:
        (lat0, lon0, lat1, lon1, array)

    or None if required elements are missing.
    """

    m_lower = RE_LOWER.search(xml_bytes)
    m_upper = RE_UPPER.search(xml_bytes)
    m_high = RE_HIGH.search(xml_bytes)
    m_start = RE_START.search(xml_bytes)
    m_tuples = RE_TUPLES.search(xml_bytes)

    if not (
        m_lower
        and m_upper
        and m_high
        and m_tuples
    ):
        return None

    lat0, lon0 = map(
        float,
        m_lower.group(1).split(),
    )

    lat1, lon1 = map(
        float,
        m_upper.group(1).split(),
    )

    hx, hy = map(
        int,
        m_high.group(1).split(),
    )

    nx = hx + 1
    ny = hy + 1

    if m_start:
        sx, sy = map(
            int,
            m_start.group(1).split(),
        )
    else:
        sx, sy = 0, 0

    arr = np.full(
        (ny, nx),
        np.nan,
        dtype=np.float32,
    )

    block = m_tuples.group(1).decode(
        "utf-8",
        errors="ignore",
    )

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
        value_text = line[pos + 1:].strip()

        try:
            z = float(value_text)
        except ValueError:
            z = np.nan

        # GSI:
        # "データなし,-9999." などは欠損として保持
        if kind in (
            "データなし",
            "データ無し",
        ):
            z = np.nan

        elif z <= -9990:
            z = np.nan

        append(z)

    if not values:
        return None

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    start_index = sy * nx + sx
    flat = arr.ravel()

    available = (
        flat.size
        - start_index
    )

    if available <= 0:
        return None

    n = min(
        values.size,
        available,
    )

    flat[
        start_index:start_index + n
    ] = values[:n]

    return (
        lat0,
        lon0,
        lat1,
        lon1,
        arr,
    )


# ============================================================
# Target HGT insertion
# ============================================================

def insert_tile(
    lat0,
    lon0,
    lat1,
    lon1,
    src,
):
    """
    Resample one GSI DEM block onto the 3601 x 3601 HGT grid.

    Nearest-neighbour is used.
    """

    ny, nx = src.shape

    if nx <= 0 or ny <= 0:
        return 0

    # No overlap
    if (
        lat1 <= LAT_S
        or lat0 >= LAT_N
        or lon1 <= LON_W
        or lon0 >= LON_E
    ):
        return 0

    dlon = (
        lon1 - lon0
    ) / nx

    dlat = (
        lat1 - lat0
    ) / ny

    if dlon <= 0 or dlat <= 0:
        return 0

    # Target HGT index range
    c0 = max(
        0,
        int(
            np.floor(
                (lon0 - LON_W)
                * 3600
            )
        ),
    )

    c1 = min(
        3600,
        int(
            np.ceil(
                (lon1 - LON_W)
                * 3600
            )
        ),
    )

    r0 = max(
        0,
        int(
            np.floor(
                (LAT_N - lat1)
                * 3600
            )
        ),
    )

    r1 = min(
        3600,
        int(
            np.ceil(
                (LAT_N - lat0)
                * 3600
            )
        ),
    )

    if c1 < c0 or r1 < r0:
        return 0

    cols = np.arange(
        c0,
        c1 + 1,
        dtype=np.int32,
    )

    rows = np.arange(
        r0,
        r1 + 1,
        dtype=np.int32,
    )

    target_lons = (
        LON_W
        + cols.astype(
            np.float64
        ) / 3600.0
    )

    target_lats = (
        LAT_N
        - rows.astype(
            np.float64
        ) / 3600.0
    )

    # Source cells treated as centres
    xi_float = (
        (target_lons - lon0)
        / dlon
        - 0.5
    )

    yi_float = (
        (lat1 - target_lats)
        / dlat
        - 0.5
    )

    xi = np.rint(
        xi_float
    ).astype(
        np.int32
    )

    yi = np.rint(
        yi_float
    ).astype(
        np.int32
    )

    # Prevent extending data outside source bounds
    valid_x = (
        (xi_float >= -0.5)
        & (
            xi_float
            <= nx - 0.5
        )
    )

    valid_y = (
        (yi_float >= -0.5)
        & (
            yi_float
            <= ny - 0.5
        )
    )

    xi = np.clip(
        xi,
        0,
        nx - 1,
    )

    yi = np.clip(
        yi,
        0,
        ny - 1,
    )

    sampled = src[
        np.ix_(
            yi,
            xi,
        )
    ]

    valid_area = (
        valid_y[:, None]
        & valid_x[None, :]
    )

    good = (
        valid_area
        & np.isfinite(
            sampled
        )
    )

    if not np.any(good):
        return 0

    dst = out[
        np.ix_(
            rows,
            cols,
        )
    ]

    dst[good] = sampled[good]

    out[
        np.ix_(
            rows,
            cols,
        )
    ] = dst

    return int(
        np.count_nonzero(
            good
        )
    )


# ============================================================
# DEM priority
# ============================================================

def dem_priority(path):
    """
    DEM10B first, DEM5A last.

    DEM5A therefore overrides DEM10B.
    """

    name = path.name.upper()

    if "DEM10B" in name:
        return 0

    if "DEM5A" in name:
        return 1

    return -1


# ============================================================
# ZIP processing
# ============================================================

def process_zip(zpath):
    """
    Read/decompress/parse one ZIP.

    Does NOT modify global `out`.
    Safe for threaded use.
    """

    results = []
    errors = []

    try:
        with zipfile.ZipFile(
            zpath
        ) as zf:

            for name in zf.namelist():

                if not name.lower().endswith(
                    ".xml"
                ):
                    continue

                try:
                    xml_bytes = zf.read(
                        name
                    )

                    result = read_xml(
                        xml_bytes
                    )

                except Exception as exc:
                    errors.append(
                        (
                            name,
                            str(exc),
                        )
                    )
                    continue

                if result is None:
                    errors.append(
                        (
                            name,
                            "unsupported or incomplete XML",
                        )
                    )
                    continue

                (
                    lat0,
                    lon0,
                    lat1,
                    lon1,
                    arr,
                ) = result

                if (
                    lat1 <= LAT_S
                    or lat0 >= LAT_N
                    or lon1 <= LON_W
                    or lon0 >= LON_E
                ):
                    continue

                results.append(
                    result
                )

    except zipfile.BadZipFile:
        return (
            zpath,
            [],
            [
                (
                    str(zpath),
                    "BAD ZIP",
                )
            ],
        )

    except Exception as exc:
        return (
            zpath,
            [],
            [
                (
                    str(zpath),
                    str(exc),
                )
            ],
        )

    return (
        zpath,
        results,
        errors,
    )


def process_group(
    zip_files,
    kind,
):
    """
    Parse all ZIPs of one DEM class in parallel.

    Writes to `out` are done only in the main thread.
    """

    if not zip_files:
        print()
        print(
            f"{kind}: no ZIP files found."
        )
        return 0, 0

    print()
    print("=" * 70)

    print(
        f"{kind}: "
        f"{len(zip_files)} ZIP files, "
        f"{MAX_WORKERS} {'proc' if USE_PROCESSES else 'threads'}"
    )

    print("=" * 70)

    xml_count = 0
    error_count = 0
    completed = 0

    started = time.perf_counter()

    executor_cls = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor
    with executor_cls(max_workers=MAX_WORKERS) as executor:

        future_map = {
            executor.submit(
                process_zip,
                path,
            ): path
            for path in zip_files
        }

        for future in as_completed(
            future_map
        ):

            zpath = future_map[
                future
            ]

            completed += 1

            try:
                (
                    _,
                    results,
                    errors,
                ) = future.result()

            except Exception as exc:
                print(
                    f"\nERROR "
                    f"{zpath.name}: "
                    f"{exc}"
                )

                error_count += 1
                continue

            # Write into global grid only here
            for result in results:

                (
                    lat0,
                    lon0,
                    lat1,
                    lon1,
                    arr,
                ) = result

                insert_tile(
                    lat0,
                    lon0,
                    lat1,
                    lon1,
                    arr,
                )

                xml_count += 1

            if errors:
                error_count += len(
                    errors
                )

                for (
                    name,
                    msg,
                ) in errors[:3]:

                    print(
                        f"\n  ERROR: "
                        f"{zpath.name} / "
                        f"{name}: "
                        f"{msg}"
                    )

            # Reduce console overhead
            if (
                completed % 10 == 0
                or completed
                == len(zip_files)
            ):

                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    "\r"
                    f"{kind}: "
                    f"{completed}/"
                    f"{len(zip_files)} ZIP"
                    f" | XML "
                    f"{xml_count}"
                    f" | "
                    f"{elapsed:.1f}s",
                    end="",
                    flush=True,
                )

    print()

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"{kind} finished: "
        f"{xml_count} XML, "
        f"{error_count} errors, "
        f"{elapsed:.2f}s"
    )

    return (
        xml_count,
        error_count,
    )


# ============================================================
# Write missing-data mask PNG function
# ============================================================

def write_missing_png(mask, path):
    """
    Write missing-data mask as grayscale PNG using only stdlib + NumPy.

    black = DEM exists
    white = missing
    """
    import struct
    import zlib

    img = np.where(mask, 255, 0).astype(np.uint8)

    height, width = img.shape

    # PNG scanlines require a filter byte at the start of every row.
    raw = b"".join(
        b"\x00" + img[y].tobytes()
        for y in range(height)
    )

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(
                ">I",
                zlib.crc32(kind + data) & 0xFFFFFFFF,
            )
        )

    png = b"\x89PNG\r\n\x1a\n"

    png += chunk(
        b"IHDR",
        struct.pack(
            ">IIBBBBB",
            width,
            height,
            8,   # bit depth
            0,   # grayscale
            0,
            0,
            0,
        ),
    )

    png += chunk(
        b"IDAT",
        zlib.compress(raw, level=6),
    )

    png += chunk(
        b"IEND",
        b"",
    )

    Path(path).write_bytes(png)


# ============================================================
# Main
# ============================================================

def main():

    start_time = (
        time.perf_counter()
    )

    print(
        "Input :",
        INPUT,
    )

    print(
        "Output:",
        OUTPUT,
    )

    if not INPUT.exists():
        raise SystemExit(
            f"Input directory not found: "
            f"{INPUT}"
        )

    all_zip_files = sorted(
        INPUT.rglob(
            "*.zip"
        )
    )

    dem_zip_files = [
        p
        for p in all_zip_files
        if dem_priority(p) >= 0
    ]

    dem10_files = sorted(
        p
        for p in dem_zip_files
        if dem_priority(p) == 0
    )

    dem5a_files = sorted(
        p
        for p in dem_zip_files
        if dem_priority(p) == 1
    )

    print()

    print(
        "DEM10B ZIP files:",
        len(
            dem10_files
        ),
    )

    print(
        "DEM5A  ZIP files:",
        len(
            dem5a_files
        ),
    )

    if not dem10_files:
        print()
        print(
            "WARNING: "
            "No DEM10B files found."
        )

    if not dem5a_files:
        print()
        print(
            "WARNING: "
            "No DEM5A files found."
        )

    # --------------------------------------------------------
    # Stage 1:
    # DEM10B base
    # --------------------------------------------------------

    (
        count10,
        errors10,
    ) = process_group(
        dem10_files,
        "DEM10B",
    )

    valid10 = np.isfinite(
        out
    )
    missing10 = out.size - valid10.sum()

    print()

    print(
        "Coverage after DEM10B:"
    )

    print(
        f"  "
        f"{valid10.sum():,}"
        f" / "
        f"{out.size:,}"
    )

    print(
        f"  "
        f"{valid10.mean() * 100:.6f}%"
    )

    # Write missing data PNG after DEM10B
    missing_dem10b_path = Path(
        "/Users/hiroshi/Developer/Ortho4XP-Mac/missing_dem10b.png"
    )
    missing_mask10 = np.isnan(out)
    write_missing_png(missing_mask10, missing_dem10b_path)

    print()
    print(
        "Missing map after DEM10B written:",
        missing_dem10b_path,
    )

    # --------------------------------------------------------
    # Stage 2:
    # DEM5A override
    # --------------------------------------------------------

    (
        count5a,
        errors5a,
    ) = process_group(
        dem5a_files,
        "DEM5A",
    )

    # --------------------------------------------------------
    # Statistics for final merged grid
    # --------------------------------------------------------

    valid = np.isfinite(
        out
    )

    valid_count = int(
        valid.sum()
    )

    missing = int(
        out.size
        - valid_count
    )

    coverage = (
        valid_count
        / out.size
        * 100.0
    )

    # Write missing data PNG after final merge
    missing_dem_final_path = Path(
        "/Users/hiroshi/Developer/Ortho4XP-Mac/missing_dem_final.png"
    )
    missing_mask_final = np.isnan(out)
    write_missing_png(missing_mask_final, missing_dem_final_path)

    print()
    print(
        "Missing map after final merge written:",
        missing_dem_final_path,
    )

    print()
    print("=" * 70)
    print(
        "FINAL STATISTICS"
    )
    print("=" * 70)

    print(
        "DEM10B XML used:",
        count10,
    )

    print(
        "DEM5A  XML used:",
        count5a,
    )

    print(
        "Total XML used :",
        count10 + count5a,
    )

    print(
        "Errors         :",
        errors10 + errors5a,
    )

    print()

    print(
        "Coverage:",
        f"{valid_count:,}",
        "/",
        f"{out.size:,}",
    )

    print(
        "Coverage %:",
        f"{coverage:.8f}",
    )

    if valid_count:

        values = out[
            valid
        ]

        print(
            "Min:",
            float(
                np.min(
                    values
                )
            ),
        )

        print(
            "Max:",
            float(
                np.max(
                    values
                )
            ),
        )

        print(
            "Mean:",
            float(
                np.mean(
                    values
                )
            ),
        )

    # --------------------------------------------------------
    # Require complete HGT coverage
    # --------------------------------------------------------

    if missing:

        print()
        print(
            "Missing HGT cells:",
            f"{missing:,}",
        )

        print(
            "Filling remaining no-data cells "
            "with sea level 0 m."
        )

        out[np.isnan(out)] = 0.0

        valid = np.isfinite(out)
        valid_count = int(valid.sum())
        missing = int(out.size - valid_count)

        print(
            "Coverage after sea fill:",
            f"{valid_count:,}",
            "/",
            f"{out.size:,}",
        )

        print(
            "Coverage after sea fill:",
            f"{valid.mean() * 100:.8f}%",
        )

        if missing != 0:
            raise SystemExit(
                f"ERROR: {missing} cells remain missing."
            )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    min_value = float(
        np.min(
            out
        )
    )

    max_value = float(
        np.max(
            out
        )
    )

    if (
        min_value <= -32000
        or max_value >= 32000
    ):

        print()

        print(
            "ERROR: "
            "Implausible DEM values detected:"
        )

        print(
            "Min:",
            min_value,
        )

        print(
            "Max:",
            max_value,
        )

        print(
            "HGT will NOT be written."
        )

        raise SystemExit(3)

    # --------------------------------------------------------
    # Convert to HGT
    # signed 16-bit, big-endian
    # --------------------------------------------------------

    hgt = np.rint(
        out
    )

    hgt = np.clip(
        hgt,
        -32768,
        32767,
    ).astype(
        ">i2"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Temporary file first
    tmp_output = (
        OUTPUT.with_suffix(
            OUTPUT.suffix
            + ".tmp"
        )
    )

    hgt.tofile(
        tmp_output
    )

    actual_size = (
        tmp_output.stat().st_size
    )

    print()

    print(
        "Temporary file:",
        tmp_output,
    )

    print(
        "Size:",
        actual_size,
    )

    print(
        "Expected:",
        EXPECTED_BYTES,
    )

    if (
        actual_size
        != EXPECTED_BYTES
    ):

        print()

        print(
            "ERROR: "
            "Invalid HGT file size."
        )

        try:
            tmp_output.unlink()

        except FileNotFoundError:
            pass

        raise SystemExit(4)

    # Atomic replacement
    tmp_output.replace(
        OUTPUT
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print(
        "SUCCESS"
    )
    print("=" * 70)

    print(
        "Written:",
        OUTPUT,
    )

    print(
        "Size:",
        OUTPUT.stat().st_size,
    )

    print(
        "Expected:",
        EXPECTED_BYTES,
    )

    print(
        f"Elapsed: "
        f"{elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()

