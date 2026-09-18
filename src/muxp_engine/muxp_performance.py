# -*- coding: utf-8 -*-
"""Optional CPU acceleration and performance helpers for MUXP.

The mesh editor itself intentionally remains responsible for mutating the
DSF object.  This module only operates on numeric triangle arrays so that the
accelerated path can be compared with the existing Python implementation and
can be discarded safely if an optional backend fails.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from multiprocessing import cpu_count
from time import perf_counter

import numpy as np

from .muxp_math import PointInPoly, PointLocationInTria


log = getLogger(__name__)

NUMBA_MIN_TRIANGLES = 1024
VALID_BACKENDS = ("auto", "python", "numba")

try:
    import numba
    from numba import njit, prange
except Exception as exc:  # Numba is deliberately optional and may lack a native wheel.
    numba = None
    njit = None
    prange = None
    NUMBA_IMPORT_ERROR = exc
else:
    NUMBA_IMPORT_ERROR = None


class PerformanceBackendError(RuntimeError):
    """Base error raised when a requested performance backend is unavailable."""


class NumbaUnavailableError(PerformanceBackendError):
    """Raised when the explicit Numba backend cannot be used."""


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    selected: str
    workers: int
    reason: str


def numba_available():
    """Return whether the optional Numba module was imported successfully."""

    return numba is not None


def default_worker_count():
    """Return the conservative default used by both CPU backends."""

    return max(1, cpu_count() - 1)


def _numba_worker_limit():
    """Return Numba's configured maximum parallel worker count."""

    if numba is None:
        return None
    try:
        limit = int(numba.config.NUMBA_NUM_THREADS)
    except (AttributeError, TypeError, ValueError):
        limit = cpu_count()
    return max(1, limit)


def normalize_worker_count(value):
    """Validate a worker setting and return ``None`` for ``auto``."""

    if value is None:
        return default_worker_count()
    if isinstance(value, str) and value.strip().lower() == "auto":
        return default_worker_count()
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("performanceWorkers must be auto or a positive integer") from exc
    if workers < 1 or workers > cpu_count():
        raise ValueError("performanceWorkers must be between 1 and the logical CPU count")
    return workers


def resolve_backend(mode="auto", triangle_count=0, workers=None):
    """Resolve the requested backend without importing or compiling kernels."""

    requested = str(mode).strip().lower()
    if requested not in VALID_BACKENDS:
        raise ValueError("performanceBackend must be auto, python, or numba")
    worker_count = normalize_worker_count(workers)

    if requested == "python":
        return BackendInfo(requested, "python", worker_count, "python backend requested")
    if requested == "numba":
        if not numba_available():
            raise NumbaUnavailableError(
                "Numba backend was requested but Numba is unavailable. "
                "Install it with: python -m pip install -r requirements-accelerated.txt"
            )
        return BackendInfo(
            requested,
            "numba",
            min(worker_count, _numba_worker_limit()),
            "numba backend requested",
        )
    if numba_available() and triangle_count >= NUMBA_MIN_TRIANGLES:
        return BackendInfo(
            requested,
            "numba",
            min(worker_count, _numba_worker_limit()),
            "triangle threshold reached",
        )
    if not numba_available():
        return BackendInfo(requested, "python", worker_count, "Numba is unavailable")
    return BackendInfo(requested, "python", worker_count, "below Numba triangle threshold")


def triangles_to_array(triangles):
    """Copy triangle lon/lat/elevation coordinates into a contiguous array."""

    if len(triangles) == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    result = np.empty((len(triangles), 3, 3), dtype=np.float64)
    for triangle_index, triangle in enumerate(triangles):
        if len(triangle) < 3:
            raise ValueError("Each triangle must contain at least three vertices")
        for vertex_index in range(3):
            if len(triangle[vertex_index]) < 3:
                raise ValueError("Each vertex must contain lon, lat, and elevation")
            result[triangle_index, vertex_index, :] = triangle[vertex_index][:3]
    return result


def apply_triangle_elevations(triangles, vertices):
    """Apply only the numeric elevation column back to existing triangle lists."""

    array = _validate_vertices(vertices)
    if array.shape[0] != len(triangles):
        raise ValueError("Triangle array length does not match triangle list")
    for triangle_index, triangle in enumerate(triangles):
        for vertex_index in range(3):
            triangle[vertex_index][2] = float(array[triangle_index, vertex_index, 2])


def _validate_vertices(vertices):
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError("vertices must have shape (N, 3, 3)")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _validate_polygon(polygon):
    array = np.asarray(polygon, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError("polygon must have shape (M, 2) with at least two points")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _validate_ramp(ramp_tria):
    array = np.asarray(ramp_tria, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError("ramp_tria must have shape (3, 3)")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _update_elevation_python(vertices, polygon, elevation):
    result = vertices.copy()
    polygon_list = polygon.tolist()
    for triangle_index in range(result.shape[0]):
        for vertex_index in range(3):
            if PointInPoly(result[triangle_index, vertex_index, :2], polygon_list):
                result[triangle_index, vertex_index, 2] = elevation
    return result


def _update_ramp_python(vertices, polygon, ramp_tria):
    result = vertices.copy()
    polygon_list = polygon.tolist()
    ramp_list = ramp_tria.tolist()
    for triangle_index in range(result.shape[0]):
        for vertex_index in range(3):
            vertex = result[triangle_index, vertex_index]
            if PointInPoly(vertex[:2], polygon_list):
                l0, l1 = PointLocationInTria(vertex[:2], ramp_list)
                vertex[2] = ramp_list[2][2] + l0 * (ramp_list[0][2] - ramp_list[2][2]) + l1 * (
                    ramp_list[1][2] - ramp_list[2][2]
                )
    return result


if numba is not None:

    @njit(cache=True, fastmath=False)
    def _point_in_polygon_numba(x, y, polygon):
        inside = False
        for index in range(polygon.shape[0] - 1):
            x1 = polygon[index, 0]
            y1 = polygon[index, 1]
            x2 = polygon[index + 1, 0]
            y2 = polygon[index + 1, 1]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside


    @njit(cache=True, parallel=True, fastmath=False)
    def _update_elevation_numba(vertices, polygon, elevation):
        result = vertices.copy()
        p_min_x = polygon[0, 0]
        p_max_x = polygon[0, 0]
        p_min_y = polygon[0, 1]
        p_max_y = polygon[0, 1]
        for polygon_index in range(1, polygon.shape[0]):
            p_min_x = min(p_min_x, polygon[polygon_index, 0])
            p_max_x = max(p_max_x, polygon[polygon_index, 0])
            p_min_y = min(p_min_y, polygon[polygon_index, 1])
            p_max_y = max(p_max_y, polygon[polygon_index, 1])
        for triangle_index in prange(result.shape[0]):
            t_min_x = min(result[triangle_index, 0, 0], result[triangle_index, 1, 0], result[triangle_index, 2, 0])
            t_max_x = max(result[triangle_index, 0, 0], result[triangle_index, 1, 0], result[triangle_index, 2, 0])
            t_min_y = min(result[triangle_index, 0, 1], result[triangle_index, 1, 1], result[triangle_index, 2, 1])
            t_max_y = max(result[triangle_index, 0, 1], result[triangle_index, 1, 1], result[triangle_index, 2, 1])
            if t_max_x < p_min_x or t_min_x > p_max_x or t_max_y < p_min_y or t_min_y > p_max_y:
                continue
            for vertex_index in range(3):
                if _point_in_polygon_numba(
                    result[triangle_index, vertex_index, 0],
                    result[triangle_index, vertex_index, 1],
                    polygon,
                ):
                    result[triangle_index, vertex_index, 2] = elevation
        return result


    @njit(cache=True, fastmath=False)
    def _point_location_numba(x, y, ramp_tria):
        denom = ((ramp_tria[1, 1] - ramp_tria[2, 1]) * (ramp_tria[0, 0] - ramp_tria[2, 0]) +
                 (ramp_tria[2, 0] - ramp_tria[1, 0]) * (ramp_tria[0, 1] - ramp_tria[2, 1]))
        if denom == 0:
            return -0.01, -0.01
        nom_a = ((ramp_tria[1, 1] - ramp_tria[2, 1]) * (x - ramp_tria[2, 0]) +
                 (ramp_tria[2, 0] - ramp_tria[1, 0]) * (y - ramp_tria[2, 1]))
        nom_b = ((ramp_tria[2, 1] - ramp_tria[0, 1]) * (x - ramp_tria[2, 0]) +
                 (ramp_tria[0, 0] - ramp_tria[2, 0]) * (y - ramp_tria[2, 1]))
        return nom_a / denom, nom_b / denom


    @njit(cache=True, parallel=True, fastmath=False)
    def _update_ramp_numba(vertices, polygon, ramp_tria):
        result = vertices.copy()
        p_min_x = polygon[0, 0]
        p_max_x = polygon[0, 0]
        p_min_y = polygon[0, 1]
        p_max_y = polygon[0, 1]
        for polygon_index in range(1, polygon.shape[0]):
            p_min_x = min(p_min_x, polygon[polygon_index, 0])
            p_max_x = max(p_max_x, polygon[polygon_index, 0])
            p_min_y = min(p_min_y, polygon[polygon_index, 1])
            p_max_y = max(p_max_y, polygon[polygon_index, 1])
        for triangle_index in prange(result.shape[0]):
            t_min_x = min(result[triangle_index, 0, 0], result[triangle_index, 1, 0], result[triangle_index, 2, 0])
            t_max_x = max(result[triangle_index, 0, 0], result[triangle_index, 1, 0], result[triangle_index, 2, 0])
            t_min_y = min(result[triangle_index, 0, 1], result[triangle_index, 1, 1], result[triangle_index, 2, 1])
            t_max_y = max(result[triangle_index, 0, 1], result[triangle_index, 1, 1], result[triangle_index, 2, 1])
            if t_max_x < p_min_x or t_min_x > p_max_x or t_max_y < p_min_y or t_min_y > p_max_y:
                continue
            for vertex_index in range(3):
                x = result[triangle_index, vertex_index, 0]
                y = result[triangle_index, vertex_index, 1]
                if _point_in_polygon_numba(x, y, polygon):
                    l0, l1 = _point_location_numba(x, y, ramp_tria)
                    result[triangle_index, vertex_index, 2] = (
                        ramp_tria[2, 2]
                        + l0 * (ramp_tria[0, 2] - ramp_tria[2, 2])
                        + l1 * (ramp_tria[1, 2] - ramp_tria[2, 2])
                    )
        return result


def _set_numba_threads(workers):
    if numba is not None:
        effective_workers = min(workers, _numba_worker_limit())
        numba.set_num_threads(effective_workers)
        return effective_workers
    return workers


def update_elevation_array(vertices, polygon, elevation, backend="python", workers=None):
    """Return an updated triangle array using the selected backend."""

    result = _validate_vertices(vertices)
    polygon_array = _validate_polygon(polygon)
    info = resolve_backend(backend, result.shape[0], workers)
    if info.selected == "python":
        return _update_elevation_python(result, polygon_array, float(elevation))
    _set_numba_threads(info.workers)
    return _update_elevation_numba(result, polygon_array, float(elevation))


def update_ramp_array(vertices, polygon, ramp_tria, backend="python", workers=None):
    """Return an updated triangle array using the selected ramp backend."""

    result = _validate_vertices(vertices)
    polygon_array = _validate_polygon(polygon)
    ramp_array = _validate_ramp(ramp_tria)
    info = resolve_backend(backend, result.shape[0], workers)
    if info.selected == "python":
        return _update_ramp_python(result, polygon_array, ramp_array)
    _set_numba_threads(info.workers)
    return _update_ramp_numba(result, polygon_array, ramp_array)


@contextmanager
def measure_phase(logger, phase, detailed=False, **details):
    """Log a phase duration, optionally including detailed metadata."""

    started = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - started) * 1000.0
        message = "[performance] phase={} elapsed_ms={:.3f}".format(phase, elapsed_ms)
        if detailed and details:
            detail_text = " ".join("{}={}".format(key, details[key]) for key in sorted(details))
            message += " " + detail_text
        logger.info(message)

