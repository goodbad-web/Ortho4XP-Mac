"""Format-aware Orthophotos cache helpers and migration operations.

The cache identity is the filename stem.  JPEG remains the legacy format,
while WebP is an optional sibling file selected without changing provider
requests or any X-Plane output format.
"""

import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy
from PIL import Image, features

import O4_File_Names as FNAMES
import O4_RAMDisk_Utils


CACHE_FORMATS = ("jpg", "webp")
DEFAULT_CACHE_FORMAT = "jpg"
WEBP_QUALITY_MIN = 80
WEBP_QUALITY_MAX = 100
MIN_PSNR_DB = 45.0
MAX_MAE_8BIT = 1.0


def normalize_cache_format(value):
    normalized = str(value if value is not None else "").strip().lower()
    if normalized == "jpeg":
        normalized = "jpg"
    return normalized


def validate_cache_settings(cache_format=DEFAULT_CACHE_FORMAT, quality=""):
    """Validate global cache settings and return ``(format, quality)``.

    JPEG accepts an empty quality value for backward compatibility.  WebP
    deliberately requires an explicit integer quality in the supported range.
    """
    normalized = normalize_cache_format(cache_format)
    if normalized not in CACHE_FORMATS:
        raise ValueError("imagery_cache_format must be jpg or webp")
    if normalized == "jpg":
        return normalized, None
    if quality is None or str(quality).strip() == "":
        raise ValueError(
            "imagery_cache_quality is required when imagery_cache_format=webp"
        )
    try:
        parsed_quality = int(str(quality).strip())
    except (TypeError, ValueError):
        raise ValueError("imagery_cache_quality must be an integer from 80 to 100")
    if not WEBP_QUALITY_MIN <= parsed_quality <= WEBP_QUALITY_MAX:
        raise ValueError("imagery_cache_quality must be an integer from 80 to 100")
    if not features.check("webp"):
        raise ValueError("Pillow WebP support is unavailable")
    return normalized, parsed_quality


def cache_file_names(til_x_left, til_y_top, zoomlevel, provider_code):
    return FNAMES.imagery_file_names_from_attributes(
        til_x_left, til_y_top, zoomlevel, provider_code
    )


def cache_paths(file_dir, til_x_left, til_y_top, zoomlevel, provider_code):
    return [
        os.path.join(file_dir, name)
        for name in cache_file_names(
            til_x_left, til_y_top, zoomlevel, provider_code
        )
    ]


def image_file_is_ready(path):
    """Validate an image and attempt the existing RAM-disk restoration path."""
    if not path:
        return False
    if not os.path.exists(path):
        O4_RAMDisk_Utils.check_and_restore_cached_image(path)
    if not os.path.isfile(path):
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        if O4_RAMDisk_Utils.check_and_restore_cached_image(path):
            try:
                with Image.open(path) as image:
                    image.verify()
                return True
            except Exception:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return False


def find_cache_path(file_dir, til_x_left, til_y_top, zoomlevel, provider_code):
    """Return the first valid WebP/JPEG candidate, in that order."""
    for path in cache_paths(
        file_dir, til_x_left, til_y_top, zoomlevel, provider_code
    ):
        if image_file_is_ready(path):
            return path
    return None


def preferred_cache_path(
    file_dir, til_x_left, til_y_top, zoomlevel, provider_code, cache_format
):
    normalized = normalize_cache_format(cache_format)
    if normalized not in CACHE_FORMATS:
        raise ValueError(f"unsupported imagery cache format: {cache_format}")
    name = FNAMES.imagery_file_name_from_attributes(
        til_x_left, til_y_top, zoomlevel, provider_code, normalized
    )
    return os.path.join(file_dir, name)


def _atomic_save(image, file_path, image_format, quality=None):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix="." + os.path.basename(file_path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(file_path) or ".",
    )
    os.close(fd)
    try:
        save_image = image
        if image_format in ("JPEG", "WEBP"):
            save_image = image.convert("RGB")
        save_kwargs = {"format": image_format}
        if image_format == "WEBP":
            save_kwargs.update(quality=int(quality), method=6)
        save_image.save(temporary_path, **save_kwargs)
        if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
            raise OSError("image save produced an empty file")
        os.replace(temporary_path, file_path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def save_cache_image(image, file_path, cache_format, quality=None):
    normalized, parsed_quality = validate_cache_settings(cache_format, quality)
    if normalized == "webp" and parsed_quality is None:
        raise ValueError("WebP cache quality is required")
    _atomic_save(
        image,
        file_path,
        "WEBP" if normalized == "webp" else "JPEG",
        parsed_quality,
    )


def prepare_external_image_input(source_path, tmp_dir):
    """Decode WebP to a temporary PNG for external DDS/upscale tools.

    JPEG and other formats are returned unchanged.  The caller owns the
    returned temporary file and must remove it after the external operation.
    """
    if not source_path or Path(source_path).suffix.lower() != ".webp":
        return source_path, False
    os.makedirs(tmp_dir, exist_ok=True)
    fd, output_path = tempfile.mkstemp(
        prefix="ortho4xp-webp-", suffix=".png", dir=tmp_dir
    )
    os.close(fd)
    try:
        with Image.open(source_path) as image:
            image.load()
            normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            normalized.save(output_path, format="PNG")
        return output_path, True
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise


def _tile_tokens(selector):
    selector = str(selector).strip()
    match = re.fullmatch(r"([+-]\d{1,3})([+-]\d{1,3})", selector)
    if not match:
        raise ValueError(
            f"invalid tile selector '{selector}'; expected e.g. +34+132"
        )
    lat = int(match.group(1))
    lon = int(match.group(2))
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"tile selector is outside valid coordinates: {selector}")
    return {
        FNAMES.short_latlon(lat, lon),
        FNAMES.long_latlon(lat, lon).replace(os.sep, "/"),
    }


def _matches_filters(path, root, tiles, providers, zoomlevels):
    relative = path.relative_to(root).as_posix()
    if tiles:
        tile_matches = False
        for selector in tiles:
            tokens = _tile_tokens(selector)
            if any(token in relative for token in tokens):
                tile_matches = True
                break
        if not tile_matches:
            return False
    stem = path.stem
    if providers:
        provider_matches = False
        for provider in providers:
            provider = str(provider).strip()
            if re.search(r"_" + re.escape(provider) + r"\d+$", stem) or any(
                part == provider or part.startswith(provider + "_")
                for part in Path(relative).parts[:-1]
            ):
                provider_matches = True
                break
        if not provider_matches:
            return False
    if zoomlevels:
        zoom_tokens = {str(int(z)) for z in zoomlevels}
        if not re.search(
            r"(?:" + "|".join(zoom_tokens) + r")$",
            stem,
        ) and not any(
            part.rsplit("_", 1)[-1] in zoom_tokens
            for part in Path(relative).parts[:-1]
        ):
            return False
    return True


def iter_jpeg_cache_files(root=None, tiles=(), providers=(), zoomlevels=()):
    root_path = Path(root or FNAMES.Imagery_dir)
    if not root_path.is_dir():
        return []
    return sorted(
        path
        for path in root_path.rglob("*.jpg")
        if path.is_file()
        and _matches_filters(path, root_path, tiles, providers, zoomlevels)
    )


def _comparison_metrics(jpeg_path, webp_path):
    with Image.open(jpeg_path) as jpeg_image:
        jpeg_image.load()
        source = jpeg_image.convert("RGB")
    with Image.open(webp_path) as webp_image:
        webp_image.load()
        target = webp_image.convert("RGB")
    if source.size != target.size:
        return {
            "dimensions": list(source.size),
            "webp_dimensions": list(target.size),
            "psnr_db": None,
            "mae_8bit": None,
            "mae_normalized": None,
            "reason": "dimension_mismatch",
        }
    source_array = numpy.asarray(source, dtype=numpy.float32)
    target_array = numpy.asarray(target, dtype=numpy.float32)
    difference = source_array - target_array
    mse = float(numpy.mean(numpy.square(difference)))
    mae_8bit = float(numpy.mean(numpy.abs(difference)))
    psnr_db = "inf" if mse == 0 else float(10 * numpy.log10((255.0**2) / mse))
    return {
        "dimensions": list(source.size),
        "webp_dimensions": list(target.size),
        "psnr_db": psnr_db,
        "mae_8bit": mae_8bit,
        "mae_normalized": mae_8bit / 255.0,
        "reason": None,
    }


def _gate_result(jpeg_path, webp_path, metrics):
    source_size = os.path.getsize(jpeg_path)
    target_size = os.path.getsize(webp_path)
    reasons = []
    if metrics.get("reason"):
        reasons.append(metrics["reason"])
    if (
        isinstance(metrics.get("psnr_db"), (int, float))
        and metrics["psnr_db"] < MIN_PSNR_DB
    ):
        reasons.append("psnr_below_threshold")
    if (
        metrics.get("mae_8bit") is not None
        and metrics["mae_8bit"] > MAX_MAE_8BIT
    ):
        reasons.append("mae_above_threshold")
    if target_size >= source_size:
        reasons.append("webp_not_smaller")
    return {
        "accepted": not reasons,
        "source_size": source_size,
        "target_size": target_size,
        "savings_bytes": source_size - target_size,
        **metrics,
        "reason": ";".join(reasons) if reasons else "verified",
    }


def _base_result(jpeg_path):
    webp_path = jpeg_path.with_suffix(".webp")
    return {
        "jpeg": str(jpeg_path),
        "webp": str(webp_path),
        "status": "error",
        "reason": None,
        "source_size": os.path.getsize(jpeg_path) if jpeg_path.is_file() else None,
        "target_size": os.path.getsize(webp_path) if webp_path.is_file() else None,
        "savings_bytes": None,
        "dimensions": None,
        "webp_dimensions": None,
        "psnr_db": None,
        "mae_8bit": None,
        "mae_normalized": None,
    }


def _convert_one(jpeg_path, quality, force, dry_run):
    result = _base_result(jpeg_path)
    webp_path = jpeg_path.with_suffix(".webp")
    if not force and image_file_is_ready(str(webp_path)):
        result.update(status="skipped_existing", reason="valid_webp_exists")
        return result
    temporary_path = None
    try:
        with Image.open(jpeg_path) as source_image:
            source_image.load()
            source_image = source_image.convert("RGB")
            fd, temporary_path = tempfile.mkstemp(
                prefix="." + webp_path.name + ".", suffix=".tmp", dir=webp_path.parent
            )
            os.close(fd)
            source_image.save(
                temporary_path, format="WEBP", quality=int(quality), method=6
            )
        if not image_file_is_ready(temporary_path):
            raise OSError("generated WebP could not be decoded")
        gate = _gate_result(jpeg_path, Path(temporary_path), _comparison_metrics(jpeg_path, temporary_path))
        result.update({key: value for key, value in gate.items() if key != "accepted"})
        if not gate["accepted"]:
            result.update(status="rejected", reason=gate["reason"])
        elif dry_run:
            result.update(status="would_convert", reason="verified_dry_run")
        else:
            os.replace(temporary_path, webp_path)
            temporary_path = None
            if not image_file_is_ready(str(webp_path)):
                raise OSError("WebP failed post-replace validation")
            result.update(status="converted", reason="verified")
    except Exception as error:
        result.update(status="error", reason=str(error))
    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return result


def _cleanup_one(jpeg_path, apply, confirmed):
    result = _base_result(jpeg_path)
    webp_path = jpeg_path.with_suffix(".webp")
    try:
        if not image_file_is_ready(str(webp_path)):
            result.update(status="kept", reason="webp_missing_or_invalid")
            return result
        gate = _gate_result(jpeg_path, webp_path, _comparison_metrics(jpeg_path, webp_path))
        result.update({key: value for key, value in gate.items() if key != "accepted"})
        if not gate["accepted"]:
            result.update(status="kept", reason=gate["reason"])
            return result
        if not apply or not confirmed:
            result.update(status="would_cleanup", reason="verified_dry_run")
            return result
        # Revalidate immediately before the destructive operation.
        if not image_file_is_ready(str(webp_path)):
            result.update(status="kept", reason="webp_changed_before_cleanup")
            return result
        recheck = _gate_result(jpeg_path, webp_path, _comparison_metrics(jpeg_path, webp_path))
        if not recheck["accepted"]:
            result.update(status="kept", reason="revalidation_failed")
            return result
        os.remove(jpeg_path)
        result.update(status="cleaned", reason="verified_and_removed")
    except Exception as error:
        result.update(status="error", reason=str(error))
    return result


def _write_report(report, report_path=None):
    output_path = Path(
        report_path
        or Path(FNAMES.Ortho4XP_dir) / "Ortho4XP_cache_reports"
        / f"cache-migration-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix="." + output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return str(output_path)


def _run_parallel(paths, worker, workers, progress):
    results = []
    total = len(paths)
    if not paths:
        return results
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(worker, path): path for path in paths}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if progress:
                progress(index, total, result)
    results.sort(key=lambda item: item["jpeg"])
    return results


def _report(mode, paths, results, dry_run, extra=None, report_path=None):
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "dry_run": bool(dry_run),
        "files_considered": len(paths),
        "files": results,
    }
    if extra:
        report.update(extra)
    report["report_path"] = _write_report(report, report_path)
    return report


def migrate_cache(
    mode,
    root=None,
    tiles=(),
    providers=(),
    zoomlevels=(),
    quality=None,
    workers=2,
    force=False,
    dry_run=False,
    apply=False,
    confirmed=False,
    progress=None,
    report_path=None,
):
    """Run conversion or cleanup and return the JSON-serializable report."""
    for selector in tiles:
        _tile_tokens(selector)
    for zoomlevel in zoomlevels:
        if int(zoomlevel) < 0:
            raise ValueError("zoom levels must be non-negative integers")
    paths = iter_jpeg_cache_files(root, tiles, providers, zoomlevels)
    if mode == "convert":
        if quality is None:
            raise ValueError("quality is required for convert")
        _, parsed_quality = validate_cache_settings("webp", quality)
        results = _run_parallel(
            paths,
            lambda path: _convert_one(path, parsed_quality, force, dry_run),
            workers,
            progress,
        )
        return _report(
            mode,
            paths,
            results,
            dry_run,
            {
                "quality": parsed_quality,
                "force": bool(force),
                "thresholds": {
                    "psnr_db_min": MIN_PSNR_DB,
                    "mae_8bit_max": MAX_MAE_8BIT,
                    "webp_must_be_smaller": True,
                },
            },
            report_path,
        )
    if mode == "cleanup":
        actual_apply = bool(apply and confirmed and not dry_run)
        results = _run_parallel(
            paths,
            lambda path: _cleanup_one(path, actual_apply, confirmed),
            workers,
            progress,
        )
        return _report(
            mode,
            paths,
            results,
            not actual_apply,
            {"apply": actual_apply, "confirmation": bool(confirmed)},
            report_path,
        )
    raise ValueError(f"unsupported cache migration mode: {mode}")
