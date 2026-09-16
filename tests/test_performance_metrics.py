import json
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from O4_Performance_Utils import PerformanceMetrics  # noqa: E402


def test_metrics_write_atomic_schema_and_attempts(tmp_path):
    class Tile:
        lat = 34
        lon = 133

    metrics = PerformanceMetrics(Tile())
    metrics.set_config({"enable_streaming_conversion": True})
    metrics.set_capabilities({"metal_available": False})
    metrics.begin_attempt(0, {"cover_zl": 18})
    with metrics.stage("imagery/DSF"):
        metrics.increment("textures_downloaded", 3)
    metrics.record_queue("conversion", 2, 8)
    metrics.record_batch("cpu", 3, duration_ms=12.5, status="completed")
    metrics.end_attempt("success")

    destination = tmp_path / "Ortho4XP_performance.json"
    metrics.write(str(destination))

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["tile"] == {"lat": 34, "lon": 133}
    assert payload["attempts"][0]["result"] == "success"
    assert payload["attempts"][0]["counters"]["textures_downloaded"] == 3
    assert payload["attempts"][0]["queue"]["conversion"]["max_size"] == 2
    assert payload["attempts"][0]["batches"]["cpu"]["items"] == 3

