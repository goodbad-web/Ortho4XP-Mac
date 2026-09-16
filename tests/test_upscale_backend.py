import sys
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
VERIFY_ROOT = Path(__file__).parents[1] / "Utils" / "run"
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

import O4_Config_Utils as CFG  # noqa: E402
import O4_Imagery_Utils as IMG  # noqa: E402
import O4_Tile_Utils as TILE  # noqa: E402
import verify_metal as VERIFY  # noqa: E402


def test_legacy_upscale_keys_prefer_lanczos_key():
    assert CFG._legacy_upscale_backend({"use_neural_upscale": True}) == "ci_lanczos"
    assert CFG._legacy_upscale_backend({"use_neural_upscale": False}) == "none"
    assert CFG._legacy_upscale_backend(
        {"use_neural_upscale": True, "use_lanczos_upscale": False}
    ) == "none"


def test_upscale_backend_display_values_follow_locale(monkeypatch):
    monkeypatch.setenv("ORTHO4XP_LANG", "ja_JP.UTF-8")
    assert CFG._config_display_value("upscale_backend", "none") == "なし"
    assert CFG._config_short_name("upscale_backend") == "アップスケール方式"
    assert CFG._config_raw_value("upscale_backend", "なし") == "none"
    assert CFG._config_raw_value("upscale_backend", "MetalFX Spatial") == "metalfx_spatial"
    assert CFG._config_display_value("upscale_backend", "fp8_tensorops") == "TensorOps"
    assert CFG._config_raw_value("upscale_backend", "FP8 TensorOps") == "tensorops"


def test_fp8_model_pack_config_is_external_and_optional():
    assert CFG.cfg_vars["fp8_model_path"]["default"] == ""
    assert "fp8_model_path" in CFG.list_dsf_vars


def test_new_upscale_backend_defaults_to_metalfx_spatial():
    assert CFG.cfg_vars["upscale_backend"]["default"] == "metalfx_spatial"


def test_fp8_missing_pack_falls_back_to_metalfx(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(source)
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "assert sys.argv[1] == '--metalfx-spatial-upscale'\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2)).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source),
        "fp8_tensorops",
        str(helper),
        str(tmp_path / "missing-pack"),
    )

    assert output is not None
    assert effective == "metalfx_spatial"
    assert reason == "fp8_model_unavailable"
    assert output.endswith("_metalfx_spatial_upscaled.png")
    with Image.open(output) as image:
        assert image.size == (4, 6)


def test_fp8_helper_failure_falls_back_to_lanczos(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(source)
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "if sys.argv[1] == '--tensorops-upscale': sys.exit(7)\n"
        "assert sys.argv[1] == '--ci-lanczos-upscale'\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2)).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source), "fp8_tensorops", str(helper), str(pack)
    )

    assert output is not None
    assert effective == "ci_lanczos"
    assert reason == "tensorops_exit_7"
    with Image.open(output) as image:
        assert image.size == (4, 6)


def test_fp8_failure_retries_original_input_with_metalfx(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(source)
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "if sys.argv[1] == '--tensorops-upscale': sys.exit(7)\n"
        "assert sys.argv[1] == '--metalfx-spatial-upscale'\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2)).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source), "tensorops", str(helper), str(pack)
    )

    assert output is not None
    assert effective == "metalfx_spatial"
    assert reason == "tensorops_exit_7"
    assert output.endswith("_metalfx_spatial_upscaled.png")


def test_metalfx_failure_falls_back_to_lanczos(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(source)
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "if sys.argv[1] == '--metalfx-spatial-upscale': sys.exit(7)\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2)).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source), "metalfx_spatial", str(helper)
    )

    assert output is not None
    assert effective == "ci_lanczos"
    assert reason == "metalfx_exit_7"
    assert output.endswith("_ci_lanczos_upscaled.png")
    with Image.open(output) as image:
        assert image.size == (4, 6)


def test_transparent_input_uses_metalfx_rgba_path(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "transparent.png"
    Image.new("RGBA", (2, 2), (20, 40, 60, 128)).save(source)
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "assert sys.argv[1] == '--metalfx-spatial-upscale'\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source), "metalfx_spatial", str(helper)
    )

    assert output is not None
    assert effective == "metalfx_spatial"
    assert reason is None
    assert output.endswith("_metalfx_spatial_upscaled.png")
    with Image.open(output) as image:
        assert image.mode == "RGBA"
        assert image.size == (4, 4)
        assert image.getchannel("A").getextrema()[0] < 255


def test_metalfx_batch_records_each_output(tmp_path):
    from PIL import Image

    source_a = tmp_path / "source-a.png"
    source_b = tmp_path / "source-b.png"
    reference_a = tmp_path / "reference-a.png"
    reference_b = tmp_path / "reference-b.png"
    Image.new("RGB", (2, 2), (30, 60, 90)).save(source_a)
    Image.new("RGB", (3, 2), (90, 60, 30)).save(source_b)
    Image.new("RGB", (4, 4), (30, 60, 90)).save(reference_a)
    Image.new("RGB", (6, 4), (90, 60, 30)).save(reference_b)
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "assert sys.argv[1] == '--metalfx-spatial-upscale-batch'\n"
        "for index in range(2, len(sys.argv), 2):\n"
        "    image = Image.open(sys.argv[index])\n"
        "    output = sys.argv[index + 1]\n"
        "    image.resize((image.width * 2, image.height * 2)).save(output)\n"
        "    print('metalfx_batch_item={}/2 backend=metalfx_spatial effective_backend=metalfx_spatial dispatch=batch'.format((index - 2) // 2 + 1))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    output_a = tmp_path / "output-a.png"
    output_b = tmp_path / "output-b.png"

    result = VERIFY.compare_metalfx_batch(
        helper,
        [
            (source_a, output_a, reference_a),
            (source_b, output_b, reference_b),
        ],
        1,
    )

    assert result["status"] == "PASS"
    assert result["dispatch"] == "batch"
    assert result["batch_tasks"] == 2
    assert result["batch_success"] == 2
    assert output_a.is_file() and output_b.is_file()


def test_metalfx_batch_distinguishes_ci_fallback(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    output_a = tmp_path / "output-a.png"
    output_b = tmp_path / "output-b.png"
    Image.new("RGB", (2, 2), (30, 60, 90)).save(source)
    Image.new("RGB", (4, 4), (30, 60, 90)).save(reference)
    helper = tmp_path / "fake_fallback_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "assert sys.argv[1] == '--metalfx-spatial-upscale-batch'\n"
        "for index in range(2, len(sys.argv), 2):\n"
        "    image = Image.open(sys.argv[index])\n"
        "    image.resize((image.width * 2, image.height * 2)).save(sys.argv[index + 1])\n"
        "    print('metalfx_batch_item={}/2 backend=metalfx_spatial effective_backend=ci_lanczos dispatch=batch'.format((index - 2) // 2 + 1))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    result = VERIFY.compare_metalfx_batch(
        helper,
        [(source, output_a, reference), (source, output_b, reference)],
        1,
    )

    assert result["status"] == "SKIP(metalfx_fallback)"
    assert result["effective_backend"] == "ci_lanczos"
    assert result["batch_fallback"] == 2


def _direct_dds_spec(tmp_path, name, item_index):
    temporary = tmp_path / f"{name}.gpu.tmp.dds"
    final = tmp_path / f"{name}.dds"
    return {
        "item": (f"tile-{item_index}", item_index, 0, 16, "BI"),
        "request": {
            "input": str(tmp_path / f"{name}.jpg"),
            "mask": "none",
            "output": str(temporary),
            "format": "BC1",
            "color": {
                "r": 1.0,
                "g": 1.0,
                "b": 1.0,
                "contrast": 1.0,
                "brightness": 0.0,
                "saturation": 1.0,
            },
        },
        "input_size": (2, 2),
        "temporary_path": str(temporary),
        "final_path": str(final),
        "target_format": "BC1",
        "cleanup_paths": [],
    }


def test_metalfx_direct_dds_batch_writes_no_png_and_cleans_requests(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(TILE.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    monkeypatch.setattr(
        TILE.IMG,
        "validate_dds_file",
        lambda path, **kwargs: (Path(path).is_file(), None),
    )
    request_log = tmp_path / "request-log"
    helper = tmp_path / "fake_direct_dds_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        f"open({str(request_log)!r} + '-' + str(os.getpid()), 'w', encoding='utf-8').write(json.dumps(request))\n"
        "for item in request['items']:\n"
        "    open(item['output'], 'wb').write(b'DDS direct')\n"
        "    print('metalfx_dds_item=1/{} backend=metalfx_spatial effective_backend=metalfx_spatial dispatch=direct_dds metalfx_ms=1 readback_ms=2 dds_ms=3 total_ms=6'.format(len(request['items'])))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    specs = [_direct_dds_spec(tmp_path, f"image-{index}", index) for index in range(9)]

    result = TILE._run_metalfx_direct_dds_batch(
        str(helper), specs, worker_limit=2, chunk_size=8
    )

    assert result["batch_tasks"] == 9
    assert result["batch_success"] == 9
    assert result["batch_workers"] == 2
    assert result["batch_chunks"] == 2
    assert result["batch_fallback"] == 0
    assert result["metalfx_ms"] == 9.0
    assert result["readback_ms"] == 18.0
    assert result["dds_ms"] == 27.0
    assert all(Path(spec["final_path"]).is_file() for spec in specs)
    assert not list(tmp_path.glob("*.png"))
    assert not list((tmp_path / "ortho4xp" / "tmp").glob(".metalfx-spatial-dds-*.json"))
    requests = list(tmp_path.glob("request-log-*"))
    assert len(requests) == 2
    for request_path in requests:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["version"] == 1
        assert all(item["format"] == "BC1" for item in request["items"])


def test_metalfx_direct_dds_batch_keeps_success_and_reports_one_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(TILE.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    monkeypatch.setattr(
        TILE.IMG,
        "validate_dds_file",
        lambda path, **kwargs: (Path(path).is_file(), None),
    )
    helper = tmp_path / "fake_partial_direct_dds_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import sys\n"
        "request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "for item in request['items']:\n"
        "    if item['input'].endswith('image-fail.jpg'):\n"
        "        print('metalfx_dds_item=1/1 backend=metalfx_spatial effective_backend=metalfx_spatial fallback_reason=gpu_failure')\n"
        "        continue\n"
        "    open(item['output'], 'wb').write(b'DDS direct')\n"
        "    print('metalfx_dds_item=1/1 backend=metalfx_spatial effective_backend=metalfx_spatial metalfx_ms=1 readback_ms=1 dds_ms=1')\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    specs = [
        _direct_dds_spec(tmp_path, "image-ok-a", 0),
        _direct_dds_spec(tmp_path, "image-fail", 1),
        _direct_dds_spec(tmp_path, "image-ok-b", 2),
    ]

    result = TILE._run_metalfx_direct_dds_batch(str(helper), specs)

    assert result["batch_success"] == 2
    assert result["batch_failed"] == 1
    assert result["failed_items"] == [specs[1]["item"]]
    assert result["fallback_reasons"] == {"gpu_failure": 1}
    assert Path(specs[0]["final_path"]).is_file()
    assert not Path(specs[1]["final_path"]).exists()
    assert Path(specs[2]["final_path"]).is_file()
    assert not Path(specs[1]["temporary_path"]).exists()


def test_tensorops_direct_dds_uses_pack_chunks_and_no_png(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(TILE.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    monkeypatch.setattr(
        TILE.IMG,
        "validate_dds_file",
        lambda path, **kwargs: (Path(path).is_file(), None),
    )
    request_log = tmp_path / "tensorops-request"
    helper = tmp_path / "fake_tensorops_direct_dds_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        f"open({str(request_log)!r} + '-' + str(os.getpid()), 'w', encoding='utf-8').write(json.dumps(request))\n"
        "for index, item in enumerate(request['items'], 1):\n"
        "    open(item['output'], 'wb').write(b'DDS tensorops')\n"
        "    print('tensorops_dds_item={}/{} backend=tensorops effective_backend=tensorops dispatch=direct_dds dtype=MetalFloat8E4M3 rss_mb=321'.format(index, len(request['items'])))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    specs = [_direct_dds_spec(tmp_path, f"image-{index}", index) for index in range(9)]
    pack_path = tmp_path / "model.fp8sr"

    result = TILE._run_tensorops_direct_dds_batch(
        str(helper), str(pack_path), specs, chunk_size=8, worker_limit=2
    )

    assert result["batch_tasks"] == 9
    assert result["batch_success"] == 9
    assert result["batch_failed"] == 0
    assert result["batch_workers"] == 2
    assert result["batch_chunks"] == 2
    assert result["chunk_size"] == 8
    assert result["rss_after_item_mb"] == 321
    assert result["peak_rss_mb"] > 0
    assert all(Path(spec["final_path"]).is_file() for spec in specs)
    assert not list(tmp_path.glob("*.png"))
    assert not list((tmp_path / "ortho4xp" / "tmp").glob(".tensorops-dds-*.json"))
    requests = list(tmp_path.glob("tensorops-request-*"))
    assert len(requests) == 2
    for request_path in requests:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["version"] == 1
        assert request["pack"] == str(pack_path)
        assert request["fallback_to_ci"] is False
        assert len(request["items"]) <= 8
        assert all(item["output"].endswith(".gpu.tmp.dds") for item in request["items"])


def test_tensorops_direct_dds_summary_reports_lanczos_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(TILE.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    monkeypatch.setattr(
        TILE.IMG,
        "validate_dds_file",
        lambda path, **kwargs: (Path(path).is_file(), None),
    )
    helper = tmp_path / "fake_tensorops_fallback_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import sys\n"
        "request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "for index, item in enumerate(request['items'], 1):\n"
        "    open(item['output'], 'wb').write(b'DDS fallback')\n"
        "    print('tensorops_dds_item={}/{} backend=tensorops effective_backend=ci_lanczos dispatch=direct_dds fallback_reason=alpha rss_mb=123'.format(index, len(request['items'])))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    specs = [_direct_dds_spec(tmp_path, "image-fallback", 0)]

    result = TILE._run_tensorops_direct_dds_batch(
        str(helper), str(tmp_path / "model.fp8sr"), specs
    )

    assert result["effective_backend"] == "ci_lanczos"
    assert result["batch_fallback"] == 1
    assert result["fallback_reasons"] == {"alpha": 1}


def test_tensorops_direct_dds_summary_reports_mixed_failure_and_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(TILE.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    monkeypatch.setattr(
        TILE.IMG,
        "validate_dds_file",
        lambda path, **kwargs: (Path(path).is_file(), None),
    )
    helper = tmp_path / "fake_tensorops_mixed_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import sys\n"
        "request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "for index, item in enumerate(request['items'], 1):\n"
        "    if item['input'].endswith('image-fail.jpg'):\n"
        "        print('tensorops_dds_item={}/{} backend=tensorops effective_backend=ci_lanczos dispatch=direct_dds fallback_reason=gpu_failure'.format(index, len(request['items'])))\n"
        "        continue\n"
        "    open(item['output'], 'wb').write(b'DDS fallback')\n"
        "    print('tensorops_dds_item={}/{} backend=tensorops effective_backend=ci_lanczos dispatch=direct_dds fallback_reason=alpha'.format(index, len(request['items'])))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    specs = [
        _direct_dds_spec(tmp_path, "image-ok", 0),
        _direct_dds_spec(tmp_path, "image-fail", 1),
    ]

    result = TILE._run_tensorops_direct_dds_batch(
        str(helper), str(tmp_path / "model.fp8sr"), specs
    )

    assert result["effective_backend"] == "mixed"
    assert result["batch_success"] == 1
    assert result["batch_fallback"] == 1
    assert result["batch_failed"] == 1


def test_streaming_gpu_eligibility_does_not_build_spec_or_materialize_mask(monkeypatch):
    from types import SimpleNamespace

    runner = object.__new__(TILE._StreamingConversionRunner)
    runner.gpu_server = SimpleNamespace(gpu_disabled=False)
    runner.dds_converter = "TextureConverter"
    monkeypatch.setattr(TILE, "_streaming_gpu_source", lambda item: object())

    def unexpected_builder(*args, **kwargs):
        raise AssertionError("builder called")

    monkeypatch.setattr(
        TILE,
        "_build_streaming_gpu_spec",
        unexpected_builder,
    )

    assert runner._gpu_eligible(SimpleNamespace(payload=("payload",)))


def test_parallel_imagery_stage_reserves_gpu_for_color_filters():
    from types import SimpleNamespace

    tile = SimpleNamespace(
        use_gpu_acceleration=True,
        use_gpu_for_color_filters=True,
        dds_converter="nvcompress",
        upscale_backend="none",
    )

    assert TILE._parallel_tile_stage_uses_gpu(tile, "imagery/DSF")


def test_tensorops_direct_dds_keeps_completed_chunk_on_sigkill(tmp_path, monkeypatch):
    monkeypatch.setattr(TILE.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    monkeypatch.setattr(
        TILE.IMG,
        "validate_dds_file",
        lambda path, **kwargs: (Path(path).is_file(), None),
    )
    helper = tmp_path / "fake_tensorops_sigkill_helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import signal\n"
        "import sys\n"
        "request = json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "if any(item['input'].endswith('image-8.jpg') for item in request['items']):\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "for item in request['items']:\n"
        "    open(item['output'], 'wb').write(b'DDS tensorops')\n"
        "    print('tensorops_dds_item=1/{} backend=tensorops effective_backend=tensorops dispatch=direct_dds rss_mb=400'.format(len(request['items'])))\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    specs = [_direct_dds_spec(tmp_path, f"image-{index}", index) for index in range(9)]

    result = TILE._run_tensorops_direct_dds_batch(
        str(helper), str(tmp_path / "model.fp8sr"), specs, chunk_size=8
    )

    assert result["batch_success"] == 8
    assert result["batch_failed"] == 1
    assert result["failed_items"] == [specs[8]["item"]]
    assert result["signal"] == 9
    assert result["fallback_reasons"]["process_signal_9"] >= 1
    assert all(Path(spec["final_path"]).is_file() for spec in specs[:8])
    assert not Path(specs[8]["final_path"]).exists()
    assert not Path(specs[8]["temporary_path"]).exists()
