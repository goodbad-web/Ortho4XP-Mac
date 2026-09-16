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


def test_fp8_missing_pack_falls_back_to_lanczos(tmp_path, monkeypatch):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(source)
    monkeypatch.setattr(IMG.UI, "Ortho4XP_dir", str(tmp_path / "ortho4xp"))
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "assert sys.argv[1] == '--ci-lanczos-upscale'\n"
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
    assert effective == "ci_lanczos"
    assert reason == "fp8_model_unavailable"
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
        "    image.resize((image.width * 2, image.height * 2)).save(output)\n",
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
