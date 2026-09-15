import sys
from pathlib import Path


SRC_ROOT = Path(__file__).parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import O4_Config_Utils as CFG  # noqa: E402
import O4_Imagery_Utils as IMG  # noqa: E402


def test_legacy_upscale_keys_prefer_lanczos_key():
    assert CFG._legacy_upscale_backend({"use_neural_upscale": True}) == "lanczos"
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
    assert CFG._config_display_value("upscale_backend", "fp8_tensorops") == "FP8 TensorOps"
    assert CFG._config_raw_value("upscale_backend", "FP8 TensorOps") == "fp8_tensorops"


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
        "assert sys.argv[1] == '--lanczos-upscale'\n"
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
    assert effective == "lanczos"
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
        "if sys.argv[1] == '--fp8-tensorops-upscale': sys.exit(7)\n"
        "assert sys.argv[1] == '--lanczos-upscale'\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2)).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source), "fp8_tensorops", str(helper), str(pack)
    )

    assert output is not None
    assert effective == "lanczos"
    assert reason == "fp8_tensorops_exit_7"
    with Image.open(output) as image:
        assert image.size == (4, 6)


def test_metalfx_failure_falls_back_to_lanczos(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (20, 40, 60)).save(source)
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
    assert effective == "lanczos"
    assert reason == "metalfx_exit_7"
    assert output.endswith("_lanczos_upscaled.png")
    with Image.open(output) as image:
        assert image.size == (4, 6)


def test_transparent_input_skips_metalfx_and_uses_lanczos(tmp_path):
    from PIL import Image

    source = tmp_path / "transparent.png"
    Image.new("RGBA", (2, 2), (20, 40, 60, 128)).save(source)
    helper = tmp_path / "fake_ashelper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from PIL import Image\n"
        "if sys.argv[1] == '--metalfx-spatial-upscale': raise AssertionError('MetalFX must not receive alpha')\n"
        "image = Image.open(sys.argv[2])\n"
        "image.resize((image.width * 2, image.height * 2)).save(sys.argv[3])\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)

    output, effective, reason = IMG.run_upscale(
        str(source), "metalfx_spatial", str(helper)
    )

    assert output is not None
    assert effective == "lanczos"
    assert reason == "alpha"
    assert output.endswith("_lanczos_upscaled.png")
