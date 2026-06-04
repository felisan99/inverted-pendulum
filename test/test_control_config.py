import math

import pytest

from gym_envs.control_config import ControlConfig


def test_defaults_reproduce_1khz():
    cfg = ControlConfig()
    assert cfg.sample_period_us == 1000.0
    assert cfg.ema_alpha == pytest.approx(0.1507, abs=1e-4)


def test_250hz_profile():
    cfg = ControlConfig(sample_freq_hz=250.0, filter_cutoff_hz=26.0)
    assert cfg.sample_period_us == 4000.0
    assert cfg.ema_alpha == pytest.approx(0.4798, abs=1e-4)


def test_ema_alpha_formula():
    cfg = ControlConfig(sample_freq_hz=500.0, filter_cutoff_hz=40.0)
    expected = 1.0 - math.exp(-2.0 * math.pi * 40.0 / 500.0)
    assert cfg.ema_alpha == expected


def test_from_toml_reads_section_and_ignores_extra(tmp_path):
    path = tmp_path / "control.toml"
    path.write_text(
        "[control_config]\n"
        "sample_freq_hz = 250.0\n"
        "filter_cutoff_hz = 26.0\n"
        "unknown_key = 99\n"
    )
    cfg = ControlConfig.from_toml(path)
    assert cfg.sample_freq_hz == 250.0
    assert cfg.filter_cutoff_hz == 26.0
    assert not hasattr(cfg, "unknown_key")


def test_from_toml_missing_section_uses_defaults(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("[other]\nx = 1\n")
    cfg = ControlConfig.from_toml(path)
    assert cfg.sample_freq_hz == 1000.0
    assert cfg.filter_cutoff_hz == 26.0
