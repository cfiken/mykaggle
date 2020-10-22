from pathlib import Path
import pytest

from mykaggle.util.config import load_config


def test_config_loader_yaml():
    config_path = Path('./tests/mykaggle/config/sample.yml')
    expected = 'path'
    actual = load_config(config_path, is_full_path=True)
    assert actual['datadir'] == expected


def test_config_loader_json():
    config_path = Path('./tests/mykaggle/config/sample.json')
    expected = 'path'
    actual = load_config(config_path, is_full_path=True)
    assert actual['datadir'] == expected


def test_config_loader_txt():
    config_path = Path('./tests/mykaggle/config/sample.txt')
    with pytest.raises(ValueError):
        _ = load_config(config_path, is_full_path=True)
