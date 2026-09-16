import pytest

from bsa.config import config_hash, flatten, load_config


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 1\ntraining:\n  epochs: 8\n  learning_rate: 2.0e-5\n", encoding="utf-8")
    return path


def test_override_parses_yaml_types(cfg_file):
    cfg = load_config(cfg_file, ["training.epochs=2", "seed=7"])
    assert cfg["training"]["epochs"] == 2
    assert cfg["seed"] == 7


def test_unknown_key_is_rejected(cfg_file):
    with pytest.raises(KeyError):
        load_config(cfg_file, ["training.epoch=2"])


def test_malformed_override_is_rejected(cfg_file):
    with pytest.raises(ValueError):
        load_config(cfg_file, ["training.epochs"])


def test_flatten_and_hash(cfg_file):
    cfg = load_config(cfg_file)
    assert flatten(cfg) == {"seed": 1, "training.epochs": 8, "training.learning_rate": 2e-5}
    assert config_hash(cfg) != config_hash(load_config(cfg_file, ["seed=2"]))


def test_repo_config_loads():
    cfg = load_config("configs/train.yaml")
    assert isinstance(cfg["training"]["learning_rate"], float)
    assert cfg["data"]["train_source_split"] != cfg["data"]["test_source_split"]
