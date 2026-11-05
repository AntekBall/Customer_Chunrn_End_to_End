from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import src.config as config_module
from src.config import (
    DEFAULT_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
)


@pytest.fixture
def valid_config(tmp_path, monkeypatch):
    contents = yaml.safe_load(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    )

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    split_path = tmp_path / "data" / "split.parquet"
    feature_path = tmp_path / "sql" / "features.sql"

    split_path.parent.mkdir()
    feature_path.parent.mkdir()
    split_path.touch()
    feature_path.write_text("SELECT 1;", encoding="utf-8")

    contents["data"] = {
        "split_manifest": r"data\split.parquet",
        "feature_sql": r"sql\features.sql",
    }
    contents["mlflow"]["tracking_directory"] = "tracking"

    return contents


def write_config(tmp_path, contents):
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump(contents),
        encoding="utf-8",
    )
    return path


def test_loads_valid_config_and_resolves_paths(tmp_path, valid_config):
    path = write_config(tmp_path, valid_config)

    config = load_training_config(path)

    assert isinstance(config, TrainingConfig)
    assert config.random_seed == 42
    assert config.optimization.metric == "average_precision"

    assert Path(config.data.split_manifest) == (
        tmp_path / "data" / "split.parquet"
    ).resolve()

    assert Path(config.data.feature_sql) == (
        tmp_path / "sql" / "features.sql"
    ).resolve()

    assert Path(config.mlflow.tracking_directory) == (
        tmp_path / "tracking"
    ).resolve()

    assert not (tmp_path / "tracking").exists()


def test_relative_config_path_uses_project_root(tmp_path, valid_config):
    write_config(tmp_path, valid_config)

    config = load_training_config(Path("training.yaml"))

    assert config.random_seed == 42


def test_absolute_input_paths_are_preserved(tmp_path, valid_config):
    valid_config["data"]["split_manifest"] = str(
        tmp_path / "data" / "split.parquet"
    )
    valid_config["data"]["feature_sql"] = str(
        tmp_path / "sql" / "features.sql"
    )

    config = load_training_config(write_config(tmp_path, valid_config))

    assert Path(config.data.split_manifest).is_absolute()
    assert Path(config.data.feature_sql).is_absolute()


def test_loading_does_not_modify_config(tmp_path, valid_config):
    path = write_config(tmp_path, valid_config)
    original = path.read_bytes()

    load_training_config(path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("random_seed",), -1),
        (("random_seed",), 4294967296),
        (("random_seed",), "42"),
        (("cross_validation", "n_splits"), 1),
        (("cross_validation", "shuffle"), "true"),
        (("optimization", "trials_per_model"), 0),
        (("optimization", "timeout_seconds_per_model"), -1),
        (("optimization", "parallel_trials"), 2),
        (("optimization", "direction"), "minimize"),
        (("optimization", "metric"), "accuracy"),
        (("evaluation", "metrics"), []),
        (("evaluation", "metrics"), ["roc_auc", "roc_auc"]),
        (("evaluation", "targeting_fractions"), []),
        (("evaluation", "targeting_fractions"), [0.0]),
        (("evaluation", "targeting_fractions"), [1.1]),
        (("evaluation", "targeting_fractions"), [0.1, 0.1]),
        (("mlflow", "experiment_name"), "   "),
        (("models", "lightgbm", "fixed_parameters", "n_jobs"), 0),
        (
            ("models", "xgboost", "fixed_parameters", "objective"),
            "reg:squarederror",
        ),
        (
            ("models", "hist_gradient_boosting",
             "fixed_parameters", "early_stopping"),
            True,
        ),
    ],
)
def test_invalid_settings_raise(tmp_path, valid_config, keys, value):
    contents = deepcopy(valid_config)
    target = contents

    for key in keys[:-1]:
        target = target[key]

    target[keys[-1]] = value

    with pytest.raises(ValidationError):
        load_training_config(write_config(tmp_path, contents))


@pytest.mark.parametrize(
    "section",
    [
        (),
        ("data",),
        ("optimization",),
        ("models", "lightgbm", "fixed_parameters"),
    ],
)
def test_unknown_keys_raise(tmp_path, valid_config, section):
    target = valid_config

    for key in section:
        target = target[key]

    target["unexpected_setting"] = 123

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_training_config(write_config(tmp_path, valid_config))


def test_all_models_disabled_raises(tmp_path, valid_config):
    for model in valid_config["models"].values():
        model["enabled"] = False

    with pytest.raises(ValidationError, match="At least one model"):
        load_training_config(write_config(tmp_path, valid_config))


def test_optimization_metric_must_be_evaluated(tmp_path, valid_config):
    valid_config["evaluation"]["metrics"] = ["roc_auc"]

    with pytest.raises(ValidationError, match="optimization metric"):
        load_training_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize("metric", ["log_loss", "brier_score"])
def test_loss_metrics_require_minimization(tmp_path, valid_config, metric):
    valid_config["optimization"]["metric"] = metric
    valid_config["optimization"]["direction"] = "minimize"

    config = load_training_config(write_config(tmp_path, valid_config))

    assert config.optimization.direction == "minimize"

    valid_config["optimization"]["direction"] = "maximize"

    with pytest.raises(ValidationError, match="requires direction"):
        load_training_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "text",
    [
        "random_seed: 42\nrandom_seed: 43\n",
        "data:\n  feature_sql: first\n  feature_sql: second\n",
    ],
)
def test_duplicate_yaml_keys_raise(tmp_path, text):
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_training_config(path)


def test_non_string_yaml_key_raises(tmp_path):
    path = tmp_path / "invalid_key.yaml"
    path.write_text("1: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="keys must be strings"):
        load_training_config(path)


@pytest.mark.parametrize("text", ["", "- item\n", "plain text\n"])
def test_non_mapping_yaml_raises(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_training_config(path)


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("data: [unfinished", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_training_config(path)


def test_unsafe_yaml_tag_is_rejected(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text(
        "!!python/tuple [1, 2]",
        encoding="utf-8",
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        load_training_config(path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_training_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("field", ["split_manifest", "feature_sql"])
def test_missing_input_file_raises(tmp_path, valid_config, field):
    valid_config["data"][field] = "missing.file"

    with pytest.raises(FileNotFoundError, match="Configured input file"):
        load_training_config(write_config(tmp_path, valid_config))


def test_missing_required_section_raises(tmp_path, valid_config):
    del valid_config["optimization"]

    with pytest.raises(ValidationError, match="optimization"):
        load_training_config(write_config(tmp_path, valid_config))