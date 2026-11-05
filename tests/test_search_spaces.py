from copy import deepcopy

import optuna
import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.models.model_builder import MODEL_NAMES, build_model_pipeline
from src.models.search_spaces import (
    TUNABLE_PARAMETERS,
    build_tuned_pipeline,
    suggest_parameters,
)


@pytest.fixture
def config():
    contents = yaml.safe_load(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    )

    for model in contents["models"].values():
        model["enabled"] = True

    return TrainingConfig.model_validate(contents)


def sample_parameters(model_name, seed=42):
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    trial = study.ask()
    parameters = suggest_parameters(trial, model_name)
    return parameters, trial


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_sampling_is_reproducible(model_name):
    first, _ = sample_parameters(model_name)
    second, _ = sample_parameters(model_name)

    assert first == second


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_sampled_keys_match_allowlist(model_name):
    parameters, trial = sample_parameters(model_name)

    assert set(parameters) == TUNABLE_PARAMETERS[model_name]
    assert parameters == trial.params


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fixed_trial_reconstructs_parameters(model_name):
    parameters, _ = sample_parameters(model_name)

    reconstructed = suggest_parameters(
        optuna.trial.FixedTrial(parameters),
        model_name,
    )

    assert reconstructed == parameters


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_parameters_are_applied_without_mutation(config, model_name):
    parameters, _ = sample_parameters(model_name)
    original_parameters = deepcopy(parameters)
    original_config = config.model_dump()

    pipeline = build_tuned_pipeline(model_name, config, parameters)
    actual = pipeline.named_steps["classifier"].get_params()

    for name, value in parameters.items():
        assert actual[name] == value

    assert parameters == original_parameters
    assert config.model_dump() == original_config

    settings = getattr(config.models, model_name)

    for name, value in settings.fixed_parameters.model_dump().items():
        assert actual[name] == value

    seed_key = (
        "random_seed" if model_name == "catboost" else "random_state"
    )
    assert actual[seed_key] == config.random_seed


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fixed_parameters_cannot_be_overridden(config, model_name):
    fixed = getattr(
        config.models, model_name
    ).fixed_parameters.model_dump()

    for name, value in fixed.items():
        with pytest.raises(ValueError, match="Cannot override fixed"):
            build_tuned_pipeline(
                model_name,
                config,
                {name: value},
            )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize(
    "parameter",
    ["unknown_parameter", "random_state", "random_seed"],
)
def test_unsupported_parameters_raise(config, model_name, parameter):
    with pytest.raises(ValueError, match="Unsupported tuning parameters"):
        build_tuned_pipeline(
            model_name,
            config,
            {parameter: 123},
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_disabled_models_raise(config, model_name):
    getattr(config.models, model_name).enabled = False

    with pytest.raises(ValueError, match="Model is disabled"):
        build_tuned_pipeline(model_name, config, {})


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_empty_parameters_preserve_defaults(config, model_name):
    actual = build_tuned_pipeline(model_name, config, {})
    expected = build_model_pipeline(model_name, config)

    assert (
        actual.named_steps["classifier"].get_params()
        == expected.named_steps["classifier"].get_params()
    )


def test_unknown_model_raises(config):
    with pytest.raises(ValueError, match="Unknown model"):
        suggest_parameters(optuna.trial.FixedTrial({}), "unknown")

    with pytest.raises(ValueError, match="Unknown model"):
        build_tuned_pipeline("unknown", config, {})


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_sampling_stays_within_resource_bounds(model_name):
    for seed in range(10):
        parameters, _ = sample_parameters(model_name, seed)

        if "learning_rate" in parameters:
            assert 0.02 <= parameters["learning_rate"] <= 0.2

        iteration_key = {
            "hist_gradient_boosting": "max_iter",
            "catboost": "iterations",
        }.get(model_name, "n_estimators")

        assert 100 <= parameters[iteration_key] <= 400

        if model_name == "random_forest":
            assert parameters["n_estimators"] in [100, 200, 300]
            assert parameters["max_depth"] in [8, 12, 16]
            assert parameters["min_samples_leaf"] in [10, 25, 50, 100]
            assert 0.5 <= parameters["max_samples"] <= 0.9

        elif model_name == "hist_gradient_boosting":
            assert parameters["max_leaf_nodes"] in [7, 15, 31]
            assert 50 <= parameters["min_samples_leaf"] <= 300

        elif model_name == "lightgbm":
            assert parameters["num_leaves"] in [7, 15, 31]
            assert 50 <= parameters["min_child_samples"] <= 300

        elif model_name == "xgboost":
            assert 3 <= parameters["max_depth"] <= 6

        elif model_name == "catboost":
            assert 4 <= parameters["depth"] <= 7