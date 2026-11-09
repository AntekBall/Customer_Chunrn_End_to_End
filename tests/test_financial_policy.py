import numpy as np
import pandas as pd
import pytest

from src.models.campaign import CampaignScenario, compare_policies
from tests.helpers import model_config


def customers(charges):
    return pd.DataFrame({
        "customer_id": [f"id-{i}" for i in range(len(charges))],
        "monthlycharges": charges,
    })


def test_exact_economics_and_ranking():
    scenario = CampaignScenario(contact_cost=10.0, capacity_fraction=0.5)
    scored, summary = compare_policies(
        customers([10, 100, 300, 20]), [0.9, 0.5, 0.3, 0.1], scenario
    )
    np.testing.assert_allclose(
        scored["expected_net_value"], [-7.84, 2, 11.6, -9.52]
    )
    result = summary.set_index("policy").loc["expected_profit_positive_only"]
    assert result["selected_customers"] == 2
    assert result["campaign_cost"] == 20
    assert result["expected_net_value"] == pytest.approx(13.6)
    assert result["scenario_roi"] == pytest.approx(0.68)
    assert result["expected_prevented_churn"] == pytest.approx(0.16)


def test_profit_ranking_beats_risk_at_same_capacity():
    _, summary = compare_policies(
        customers([100, 300]), [0.5, 0.3],
        CampaignScenario(contact_cost=10.0, capacity_fraction=0.5),
    )
    results = summary.set_index("policy")
    assert results.loc["risk_positive_only", "expected_net_value"] == pytest.approx(2)
    assert results.loc[
        "expected_profit_positive_only", "expected_net_value"
    ] == pytest.approx(11.6)


@pytest.mark.parametrize("scenario", [
    CampaignScenario(prevention_probability=0),
    CampaignScenario(gross_margin=0),
    CampaignScenario(capacity_fraction=0),
])
def test_no_contacts_when_unprofitable_or_zero_capacity(scenario):
    _, summary = compare_policies(customers([100, 200]), [0.1, 0.2], scenario)
    assert summary["selected_customers"].eq(0).all()
    assert summary["expected_net_value"].eq(0).all()
    assert summary["scenario_roi"].isna().all()


def test_floor_capacity_and_zero_profit_excluded():
    _, summary = compare_policies(
        customers([100] * 3), [1, 1, 1],
        CampaignScenario(capacity_fraction=0.1),
    )
    assert summary["capacity_customers"].eq(0).all()
    _, summary = compare_policies(
        customers([10, 20]), [0.5, 0.25],
        CampaignScenario(
            contact_cost=5, gross_margin=1, retained_months=1,
            prevention_probability=1, capacity_fraction=1,
        ),
    )
    assert summary["selected_customers"].eq(0).all()


def test_free_campaign_has_undefined_roi():
    _, summary = compare_policies(
        customers([100, 200]), [0.5, 0.5],
        CampaignScenario(contact_cost=0, capacity_fraction=1),
    )
    assert summary["scenario_roi"].isna().all()
    assert summary.iloc[-1]["expected_net_value"] > 0


@pytest.mark.parametrize("field,value", [
    ("contact_cost", -1), ("gross_margin", 1.1),
    ("retained_months", 0), ("prevention_probability", float("nan")),
    ("capacity_fraction", 2), ("contact_cost", True),
])
def test_invalid_scenarios(field, value):
    with pytest.raises(ValueError):
        CampaignScenario(**{field: value})


@pytest.mark.parametrize("case", ["nan", "negative", "probability", "ids", "shape"])
def test_invalid_customers(case):
    frame = customers([100, 200])
    probabilities = [0.1, 0.2]
    if case == "nan":
        frame.loc[0, "monthlycharges"] = np.nan
    elif case == "negative":
        frame.loc[0, "monthlycharges"] = -1
    elif case == "probability":
        probabilities = [0.1, 1.1]
    elif case == "shape":
        probabilities = [0.1]
    else:
        frame.loc[1, "customer_id"] = "id-0"
    with pytest.raises(ValueError):
        compare_policies(frame, probabilities, CampaignScenario())


def test_labels_are_not_used_for_policy_selection():
    frame = customers([100, 200])
    _, first = compare_policies(
        frame.assign(churn=[0, 1]), [0.3, 0.4],
        CampaignScenario(capacity_fraction=1),
    )
    _, second = compare_policies(
        frame.assign(churn=[1, 0]), [0.3, 0.4],
        CampaignScenario(capacity_fraction=1),
    )
    pd.testing.assert_frame_equal(first, second)


def test_main_logs_scenario_with_temporary_model(tmp_path, monkeypatch, capsys):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import brier_score_loss, log_loss

    import experiments.financial_policy as module
    from src.models.pipeline import (
        BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
    )

    if mlflow.active_run() is not None:
        pytest.fail("Run without an active MLflow run.")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        frame = pd.DataFrame({
            name: np.full(20, 100.0) for name in NUMERIC_COLUMNS
        })
        for name in CATEGORICAL_COLUMNS:
            frame[name] = "A"
        for name in BINARY_COLUMNS:
            frame[name] = 0
        frame["customer_id"] = [f"id-{i}" for i in range(20)]
        frame["churn"] = [0, 1] * 10
        frame["partition"] = "validation"
        config = model_config()
        config.mlflow.tracking_directory = str(tmp_path)
        provenance = {"raw_data": {"sha256": "synthetic"}}
        experiment_id = mlflow.create_experiment(
            "financial-tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        model = DummyClassifier(strategy="prior").fit(
            frame[list(module.FEATURE_COLUMNS)], frame["churn"]
        )
        with mlflow.start_run(experiment_id=experiment_id) as parent:
            mlflow.log_dict(config.model_dump(mode="json"), "evaluation_config.json")
            mlflow.log_dict(provenance, "input_provenance.json")
            with mlflow.start_run(experiment_id=experiment_id, nested=True) as child:
                mlflow.set_tags({
                    "model": "catboost", "evaluation_partition": "validation",
                })
                mlflow.log_param("positive_class_weight", 1.0)
                mlflow.log_metrics({
                    "validation_log_loss": log_loss(frame["churn"], np.full(20, 0.5)),
                    "validation_brier_score": brier_score_loss(
                        frame["churn"], np.full(20, 0.5)
                    ),
                })
                logged = mlflow.sklearn.log_model(
                    model, name="pipeline",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                source_id = child.info.run_id
            mlflow.log_dict([{
                "run_id": source_id, "model_uri": logged.model_uri,
            }], "validation_comparison.json")
        monkeypatch.setattr(module, "SOURCE_RUN_ID", source_id)
        monkeypatch.setattr(module, "load_training_config", lambda: config)
        monkeypatch.setattr(module, "current_provenance", lambda config: provenance)

        def load_customers(connection):
            connection.register("fixture", frame)
            connection.execute("CREATE TABLE baseline_features AS SELECT * FROM fixture")

        monkeypatch.setattr(module, "load_customers", load_customers)
        monkeypatch.setattr(module, "load_split_views", lambda *args: None)
        monkeypatch.setattr(module, "load_feature_view", lambda *args: None)
        expected_uri = mlflow.get_tracking_uri()
        module.main()
        assert mlflow.get_tracking_uri() == expected_uri
        assert mlflow.active_run() is None
        client = MlflowClient()
        runs = client.search_runs(
            [experiment_id], filter_string="tags.mode = 'financial_scenario'"
        )
        assert len(runs) == 1
        assert runs[0].info.status == "FINISHED"
        assert runs[0].data.metrics[
            "expected_profit_positive_only_selected_customers"
        ] == 2
        assert runs[0].data.metrics[
            "expected_profit_positive_only_expected_net_value"
        ] == pytest.approx(20)
        assert float(runs[0].data.params["contact_cost"]) == 2.0
        grid = module.read_json(
            client, runs[0].info.run_id, "sensitivity/grid_definition.json"
        )
        assert grid["contact_costs"] == [1.0, 2.0, 5.0, 10.0]
        csv_path = client.download_artifacts(
            runs[0].info.run_id, "sensitivity/scenario_comparison.csv"
        )
        assert len(pd.read_csv(csv_path)) == 108
        html_path = client.download_artifacts(
            runs[0].info.run_id, "sensitivity/profit_advantage_6months.html"
        )
        from pathlib import Path

        assert "plotly" in Path(html_path).read_text(encoding="utf-8")
        policy = module.read_json(client, runs[0].info.run_id, "policy_definition.json")
        assert policy["eligibility"] == "expected_net_value > 0"
        assert "NOT MEASURED" in capsys.readouterr().out
    finally:
        mlflow.set_tracking_uri(previous_uri)
