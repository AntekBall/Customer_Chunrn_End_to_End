from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import brier_score_loss, log_loss

from experiments.finalist_validation import current_provenance, read_json
from src.config import TrainingConfig, load_training_config
from src.data.features import load_feature_view
from src.data.loader import load_customers
from src.data.splits import load_split_views
from src.models.pipeline import FEATURE_COLUMNS


SOURCE_RUN_ID = "4b8fa8456a984b8da19bb09e4a5d2334"


@dataclass(frozen=True)
class CampaignScenario:
    contact_cost: float = 2.0
    gross_margin: float = 0.2
    retained_months: float = 6.0
    prevention_probability: float = 0.2
    capacity_fraction: float = 0.1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("gross_margin", "prevention_probability", "capacity_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1.")
        if self.retained_months <= 0:
            raise ValueError("retained_months must be positive.")


def compare_policies(customers, probabilities, scenario):
    if customers.empty:
        raise ValueError("Customer data must be nonempty.")
    if not customers.columns.is_unique or not {
        "customer_id", "monthlycharges"
    }.issubset(customers.columns):
        raise ValueError("Expected unique columns including IDs and monthlycharges.")
    ids = customers["customer_id"]
    if not ids.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all() or not ids.is_unique:
        raise ValueError("Customer IDs must be unique nonblank strings.")
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != (len(customers),)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid probabilities.")
    charges = customers["monthlycharges"].to_numpy(dtype=float)
    if not np.isfinite(charges).all() or (charges < 0).any():
        raise ValueError("Monthly charges must be finite and nonnegative.")

    scored = customers[["customer_id", "monthlycharges"]].copy()
    scored["churn_probability"] = probabilities
    scored["retained_contribution_if_prevented"] = (
        charges * scenario.gross_margin * scenario.retained_months
    )
    scored["expected_prevented_churn"] = (
        probabilities * scenario.prevention_probability
    )
    scored["expected_retained_contribution"] = (
        scored["expected_prevented_churn"]
        * scored["retained_contribution_if_prevented"]
    )
    scored["expected_net_value"] = (
        scored["expected_retained_contribution"] - scenario.contact_cost
    )
    if not np.isfinite(scored.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Scenario calculations overflowed.")

    capacity = int(np.floor(scenario.capacity_fraction * len(customers)))
    eligible = scored.loc[scored["expected_net_value"] > 0]
    selections = {"no_contact": scored.iloc[:0]}
    for policy, ranking in (
        ("risk_positive_only", "churn_probability"),
        ("expected_profit_positive_only", "expected_net_value"),
    ):
        selections[policy] = eligible.sort_values(
            [ranking, "customer_id"], ascending=[False, True]
        ).head(capacity)

    results = []
    for policy, selected in selections.items():
        cost = float(len(selected) * scenario.contact_cost)
        contribution = float(selected["expected_retained_contribution"].sum())
        net = contribution - cost
        results.append({
            "policy": policy,
            "capacity_customers": capacity,
            "selected_customers": len(selected),
            "selected_fraction": len(selected) / len(customers),
            "unused_capacity": capacity - len(selected),
            "expected_churners_contacted": float(selected["churn_probability"].sum()),
            "expected_prevented_churn": float(selected["expected_prevented_churn"].sum()),
            "expected_retained_contribution": contribution,
            "campaign_cost": cost,
            "expected_net_value": net,
            "scenario_roi": net / cost if cost > 0 else None,
        })
    return scored, pd.DataFrame(results)


def main():
    if mlflow.active_run() is not None:
        raise RuntimeError("Start without an active MLflow run.")
    local_config = load_training_config()
    database = Path(local_config.mlflow.tracking_directory) / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database}")
    scenario = CampaignScenario()
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    try:
        client = MlflowClient()
        source = client.get_run(SOURCE_RUN_ID)
        if (
            source.info.status != "FINISHED"
            or source.data.tags.get("model") != "catboost"
            or source.data.tags.get("evaluation_partition") != "validation"
            or float(source.data.params["positive_class_weight"]) != 1
        ):
            raise ValueError("Expected the completed unweighted CatBoost finalist.")
        parent_id = source.data.tags["mlflow.parentRunId"]
        if client.get_run(parent_id).info.status != "FINISHED":
            raise ValueError("Source finalist comparison did not finish.")
        config = TrainingConfig.model_validate(
            read_json(client, parent_id, "evaluation_config.json")
        )
        provenance = current_provenance(config)
        saved = read_json(client, parent_id, "input_provenance.json")
        if any(
            provenance[name]["sha256"] != saved[name]["sha256"]
            for name in provenance
        ):
            raise ValueError("Source data or feature SQL changed.")
        comparisons = read_json(client, parent_id, "validation_comparison.json")
        matches = [row for row in comparisons if row["run_id"] == SOURCE_RUN_ID]
        if len(matches) != 1 or not matches[0]["model_uri"]:
            raise ValueError("Expected exactly one saved CatBoost model URI.")
        model = mlflow.sklearn.load_model(matches[0]["model_uri"])
        with duckdb.connect() as connection:
            load_customers(connection)
            load_split_views(connection, Path(config.data.split_manifest))
            load_feature_view(connection, Path(config.data.feature_sql))
            validation = connection.sql("""
                SELECT * FROM baseline_features
                WHERE partition = 'validation'
                ORDER BY customer_id
            """).df()
        required = set(FEATURE_COLUMNS) | {"customer_id", "churn", "partition"}
        if (
            validation.empty
            or not validation.columns.is_unique
            or set(validation.columns) != required
            or not validation["partition"].eq("validation").all()
            or validation["churn"].isna().any()
            or set(validation["churn"].unique()) != {0, 1}
        ):
            raise ValueError("Invalid validation data.")
        positive = np.flatnonzero(model.classes_ == 1)
        if len(positive) != 1:
            raise ValueError("Expected one positive class.")
        probabilities = model.predict_proba(
            validation.loc[:, list(FEATURE_COLUMNS)]
        )[:, positive[0]]
        scored, summary = compare_policies(
            validation[["customer_id", "monthlycharges"]], probabilities, scenario
        )
        for name, function in (
            ("log_loss", log_loss), ("brier_score", brier_score_loss)
        ):
            if not np.isclose(
                function(validation["churn"], probabilities),
                source.data.metrics[f"validation_{name}"], rtol=1e-8, atol=1e-10,
            ):
                raise ValueError(f"Saved finalist {name} was not reproduced.")

        with mlflow.start_run(
            experiment_id=source.info.experiment_id,
            run_name="catboost_validation_campaign_scenario",
        ) as run:
            mlflow.set_tags({
                "mode": "financial_scenario",
                "source_finalist_run_id": SOURCE_RUN_ID,
                "evaluation_partition": "validation",
                "not_measured_causal_roi": "true",
            })
            mlflow.log_params(asdict(scenario))
            mlflow.log_dict(asdict(scenario), "campaign_scenario.json")
            mlflow.log_dict(provenance, "input_provenance.json")
            mlflow.log_dict({
                "ranking": "expected_net_value_desc_customer_id_asc",
                "eligibility": "expected_net_value > 0",
                "capacity": "floor(capacity_fraction * population_size)",
                "cost": "incurred for every contacted customer",
                "effect": "constant assumed prevention probability among would-be churners",
                "value": "monthlycharges * gross_margin * retained_months",
                "limitations": "No measured treatment effects or established prediction horizon.",
            }, "policy_definition.json")
            records = summary.astype(object).where(summary.notna(), None).to_dict(
                orient="records"
            )
            mlflow.log_dict(records, "policy_comparison.json")
            for row in records:
                mlflow.log_metrics({
                    f"{row['policy']}_{key}": value
                    for key, value in row.items()
                    if key != "policy" and value is not None
                })
            mlflow.log_metric(
                "positive_expected_value_customers",
                int((scored["expected_net_value"] > 0).sum()),
            )
            from experiments.sensitivity_analysis import (
                CONTACT_COSTS,
                PREVENTION_PROBABILITIES,
                RETAINED_MONTHS,
                advantage_figure,
                run_sensitivity,
            )

            sensitivity = run_sensitivity(
                validation[["customer_id", "monthlycharges"]],
                probabilities,
                scenario,
            )
            mlflow.log_text(
                sensitivity.to_csv(index=False),
                "sensitivity/scenario_comparison.csv",
            )
            mlflow.log_text(
                advantage_figure(sensitivity).to_html(
                    include_plotlyjs=True,
                    full_html=True,
                ),
                "sensitivity/profit_advantage_6months.html",
            )
            mlflow.log_dict(
                {
                    "contact_costs": list(CONTACT_COSTS),
                    "prevention_probabilities": list(PREVENTION_PROBABILITIES),
                    "retained_months": list(RETAINED_MONTHS),
                    "gross_margin": scenario.gross_margin,
                    "capacity_fraction": scenario.capacity_fraction,
                    "audience_selection": "Reselected independently per scenario",
                    "interpretation": "Assumption-based estimates, not measured ROI",
                },
                "sensitivity/grid_definition.json",
            )
            print(f"MLflow scenario run: {run.info.run_id}")
        print("ASSUMPTION-BASED EXPECTED VALUES, NOT MEASURED CAMPAIGN RETURNS")
        print(pd.Series(asdict(scenario)).to_string())
        print(summary.to_string(index=False))
    finally:
        mlflow.set_tracking_uri(previous_uri)


if __name__ == "__main__":
    main()
