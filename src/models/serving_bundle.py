r"""Export an immutable serving bundle without retraining or model evaluation.

Run: uv run python -m src.models.serving_bundle --output artifacts\\serving_v1
Serve: set CHURN_BUNDLE_DIRECTORY to that directory before starting FastAPI.

Deploy this directory alongside the application and its locked dependencies.
Only trusted bundles are supported; checksums detect damage, not authenticity.
Native CatBoost replaces pickle; original MLflow URIs are audit metadata only.
"""

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from catboost import CatBoostClassifier
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.dashboard import open_portfolio
from src.data.loader import PROJECT_ROOT
from src.models.artifacts import file_sha256
from src.models.explain import ChurnExplainer
from src.models.model_builder import prepare_catboost_features
from src.models.pipeline import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from src.models.predict import ChurnPredictor, ModelProvenance


FILES = (
    "model.cbm", "portfolio/customers.parquet",
    "portfolio/thresholds.json", "portfolio/manifest.json",
)


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    model_format: Literal["catboost_cbm"]
    preprocessing: Literal["prepare_catboost_features_v1"]
    created_at_utc: str
    provenance: ModelProvenance
    feature_columns: list[str]
    categorical_columns: list[str]
    package_versions: dict[str, str]
    files: dict[str, str]


@dataclass(frozen=True)
class ServingBundle:
    predictor: ChurnPredictor
    portfolio_directory: Path


def _validate_model(predictor: ChurnPredictor) -> CatBoostClassifier:
    ChurnExplainer(predictor)
    model = predictor.model
    if not isinstance(model, Pipeline):
        raise ValueError("Bundle requires the selected CatBoost pipeline.")
    preprocessor = model.named_steps["preprocessor"]
    if (
        not isinstance(preprocessor, FunctionTransformer)
        or preprocessor.get_params()
        != FunctionTransformer(prepare_catboost_features).get_params()
    ):
        raise ValueError("Bundle requires unchanged stateless CatBoost preprocessing.")
    classifier = model.named_steps["classifier"]
    if classifier.get_cat_feature_indices() != [
        FEATURE_COLUMNS.index(name) for name in CATEGORICAL_COLUMNS
    ]:
        raise ValueError("CatBoost categorical features differ from the serving contract.")
    return classifier


def load_bundle(directory: Path) -> ServingBundle:
    """Load native artifacts locally; never resolve the recorded MLflow URI."""
    manifest = BundleManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = manifest.provenance
    if (
        manifest.feature_columns != list(FEATURE_COLUMNS)
        or manifest.categorical_columns != list(CATEGORICAL_COLUMNS)
        or set(manifest.files) != set(FILES)
        or provenance.calibration != "none"
        or not all(isinstance(value, str) and value.strip() for value in (
            provenance.model_uri, provenance.source_finalist_run_id,
            provenance.freeze_run_id, provenance.final_test_run_id,
        ))
    ):
        raise ValueError("Unsupported or inconsistent serving bundle contract.")
    for name in FILES:
        if file_sha256(directory / name) != manifest.files[name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    classifier = CatBoostClassifier(thread_count=1)
    classifier.load_model(str(directory / "model.cbm"), format="cbm")
    predictor = ChurnPredictor(Pipeline([
        ("preprocessor", FunctionTransformer(prepare_catboost_features)),
        ("classifier", classifier),
    ]), provenance)
    _validate_model(predictor)
    if file_sha256(directory / "model.cbm") != manifest.files["model.cbm"]:
        raise ValueError("Bundle model changed during loading.")
    snapshot = json.loads(
        (directory / "portfolio" / "manifest.json").read_text(encoding="utf-8")
    )
    if snapshot["model"] != asdict(provenance):
        raise ValueError("Bundle model and portfolio provenance differ.")
    return ServingBundle(predictor, directory / "portfolio")


def export_bundle(
    predictor: ChurnPredictor, portfolio_directory: Path, output: Path,
) -> Path:
    """Verify, copy, round-trip check, then atomically publish a new directory."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Bundle output already exists: {output}")
    classifier = _validate_model(predictor)
    snapshot_paths = {
        name: portfolio_directory / name
        for name in ("customers.parquet", "thresholds.json", "manifest.json")
    }
    original_hashes = {name: file_sha256(path) for name, path in snapshot_paths.items()}
    with open_portfolio(portfolio_directory) as connection:
        columns = ", ".join(f'"{name}"' for name in ("customer_id", *FEATURE_COLUMNS))
        sample = connection.sql(
            f"SELECT {columns} FROM portfolio WHERE partition = 'train' "
            "ORDER BY customer_id LIMIT 32"
        ).df()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "portfolio").mkdir()
        for name, path in snapshot_paths.items():
            shutil.copyfile(path, staging / "portfolio" / name)
            if (
                file_sha256(path) != original_hashes[name]
                or file_sha256(staging / "portfolio" / name) != original_hashes[name]
            ):
                raise ValueError(f"Source snapshot changed during export: {name}")
        classifier.save_model(str(staging / "model.cbm"), format="cbm")
        manifest = BundleManifest(
            schema_version=1, model_format="catboost_cbm",
            preprocessing="prepare_catboost_features_v1",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            provenance=predictor.provenance,
            feature_columns=list(FEATURE_COLUMNS),
            categorical_columns=list(CATEGORICAL_COLUMNS),
            package_versions={
                name: version(name)
                for name in ("catboost", "scikit-learn", "numpy", "pandas", "duckdb")
            },
            files={name: file_sha256(staging / name) for name in FILES},
        )
        (staging / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )
        restored = load_bundle(staging).predictor
        with open_portfolio(staging / "portfolio"):
            pass
        expected = predictor.predict_batch(sample)
        actual = restored.predict_batch(sample)
        if not expected.drop(columns="churn_probability").equals(
            actual.drop(columns="churn_probability")
        ) or not np.allclose(
            expected["churn_probability"], actual["churn_probability"],
            rtol=1e-12, atol=1e-14,
        ):
            raise ValueError("Exported model predictions differ from the frozen model.")
        customer = sample.iloc[0].to_dict()
        before = ChurnExplainer(predictor).explain_one(customer)
        after = ChurnExplainer(restored).explain_one(customer)
        if not np.allclose(
            [before.base_log_odds, before.raw_log_odds,
             *before.contributions["shap_log_odds"]],
            [after.base_log_odds, after.raw_log_odds,
             *after.contributions["shap_log_odds"]],
            rtol=1e-12, atol=1e-14,
        ):
            raise ValueError("Exported model explanations differ from the frozen model.")
        if output.exists():
            raise FileExistsError(f"Bundle output already exists: {output}")
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-directory", type=Path, default=PROJECT_ROOT / "mlruns")
    parser.add_argument(
        "--portfolio-directory", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "portfolio_v1",
    )
    parser.add_argument("--final-test-run-id", default="5818be9683af42a485ffc2896a868137")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Bundle output already exists: {args.output}")
    predictor = ChurnPredictor.from_final_test(
        args.tracking_directory, final_test_run_id=args.final_test_run_id,
    )
    destination = export_bundle(predictor, args.portfolio_directory, args.output)
    print(f"Saved verified serving bundle: {destination}")


if __name__ == "__main__":
    main()
