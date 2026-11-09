import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models.artifacts import file_sha256
from src.models.explain import ChurnExplainer
from src.models.predict import ChurnPredictor
from src.models.serving_bundle import FILES, export_bundle, load_bundle
from tests.test_api import fitted, setup


@pytest.fixture
def bundle(setup, tmp_path):
    _, customer, predictor, _, snapshot = setup
    directory = export_bundle(predictor, snapshot, tmp_path / "bundle")
    return directory, customer, predictor


def test_native_roundtrip_and_relocation(bundle, tmp_path):
    directory, customer, original = bundle
    moved = tmp_path / "relocated"
    shutil.move(str(directory), moved)
    restored = load_bundle(moved)
    data = pd.DataFrame([
        customer,
        customer | {"customer_id": "unseen", "contract": "new-category"},
    ])
    pd.testing.assert_frame_equal(
        original.predict_batch(data), restored.predictor.predict_batch(data),
        rtol=1e-12, atol=1e-14,
    )
    before = ChurnExplainer(original).explain_one(customer)
    after = ChurnExplainer(restored.predictor).explain_one(customer)
    pd.testing.assert_frame_equal(before.contributions, after.contributions)
    assert before.base_log_odds == pytest.approx(after.base_log_odds, abs=1e-14)
    assert restored.predictor.provenance == original.provenance
    assert restored.portfolio_directory == moved / "portfolio"
    assert {path.relative_to(moved).as_posix() for path in moved.rglob("*")
            if path.is_file()} == {*FILES, "manifest.json"}


def test_api_serves_bundle_without_tracking_or_source_snapshot(
    bundle, setup, monkeypatch, tmp_path,
):
    directory, customer, predictor = bundle
    setup[-1].rename(tmp_path / "offline-source-snapshot")
    monkeypatch.setattr(
        ChurnPredictor, "from_final_test",
        classmethod(lambda *args, **kwargs: pytest.fail("MLflow must not be used")),
    )
    monkeypatch.setenv("CHURN_BUNDLE_DIRECTORY", str(directory))
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["portfolio_customers"] == 40
        assert client.post("/predict", json=customer).json() == predictor.predict_one(customer)
        batch = client.post("/predict/batch", json={"customers": [customer]})
        assert batch.json()["predictions"] == [predictor.predict_one(customer)]
        explanation = client.post("/explain", json=customer)
        assert explanation.status_code == 200
        assert len(explanation.json()["contributions"]) == 28
        summary = client.get("/portfolio/summary")
        assert summary.status_code == 200
        assert summary.json()["overview"]["customers"] == 40
        campaign = client.post("/campaign/simulate", json={"scenario": {}})
        assert campaign.status_code == 200


@pytest.mark.parametrize("filename", FILES)
def test_rejects_damaged_artifacts(bundle, filename):
    directory, _, _ = bundle
    path = directory / filename
    with path.open("ab") as file:
        file.write(b"damage")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(directory)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2),
    ("model_format", "pickle"),
    ("preprocessing", "other"),
    ("feature_columns", ["age"]),
    ("categorical_columns", []),
    ("files", {"../outside": "abc"}),
])
def test_rejects_incompatible_contract(bundle, field, value):
    directory, _, _ = bundle
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_bundle(directory)


def test_rejects_mismatched_portfolio_identity(bundle):
    directory, _, _ = bundle
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provenance"]["freeze_run_id"] = "different"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance differ"):
        load_bundle(directory)


def test_rejects_internally_invalid_snapshot_at_startup(bundle):
    directory, _, _ = bundle
    snapshot_path = directory / "portfolio" / "manifest.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["rows"] += 1
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["portfolio/manifest.json"] = file_sha256(snapshot_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Snapshot rows"):
        with TestClient(create_app(bundle_directory=directory)):
            pytest.fail("Invalid snapshot must not reach readiness.")


def test_no_overwrite(bundle, setup):
    directory, _, predictor = bundle
    before = file_sha256(directory / "manifest.json")
    with pytest.raises(FileExistsError):
        export_bundle(predictor, setup[-1], directory)
    assert file_sha256(directory / "manifest.json") == before


def test_failed_export_does_not_publish(setup, tmp_path, monkeypatch):
    _, _, predictor, _, snapshot = setup
    output = tmp_path / "failed"
    classifier = predictor.model.named_steps["classifier"]
    def fail_save(*args, **kwargs):
        raise RuntimeError("synthetic write failure")
    monkeypatch.setattr(classifier, "save_model", fail_save)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        export_bundle(predictor, snapshot, output)
    assert not output.exists()
    assert list(tmp_path.glob(".failed-*")) == []


def test_rejects_changed_preprocessor(setup, tmp_path):
    _, _, predictor, _, snapshot = setup
    predictor.model.named_steps["preprocessor"].set_params(kw_args={"unexpected": True})
    try:
        with pytest.raises(ValueError, match="unchanged stateless"):
            export_bundle(predictor, snapshot, tmp_path / "invalid")
    finally:
        predictor.model.named_steps["preprocessor"].set_params(kw_args=None)


@pytest.mark.parametrize("setting", [
    "CHURN_TRACKING_DIRECTORY", "CHURN_PORTFOLIO_DIRECTORY", "CHURN_FINAL_TEST_RUN_ID",
])
def test_rejects_ambiguous_environment(monkeypatch, tmp_path, setting):
    monkeypatch.setenv(setting, "conflict")
    with pytest.raises(ValueError, match="not both"):
        create_app(bundle_directory=tmp_path)


def test_rejects_ambiguous_arguments(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        create_app(bundle_directory=tmp_path, tracking_directory=tmp_path)


def test_missing_bundle_fails_instead_of_falling_back(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ChurnPredictor, "from_final_test",
        classmethod(lambda *args, **kwargs: pytest.fail("No silent MLflow fallback")),
    )
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(bundle_directory=tmp_path / "missing")):
            pytest.fail("Missing bundle must not reach readiness.")


def test_blank_bundle_configuration_is_not_a_fallback(monkeypatch):
    monkeypatch.setenv("CHURN_BUNDLE_DIRECTORY", " ")
    with pytest.raises(ValueError, match="must not be blank"):
        create_app()
