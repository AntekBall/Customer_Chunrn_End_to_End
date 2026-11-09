import json
import shutil
import subprocess
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.dashboard import open_portfolio
from src.data.demo import export_demo
from src.data.loader import PROJECT_ROOT
from src.models.artifacts import file_sha256
from src.models.serving_bundle import export_bundle, load_bundle
from src.web import create_app
from tests.test_api import fitted, setup


@pytest.fixture
def source(setup, tmp_path):
    _, _, predictor, _, snapshot = setup
    return export_bundle(predictor, snapshot, tmp_path / "source")


def test_demo_is_deterministic_subset_with_original_thresholds(source, tmp_path):
    source_hash = file_sha256(source / "portfolio" / "customers.parquet")
    first = export_demo(source, tmp_path / "first", rows=16)
    second = export_demo(source, tmp_path / "second", rows=16)
    assert file_sha256(first / "portfolio" / "customers.parquet") == file_sha256(
        second / "portfolio" / "customers.parquet"
    )
    assert file_sha256(source / "portfolio" / "customers.parquet") == source_hash
    assert (first / "portfolio" / "thresholds.json").read_bytes() == (
        source / "portfolio" / "thresholds.json"
    ).read_bytes()
    with open_portfolio(source / "portfolio") as connection:
        original = connection.sql("SELECT * FROM portfolio ORDER BY customer_id").df()
    with open_portfolio(first / "portfolio") as connection:
        sample = connection.sql("SELECT * FROM portfolio ORDER BY customer_id").df()
    assert len(sample) == 16
    assert sample["partition"].value_counts().to_dict() == {
        "train": 12, "validation": 2, "test": 2,
    }
    pd.testing.assert_frame_equal(
        sample, original.loc[original["customer_id"].isin(sample["customer_id"])].reset_index(drop=True),
    )
    manifest = json.loads((first / "portfolio" / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["demo"]["source_rows"] == 40
    assert "input_provenance" not in manifest
    assert load_bundle(first).predictor.provenance == load_bundle(source).predictor.provenance


def test_single_service_serves_dashboard_and_api(source, tmp_path):
    demo = export_demo(source, tmp_path / "demo", rows=16)
    with TestClient(create_app(bundle_directory=demo, port=18082)) as client:
        assert client.get("/").status_code == 200
        layout = client.get("/_dash-layout")
        assert layout.status_code == 200
        assert "PUBLIC DEMO" in layout.text
        assert "16 sampled synthetic customers from 40" in layout.text
        assert "not the full dataset" in layout.text
        assert client.get("/_dash-dependencies").status_code == 200
        assert client.get("/health").json()["portfolio_customers"] == 16
        assert client.get("/portfolio/summary").json()["overview"]["customers"] == 16
        simulation = client.post("/campaign/simulate", json={"scenario": {}})
        assert simulation.status_code == 200
        assert simulation.json()["global_summary"]["customers"] == 16
        assert simulation.json()["capacity_customers"] == 1
        oversized = client.post(
            "/predict", content=b"x" * (2 * 1024 * 1024 + 1),
            headers={"content-type": "application/json"},
        )
        assert oversized.status_code == 413
        assert "limited to 2 MiB" in oversized.json()["detail"]


@pytest.mark.parametrize("rows", [0, 1, 2, 40, 41, True, 2.5])
def test_invalid_demo_size(source, tmp_path, rows):
    with pytest.raises(ValueError):
        export_demo(source, tmp_path / "bad", rows=rows)
    assert not (tmp_path / "bad").exists()


def test_different_seed_and_no_overwrite(source, tmp_path):
    first = export_demo(source, tmp_path / "first", rows=16, seed=42)
    second = export_demo(source, tmp_path / "second", rows=16, seed=43)
    assert file_sha256(first / "portfolio" / "customers.parquet") != file_sha256(
        second / "portfolio" / "customers.parquet"
    )
    with pytest.raises(FileExistsError):
        export_demo(source, first, rows=16)
    with pytest.raises(ValueError, match="not another demo"):
        export_demo(first, tmp_path / "nested", rows=10)


def test_full_snapshot_cannot_be_exposed_as_public_demo(source):
    with pytest.raises(ValueError, match="labelled demo"):
        create_app(bundle_directory=source, port=8000)


def test_corrupt_demo_reference_is_rejected(source, tmp_path):
    demo = export_demo(source, tmp_path / "demo", rows=16)
    path = demo / "portfolio" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["demo"]["source_partition_rows"]["train"] = 12
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid public demo"):
        with open_portfolio(demo / "portfolio"):
            pytest.fail("Invalid reference accepted.")


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_demo_seed(source, tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        export_demo(source, tmp_path / "bad", rows=16, seed=seed)


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_public_bundle_survives_git_storage(tmp_path, autocrlf):
    repository = tmp_path / "repository"
    repository.mkdir()
    shutil.copyfile(PROJECT_ROOT / ".gitattributes", repository / ".gitattributes")
    original = PROJECT_ROOT / "deploy" / "demo_bundle"
    shutil.copytree(original, repository / "deploy" / "demo_bundle")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], stderr=subprocess.STDOUT,
        )

    git("init")
    git("config", "core.autocrlf", autocrlf)
    git("add", ".")
    tree = git("write-tree").decode().strip()
    archive = tmp_path / "bundle.zip"
    git("archive", "--format=zip", f"--output={archive}", tree)
    checkout = tmp_path / "checkout"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(checkout)
    restored = checkout / "deploy" / "demo_bundle"
    for path in original.rglob("*"):
        if path.is_file():
            assert (restored / path.relative_to(original)).read_bytes() == path.read_bytes()
    load_bundle(restored)
    with TestClient(create_app(bundle_directory=restored, port=18082)) as client:
        assert client.get("/health").json()["portfolio_customers"] == 10_000
