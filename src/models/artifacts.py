"""Local experiment artifacts and input fingerprints; importing does no I/O."""

import hashlib
import json
from pathlib import Path

from src.data.loader import DATA_PATH


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_json(client, run_id, filename):
    path = client.download_artifacts(run_id, filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def current_provenance(config):
    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in {
            "raw_data": DATA_PATH,
            "split_manifest": Path(config.data.split_manifest),
            "feature_sql": Path(config.data.feature_sql),
        }.items()
    }
