"""Export a smaller synthetic demo without rescoring or refitting thresholds."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.dashboard import DemoMetadata, open_portfolio
from src.models.artifacts import file_sha256
from src.models.serving_bundle import export_bundle, load_bundle


def export_demo(source: Path, output: Path, *, rows: int = 10_000, seed: int = 42) -> Path:
    """Keep proportional partition counts and original full-training band cuts."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 3:
        raise ValueError("Demo rows must be an integer of at least 3.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Demo seed must be an integer in [0, 2**32).")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Demo output already exists: {output}")
    bundle = load_bundle(source)
    directory = bundle.portfolio_directory
    manifest_path = directory / "manifest.json"
    original_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Create demos from a full snapshot, not another demo.")
    counts = manifest["partition_rows"]
    if set(counts) != {"train", "validation", "test"} or rows >= manifest["rows"]:
        raise ValueError("Demo requires all source partitions and fewer rows than the source.")
    quotas = {name: rows * count // manifest["rows"] for name, count in counts.items()}
    remainder = rows - sum(quotas.values())
    order = sorted(counts, key=lambda name: (
        -(rows * counts[name] % manifest["rows"]), name,
    ))
    for name in order[:remainder]:
        quotas[name] += 1
    if min(quotas.values()) < 1:
        raise ValueError("Increase demo rows to include every source partition.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-demo-", dir=output.parent) as temporary:
        snapshot = Path(temporary) / "snapshot"
        snapshot.mkdir()
        with open_portfolio(directory) as connection:
            clauses = []
            parameters = []
            for name in sorted(quotas):
                clauses.append(
                    "(SELECT * FROM portfolio WHERE partition = ? "
                    "ORDER BY sha256(? || ':' || customer_id), customer_id LIMIT ?)"
                )
                parameters.extend((name, str(seed), quotas[name]))
            connection.sql(
                "SELECT * FROM (" + " UNION ALL ".join(clauses) + ") ORDER BY customer_id",
                params=parameters,
            ).write_parquet(str(snapshot / "customers.parquet"), compression="zstd")
        shutil.copyfile(directory / "thresholds.json", snapshot / "thresholds.json")
        metadata = DemoMetadata(
            selection="partition_proportional_sha256_v1", seed=seed,
            source_rows=manifest["rows"], source_partition_rows=counts,
            source_manifest_sha256=original_hash,
            source_thresholds_sha256=file_sha256(directory / "thresholds.json"),
        )
        demo_manifest = {
            key: value for key, value in manifest.items()
            if key not in {"input_provenance", "created_at_utc", "files", "rows", "partition_rows"}
        }
        demo_manifest.update(
            schema_version=2, demo=metadata.model_dump(),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            rows=rows, partition_rows=quotas,
            files={name: {"sha256": file_sha256(snapshot / name)}
                   for name in ("customers.parquet", "thresholds.json")},
        )
        (snapshot / "manifest.json").write_text(
            json.dumps(demo_manifest, indent=2, allow_nan=False), encoding="utf-8",
        )
        if file_sha256(manifest_path) != original_hash:
            raise ValueError("Source manifest changed during demo export.")
        with open_portfolio(snapshot):
            pass
        # export_bundle checks prediction/explanation parity before publication.
        return export_bundle(bundle.predictor, snapshot, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Saved public demo bundle: {export_demo(args.source, args.output, rows=args.rows, seed=args.seed)}")


if __name__ == "__main__":
    main()
