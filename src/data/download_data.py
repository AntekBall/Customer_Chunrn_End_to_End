from pathlib import Path

import kagglehub


DATASET = "isandeep06/customer-churn-prediction-dataset-1m"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "customer-churn-1m"


def main() -> None:
    DOWNLOAD_DIR.parent.mkdir(parents=True, exist_ok=True)

    download_path = Path(
        kagglehub.dataset_download(
            DATASET,
            output_dir=str(DOWNLOAD_DIR),
        )
    )

    files = sorted(path for path in download_path.rglob("*") if path.is_file())

    if not files:
        raise RuntimeError(f"No dataset files found in {download_path}")

    print(f"Dataset downloaded to: {download_path}")

    for path in files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.relative_to(download_path)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()