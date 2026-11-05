from pathlib import Path

ROOT = Path(r"C:\Users\antek\Customer_Chunrn_End_to_End")

DIRECTORIES = [
    r".github\workflows",
    "sql",
    "experiments",
    r"src\data",
    r"src\models",
    r"src\api",
    r"src\frontend",
    "tests",
    r"data\raw",
    r"data\processed",
    "mlruns",
]

FILES = [
    r".github\workflows\ci-cd.yml",
    r"sql\01_raw_ingestion.sql",
    r"sql\02_feature_engineering.sql",
    r"sql\03_cohort_analysis.sql",
    r"experiments\01_eda_exploration.ipynb",
    r"experiments\02_optuna_tuning_sandbox.py",
    r"experiments\03_shap_scratchpad.ipynb",
    r"src\__init__.py",
    r"src\data\__init__.py",
    r"src\data\loader.py",
    r"src\models\__init__.py",
    r"src\models\train.py",
    r"src\models\evaluate.py",
    r"src\models\predict.py",
    r"src\api\__init__.py",
    r"src\api\main.py",
    r"src\frontend\__init__.py",
    r"src\frontend\app.py",
    r"tests\__init__.py",
    r"tests\test_data.py",
    r"tests\test_model.py",
    r"tests\test_api.py",
    "Dockerfile",
    "docker-compose.yml",
    "pyproject.toml",
    ".env.example",
    ".env",
    ".gitignore",
    ".dockerignore",
    "README.md",
]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    for directory in DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    created = 0

    for filename in FILES:
        path = ROOT / filename

        try:
            with path.open("xb"):
                pass
        except FileExistsError:
            if not path.is_file():
                raise
        else:
            created += 1

    print(f"Created {created} empty files in {ROOT}")
    print("Existing files were left unchanged.")


if __name__ == "__main__":
    main()