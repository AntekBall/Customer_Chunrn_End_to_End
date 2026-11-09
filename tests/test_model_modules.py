import ast
import subprocess
import sys

from src.data.loader import PROJECT_ROOT


def test_source_modules_do_not_import_experiments():
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(
                name == "experiments" or name.startswith("experiments.")
                for name in names
            ), path


def test_shared_imports_need_no_workspace_data_or_tracking(tmp_path):
    script = """
import sys
from pathlib import Path
from src.models import artifacts, calibration, campaign, evaluate, sensitivity

assert not any(
    name == 'experiments' or name.startswith('experiments.')
    for name in sys.modules
)
assert campaign.CampaignScenario().contact_cost == 2.0
assert set(evaluate.METRIC_FUNCTIONS) == {
    'average_precision', 'roc_auc', 'log_loss', 'brier_score'
}
assert not list(Path.cwd().iterdir())
"""
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {str(PROJECT_ROOT)!r})\n{script}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_experiment_imports_reuse_source_implementations():
    from experiments import (
        check_calibration, final_test, finalist_validation,
        financial_policy, sensitivity_analysis,
    )
    from src.models import artifacts, calibration, campaign, evaluate, sensitivity

    assert financial_policy.CampaignScenario is campaign.CampaignScenario
    assert financial_policy.compare_policies is campaign.compare_policies
    assert check_calibration.calibration_report is calibration.calibration_report
    assert final_test.calibration_report is calibration.calibration_report
    assert final_test.compare_policies is campaign.compare_policies
    assert final_test.score_validation is evaluate.score_predictions
    assert finalist_validation.score_validation is evaluate.score_predictions
    assert finalist_validation.current_provenance is artifacts.current_provenance
    assert sensitivity_analysis.run_sensitivity is sensitivity.run_sensitivity
