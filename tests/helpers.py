import yaml

from src.config import DEFAULT_CONFIG_PATH, TrainingConfig


def model_config() -> TrainingConfig:
    """Validate model settings without requiring real training input files."""
    return TrainingConfig.model_validate(
        yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    )
