"""Configuration module for model parameters and training settings."""
import yaml
from pathlib import Path

def load_yaml(path: str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)