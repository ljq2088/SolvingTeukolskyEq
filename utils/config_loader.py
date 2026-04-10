# YAML配置文件加载与合并工具

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str, base_config: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Supports inheritance via 'defaults' key.

    Args:
        config_path: Path to config YAML file
        base_config: Optional base config to merge with

    Returns:
        Merged configuration dict
    """
    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle defaults/inheritance
    if 'defaults' in config:
        defaults = config.pop('defaults')
        base_configs = []

        for default in defaults:
            if isinstance(default, str):
                # Load base config
                base_path = config_path.parent / f"{default}.yaml"
                if base_path.exists():
                    base_configs.append(load_config(str(base_path)))

        # Merge: base configs first, then current config
        merged = {}
        for base in base_configs:
            merged = deep_merge(merged, base)
        merged = deep_merge(merged, config)

        return merged

    return config


def deep_merge(base: Dict, update: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        update: Dictionary with updates

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dict
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
