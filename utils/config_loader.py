from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Load and manage configuration from YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get_task3_config(self) -> Dict[str, Any]:
        """Get Task 3 configuration (deprecated, use get_task11_config)."""
        return self.get_task11_config()

    def get_task4_config(self) -> Dict[str, Any]:
        """Get Task 4 configuration."""
        return self.config.get("task4", {})

    def get_task11_config(self) -> Dict[str, Any]:
        """Get Task 11 configuration."""
        return self.config.get("task11", {})

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration."""
        return self.config.get("global", {})

    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
