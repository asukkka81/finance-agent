# data_layer/config.py
"""全局配置加载模块."""

from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    """应用配置 — 从 YAML 文件加载并暴露为属性."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def db_path(self) -> str:
        return self._data["database"]["path"]

    @property
    def db_pragma(self) -> dict[str, str]:
        return self._data["database"]["pragma"]

    @property
    def yfinance_config(self) -> dict[str, Any]:
        return self._data["data_sources"]["yfinance"]

    @property
    def akshare_config(self) -> dict[str, Any]:
        return self._data["data_sources"]["akshare"]

    @property
    def us_symbols(self) -> list[str]:
        return self._data["sync"]["us_symbols"]

    @property
    def cn_symbols(self) -> list[str]:
        return self._data["sync"]["cn_symbols"]

    @property
    def fund_codes(self) -> list[str]:
        return self._data["sync"]["fund_codes"]

    @property
    def macro_indicators(self) -> list[str]:
        return self._data["sync"]["macro_indicators"]

    @property
    def logging_config(self) -> dict[str, str]:
        return self._data["logging"]

    def get(self, *keys: str, default: Any = None) -> Any:
        """按路径读取配置值. 例如 config.get('sync', 'us_symbols')."""
        node = self._data
        for key in keys:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return default
            if node is None:
                return default
        return node


def load_config(path: Optional[str] = None) -> Config:
    """加载配置文件.

    Args:
        path: YAML 配置文件的路径. 默认使用 `config/settings.yaml`.

    Returns:
        Config 实例.

    Raises:
        FileNotFoundError: 配置文件不存在时抛出.
    """
    if path is None:
        # 相对于项目根目录
        project_root = Path(__file__).resolve().parent.parent
        path = str(project_root / "config" / "settings.yaml")

    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(data)
