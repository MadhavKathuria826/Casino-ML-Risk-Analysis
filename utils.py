"""Utility helpers for the casino ML project."""

from __future__ import annotations

import ast
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RANDOM_STATE = 42


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Set deterministic random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(output_dir: str | Path) -> logging.Logger:
    """Configure project logging to both stdout and a log file."""
    output_dir = ensure_dir(output_dir)
    log_path = output_dir / "project.log"

    logger = logging.getLogger("casino_ml")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def safe_literal_eval(value: Any, default: Any = None) -> Any:
    """Safely parse a Python literal embedded in a string column."""
    if value is None:
        return default
    if isinstance(value, (list, dict, tuple, int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return default
    return default


def flatten_nested_list(value: Any) -> list[Any]:
    """Flatten nested list-like structures into a simple list."""
    if value is None:
        return []
    if not isinstance(value, list):
        return [value]

    flattened: list[Any] = []
    stack = list(value)
    while stack:
        current = stack.pop(0)
        if isinstance(current, list):
            stack = current + stack
        else:
            flattened.append(current)
    return flattened


def shannon_entropy(values: list[Any]) -> float:
    """Compute Shannon entropy for a sequence of categorical values."""
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities + 1e-12)).sum())


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON artifact with stable formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_name(value: str) -> str:
    """Normalize free-text player names or labels."""
    return " ".join(str(value).strip().split())

