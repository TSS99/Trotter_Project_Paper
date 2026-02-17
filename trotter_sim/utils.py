from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

def json_error(message: str, param: str | None = None) -> None:
    """Raise a ValueError whose message is a JSON string (matches notebook behavior)."""
    err: Dict[str, Any] = {"status": "error", "message": message}
    if param is not None:
        err["param"] = param
    raise ValueError(json.dumps(err))

def is_power_of_two(n: int) -> bool:
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0

def validate_common(config: Dict[str, Any]) -> Tuple[int, int, float]:
    """
    Validate {n_discretize, n_steps, total_time} exactly like the notebook.
    Returns (n_discretize, n_steps, total_time).
    """
    required = ["n_discretize", "n_steps", "total_time"]
    for key in required:
        if key not in config:
            json_error(f"Missing required parameter '{key}'", param=key)

    try:
        n_discretize = int(config["n_discretize"])
    except Exception:
        json_error("n_discretize must be an integer", param="n_discretize")
    if n_discretize < 4:
        json_error("n_discretize must be >= 4", param="n_discretize")
    if not is_power_of_two(n_discretize):
        json_error("n_discretize must be a power of two (2^k)", param="n_discretize")

    try:
        n_steps = int(config["n_steps"])
    except Exception:
        json_error("n_steps must be an integer", param="n_steps")
    if n_steps < 1:
        json_error("n_steps must be >= 1", param="n_steps")

    try:
        total_time = float(config["total_time"])
    except Exception:
        json_error("total_time must be a number", param="total_time")
    if total_time <= 0:
        json_error("total_time must be > 0", param="total_time")

    return n_discretize, n_steps, total_time

@dataclass(frozen=True)
class RunConfig:
    """Convenience typed wrapper around your JSON config."""
    well: Dict[str, Any]
    ho: Dict[str, Any]

    @staticmethod
    def from_json(path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "well" not in cfg or "ho" not in cfg:
            json_error("Config JSON must have top-level keys: 'well' and 'ho'", param="config")
        return RunConfig(well=cfg["well"], ho=cfg["ho"])
