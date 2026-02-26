"""
config.py — Load YAML config, resolve stochastic/deterministic params.

Supports multi-class rigs: each rig class has its own market params,
timeline, and revenue model.
"""

import yaml
import numpy as np
from copy import deepcopy


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_stochastic(param) -> bool:
    return isinstance(param, dict) and "distribution" in param


def sample_param(param, rng: np.random.Generator) -> float:
    if not is_stochastic(param):
        return float(param)
    dist = param["distribution"]
    p = param.get("params", {})
    bounds = param.get("bounds", None)
    if dist == "triangular":
        value = rng.triangular(p["low"], p["mode"], p["high"])
    elif dist == "normal":
        value = rng.normal(p["mean"], p["std"])
    elif dist == "uniform":
        value = rng.uniform(p["low"], p["high"])
    elif dist == "lognormal":
        value = rng.lognormal(p["mean"], p["sigma"])
    elif dist == "beta":
        value = rng.beta(p["alpha"], p["beta"])
    elif dist == "poisson":
        value = rng.poisson(p["mean"])
    elif dist == "constant":
        value = p["value"]
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    if bounds is not None:
        value = np.clip(value, bounds[0], bounds[1])
    return float(value)


def mode_value(param) -> float:
    """Extract the mode (most likely) value from a distribution spec."""
    if not is_stochastic(param):
        return float(param)
    dist = param["distribution"]
    p = param.get("params", {})
    if dist == "triangular":
        return float(p["mode"])
    elif dist == "normal":
        return float(p["mean"])
    elif dist == "uniform":
        return float((p["low"] + p["high"]) / 2)
    elif dist == "constant":
        return float(p["value"])
    elif dist == "beta":
        a, b = p["alpha"], p["beta"]
        return float((a - 1) / (a + b - 2)) if (a > 1 and b > 1) else float(a / (a + b))
    elif dist == "poisson":
        return float(p["mean"])
    else:
        return sample_param(param, np.random.default_rng(0))


def _resolve_market(market: dict, func) -> dict:
    """Apply func to each stochastic param in a market dict."""
    resolved = {}
    for key, value in market.items():
        resolved[key] = func(value) if is_stochastic(value) else value
    return resolved


def resolve_stochastic(config: dict, rng: np.random.Generator) -> dict:
    """Resolve all stochastic params in rig_classes and economics."""
    resolved = deepcopy(config)
    if "rig_classes" in resolved:
        for cls_name, cls_cfg in resolved["rig_classes"].items():
            if "market" in cls_cfg:
                cls_cfg["market"] = _resolve_market(cls_cfg["market"], lambda v: sample_param(v, rng))
    if "market" in resolved:
        resolved["market"] = _resolve_market(resolved["market"], lambda v: sample_param(v, rng))
    if "economics" in resolved:
        for key, value in resolved["economics"].items():
            if is_stochastic(value):
                resolved["economics"][key] = sample_param(value, rng)
    return resolved


def resolve_deterministic(config: dict) -> dict:
    """Resolve all stochastic params to their mode (most likely) values."""
    resolved = deepcopy(config)
    if "rig_classes" in resolved:
        for cls_name, cls_cfg in resolved["rig_classes"].items():
            if "market" in cls_cfg:
                cls_cfg["market"] = _resolve_market(cls_cfg["market"], mode_value)
    if "market" in resolved:
        resolved["market"] = _resolve_market(resolved["market"], mode_value)
    if "economics" in resolved:
        for key, value in resolved["economics"].items():
            if is_stochastic(value):
                resolved["economics"][key] = mode_value(value)
    return resolved


def get_all_stochastic_params(config: dict) -> list[dict]:
    """Extract all stochastic parameters across rig classes and economics."""
    params = []
    if "rig_classes" in config:
        for cls_name, cls_cfg in config["rig_classes"].items():
            if "market" in cls_cfg:
                for key, value in cls_cfg["market"].items():
                    if is_stochastic(value):
                        params.append({"class": cls_name, "key": key, "spec": value})
    if "economics" in config:
        for key, value in config["economics"].items():
            if is_stochastic(value):
                params.append({"class": "economics", "key": key, "spec": value})
    return params
