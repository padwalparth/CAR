"""
Configuration Loader and Validator.

Loads models.yaml, pipeline.yaml, and simulation.yaml.
Provides robust fallback parsing if PyYAML is not installed.
Validates required sections and paths with descriptive errors.
"""

import os
from typing import Any, Dict, Optional

# Try importing pyyaml, fallback to built-in simple parser if absent
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Lightweight recursive parser for basic YAML key-value pairs, nested blocks, and lists.
    Used when PyYAML is not installed.
    """
    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current_dict = stack[-1][1]

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()

            # Inline comment removal
            if "#" in val:
                val = val.split("#")[0].strip()

            if not val:
                new_dict: Dict[str, Any] = {}
                current_dict[key] = new_dict
                stack.append((indent, new_dict))
            else:
                # Value parsing
                parsed_val: Any = val
                if val.startswith("[") and val.endswith("]"):
                    items = [x.strip() for x in val[1:-1].split(",") if x.strip()]
                    parsed_items = []
                    for it in items:
                        if it.lower() == "true":
                            parsed_items.append(True)
                        elif it.lower() == "false":
                            parsed_items.append(False)
                        else:
                            try:
                                parsed_items.append(int(it))
                            except ValueError:
                                try:
                                    parsed_items.append(float(it))
                                except ValueError:
                                    parsed_items.append(it.strip("\"'"))
                    parsed_val = parsed_items
                elif val.lower() == "true":
                    parsed_val = True
                elif val.lower() == "false":
                    parsed_val = False
                elif val.lower() in ("null", "none"):
                    parsed_val = None
                else:
                    try:
                        parsed_val = int(val)
                    except ValueError:
                        try:
                            parsed_val = float(val)
                        except ValueError:
                            parsed_val = val.strip("\"'")

                current_dict[key] = parsed_val

    return root


def load_yaml_file(filepath: str) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if HAVE_YAML:
        data = yaml.safe_load(content)
        return data if isinstance(data, dict) else {}
    else:
        return _parse_simple_yaml(content)


def validate_models_config(cfg: Dict[str, Any]) -> None:
    """Validate models.yaml structure."""
    required_sections = ["road_model", "traffic_model", "pothole_model"]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Missing required model section in models.yaml: '{sec}'")
        if not isinstance(cfg[sec], dict):
            raise ValueError(f"Section '{sec}' in models.yaml must be a dictionary.")
        if "backend" not in cfg[sec]:
            raise ValueError(f"Model section '{sec}' missing required 'backend' field.")


def validate_pipeline_config(cfg: Dict[str, Any]) -> None:
    """Validate pipeline.yaml structure and parameter ranges."""
    required_sections = ["confidence_thresholds", "road_association", "tracking", "pothole_severity", "road_condition"]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Missing required pipeline section in pipeline.yaml: '{sec}'")

    # Validate confidence thresholds
    conf = cfg.get("confidence_thresholds", {})
    for k, v in conf.items():
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence threshold '{k}' must be a float in range [0.0, 1.0], got: {v}")

    # Validate pothole severity thresholds
    sev = cfg.get("pothole_severity", {})
    s_max = sev.get("small_max_relative_area")
    m_max = sev.get("medium_max_relative_area")
    if s_max is not None and m_max is not None:
        if not (0.0 < s_max < m_max < 1.0):
            raise ValueError(
                f"Pothole severity thresholds must satisfy 0 < small ({s_max}) < medium ({m_max}) < 1.0"
            )

    # Validate road condition score thresholds
    rc = cfg.get("road_condition", {})
    thresh = rc.get("thresholds", {})
    good = thresh.get("good_min_score")
    mod = thresh.get("moderate_min_score")
    poor = thresh.get("poor_min_score")
    if good is not None and mod is not None and poor is not None:
        if not (0.0 <= poor < mod < good <= 100.0):
            raise ValueError(
                f"Road condition thresholds must satisfy 0 <= poor ({poor}) < moderate ({mod}) < good ({good}) <= 100.0"
            )

    # Validate traffic congestion thresholds if present
    ta = cfg.get("traffic_analysis", {})
    c_thresh = ta.get("congestion_thresholds", {})
    c_mod = c_thresh.get("moderate_density")
    c_hvy = c_thresh.get("heavy_density")
    c_grid = c_thresh.get("gridlock_density")
    if c_mod is not None and c_hvy is not None and c_grid is not None:
        if not (0.0 < c_mod < c_hvy < c_grid <= 1.0):
            raise ValueError(
                f"Congestion thresholds must satisfy 0 < moderate ({c_mod}) < heavy ({c_hvy}) < gridlock ({c_grid}) <= 1.0"
            )


def load_models_config(path: str = "config/models.yaml") -> Dict[str, Any]:
    cfg = load_yaml_file(path)
    validate_models_config(cfg)
    return cfg


def load_pipeline_config(path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    cfg = load_yaml_file(path)
    validate_pipeline_config(cfg)
    return cfg


def load_simulation_config(path: str = "config/simulation.yaml") -> Dict[str, Any]:
    return load_yaml_file(path)
