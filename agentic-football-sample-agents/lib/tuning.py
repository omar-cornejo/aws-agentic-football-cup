"""Deploy-time tunable tactical parameters."""

from __future__ import annotations

import json
import os
from dataclasses import fields, replace

_PATH = os.path.join(os.path.dirname(__file__), "tuning.json")

BOUNDS: dict[str, tuple[float, float]] = {
    "shoot_threshold": (35.0, 50.0),
    "longshot_max": (45.0, 58.0),
    "gk_out_dist": (8.0, 18.0),
    "blast_pressure_dist": (6.0, 16.0),
    "outlet_min_gain": (6.0, 20.0),
    "outlet_lane_radius": (3.0, 8.0),
    "press_bodies": (2, 4),
    "chase_radius": (5.0, 12.0),
    "anchor_slack": (8.0, 18.0),
    "mark_radius": (18.0, 32.0),
    "press_radius": (12.0, 28.0),
    "compact_max_spread": (14.0, 28.0),
    "anchor_slack": (6.0, 14.0),
    "wing_y": (-18.0, 18.0),
    "tackle_dist": (2.0, 6.0),
    "gk_crowd_dist": (8.0, 28.0),
    "intercept_radius": (6.0, 15.0),
    "carry_cap_x": (38.0, 50.0),
    "shoot_lead_per_tick": (4.0, 8.0),
    "shoot_lead_ticks": (1, 3),
    "opp_half_line": (0.0, 12.0),
    "llm_deadline_ms": (400, 4500),
    "press_count": (1, 3),
    "field_y_half": (25.0, 40.0),
    "memory_window_ticks": (1, 15),
    "line_push_DEF": (-10.0, 25.0),
    "line_push_MID": (-10.0, 25.0),
    "ev_shot_bias": (0.3, 3.0),
    "stamina_low": (0.0, 100.0),
    "gk_sprint_ball_m": (10.0, 60.0),
    "sprint_min_m": (0.0, 15.0),
    "mem_long_m": (30.0, 70.0),
    "kickoff_no_long_s": (0.0, 30.0),
    "mem_window_s": (2.0, 10.0),
    "mem_miss_after": (1, 10),
    "mem_miss_mult": (0.0, 1.0),
    "lead_long_shot_mult": (0.0, 1.5),
    "ev_support_shift": (-12.0, 12.0),
    "mem_prior_long_mult": (0.3, 1.0),
    "shoot_early_m": (16.5, 40.0),
    "coach_shoot_max_m": (20.0, 110.0),
    "def_floor_m": (-50.0, -5.0),
    "mid_max_y": (2.0, 30.0),
    "mid_pass_gain_m": (0.0, 25.0),
    "mem3_tackle_m": (1.0, 8.0),
}

_INT_KEYS = {"press_bodies", "shoot_lead_ticks", "llm_deadline_ms", "press_count", "mem_miss_after", "memory_window_ticks"}


def _load(path: str = _PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_DATA = _load()


def clamp(key: str, value):
    """Clamp numeric values into BOUNDS; cast int keys; pass others through."""
    if key in BOUNDS and isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = BOUNDS[key]
        value = max(lo, min(hi, value))
        return int(round(value)) if key in _INT_KEYS else float(value)
    return value


def apply_tuning(cfg, position_label: str, data: dict | None = None):
    """Overlay tuning (global, then position) onto an OverrideConfig copy."""
    if cfg is None:
        return None
    data = _DATA if data is None else data
    merged = {}
    merged.update(data.get("global") or {})
    merged.update(data.get(position_label) or {})
    if not merged:
        return cfg
    valid = {f.name for f in fields(cfg)}
    kwargs = {k: clamp(k, v) for k, v in merged.items() if k in valid}
    return replace(cfg, **kwargs) if kwargs else cfg


def get(key: str, default, data: dict | None = None):
    """Read a single global tuning value (used by state.py thresholds)."""
    data = _DATA if data is None else data
    value = (data.get("global") or {}).get(key, default)
    return clamp(key, value)
