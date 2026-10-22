"""Convert the live game state from engine units to the logical pitch (metres)."""
from __future__ import annotations

import copy

import tuning as tuning_mod

ENGINE_X_HALF = 7.0
ENGINE_Y_HALF = 3.5
FIELD_X_HALF = 55.0


def _xs(game_state: dict):
    ball = (game_state.get("ball") or {}).get("position") or {}
    yield ball.get("x"), ball.get("y")
    for p in game_state.get("players") or []:
        q = p.get("position") or {}
        yield q.get("x"), q.get("y")


def is_engine_units(game_state: dict) -> bool:
    """True when every PLAYER fits the compact engine pitch (|x|<=8.4, |y|<=4.55)."""
    seen = False
    pts = [((p.get("position") or {}).get("x"), (p.get("position") or {}).get("y"))
           for p in game_state.get("players") or []]
    for x, y in pts or _xs(game_state):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        seen = True
        if abs(x) > ENGINE_X_HALF * 1.2 or abs(y) > ENGINE_Y_HALF * 1.3:
            return False
    return seen


def _scale(v: dict | None, sx: float, sy: float) -> None:
    if not isinstance(v, dict):
        return
    if isinstance(v.get("x"), (int, float)):
        v["x"] = v["x"] * sx
    if isinstance(v.get("y"), (int, float)):
        v["y"] = v["y"] * sy


def to_field(game_state: dict, force: bool = False) -> tuple[dict, bool]:
    """(state in metres, converted?) — returns a scaled deep copy when needed."""
    if not force and not tuning_mod.get("units_convert", False):
        return game_state, False
    if not game_state or not is_engine_units(game_state):
        return game_state, False
    sx = FIELD_X_HALF / ENGINE_X_HALF
    sy = float(tuning_mod.get("field_y_half", 30.0)) / ENGINE_Y_HALF
    gs = copy.deepcopy(game_state)
    ball = gs.get("ball") or {}
    _scale(ball.get("position"), sx, sy)
    _scale(ball.get("velocity"), sx, sy)
    for p in gs.get("players") or []:
        _scale(p.get("position"), sx, sy)
        _scale(p.get("velocity"), sx, sy)
    return gs, True
