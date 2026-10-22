"""Stamina guard applied to every outgoing command (code-first, LLM, fallback)."""
from __future__ import annotations

import math

import tuning as tuning_mod
import units
from state import _is_my_team, _player_idx, resolve_holder


def stamina_pct(player: dict | None) -> float | None:
    if not player:
        return None
    s = player.get("stamina")
    if not isinstance(s, (int, float)) or isinstance(s, bool):
        return None
    return float(s) * 100.0 if 0.0 <= s <= 1.0 else float(s)


def _me(game_state: dict, team_id: int, pid: int):
    return next((p for p in game_state.get("players") or []
                 if _player_idx(p) == pid and _is_my_team(p, team_id)), None)


def _closest_to_ball(game_state: dict, team_id: int, pid: int) -> bool:
    ball = (game_state.get("ball") or {}).get("position") or {}
    bx, by = ball.get("x", 0) or 0, ball.get("y", 0) or 0
    mates = [p for p in game_state.get("players") or [] if _is_my_team(p, team_id)]
    if not mates:
        return False

    def d(p):
        q = p.get("position") or {}
        return math.hypot((q.get("x", 0) or 0) - bx, (q.get("y", 0) or 0) - by)

    return _player_idx(min(mates, key=d)) == pid


def _xy(p) -> tuple[float, float]:
    q = (p or {}).get("position") or {}
    return float(q.get("x", 0) or 0), float(q.get("y", 0) or 0)


def _cheap_sprints(commands: list, game_state: dict, team_id: int, pid: int, me) -> str | None:
    """Sprint rules that apply at ANY stamina: GK only near danger, no short sprints."""
    tag = None
    mx, my = _xy(me)
    ball = (game_state.get("ball") or {}).get("position") or {}
    bx, by = float(ball.get("x", 0) or 0), float(ball.get("y", 0) or 0)
    own_goal_x = -55.0 if team_id == 0 else 55.0
    min_m = float(tuning_mod.get("sprint_min_m", 6.0))
    for cmd in commands:
        if cmd.get("commandType") != "MOVE_TO":
            continue
        params = cmd.get("parameters") or {}
        if not params.get("sprint"):
            continue
        tx, ty = params.get("target_x"), params.get("target_y")
        if pid == 0 and math.hypot(bx - own_goal_x, by) > float(tuning_mod.get("gk_sprint_ball_m", 30.0)):
            params["sprint"] = False
            tag = "gk-walk"
        elif (isinstance(tx, (int, float)) and isinstance(ty, (int, float))
              and math.hypot(tx - mx, ty - my) < min_m):
            params["sprint"] = False
            tag = tag or "short"
    return tag


def guard(commands: list, game_state: dict, team_id: int, pid: int) -> tuple[list, str | None]:
    """Return (commands, tag) — 'gk-walk' / 'short' / 'stam' when a sprint was toned down."""
    if not commands:
        return commands, None
    game_state, _ = units.to_field(game_state, force=True)
    me = _me(game_state, team_id, pid)
    holder = resolve_holder(game_state.get("ball") or {}, game_state.get("players") or [])
    if holder is not None and holder is me:
        return commands, None
    cheap = _cheap_sprints(commands, game_state, team_id, pid, me) if me else None
    st = stamina_pct(me)
    low = float(tuning_mod.get("stamina_low", 50.0))
    if st is None or st >= low or _closest_to_ball(game_state, team_id, pid):
        return commands, cheap
    tag = cheap
    for cmd in commands:
        params = cmd.get("parameters") or {}
        ctype = cmd.get("commandType")
        if ctype == "MOVE_TO" and params.get("sprint"):
            params["sprint"] = False
            tag = "stam"
        elif ctype == "PRESS_BALL":
            inten = params.get("intensity")
            if not isinstance(inten, (int, float)) or inten > 0.5:
                params["intensity"] = 0.5
                tag = "stam"
        elif ctype == "SLIDE_TACKLE" and params.get("sprint"):
            params["sprint"] = False
            tag = "stam"
        cmd["parameters"] = params
    return commands, tag
