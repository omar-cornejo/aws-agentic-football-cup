"""Shared 2-tick attack forecast, used by overrides and the LLM prompt."""

from __future__ import annotations

import math


def get_goal_positions(team_id: int) -> tuple[float, float]:
    if team_id == 0:
        return -55.0, 55.0
    return 55.0, -55.0


def dist(pos1: dict, pos2: dict) -> float:
    return math.sqrt(
        (pos1.get("x", 0) - pos2.get("x", 0)) ** 2
        + (pos1.get("y", 0) - pos2.get("y", 0)) ** 2
    )


def dir_my_for(team_id: int) -> float:
    my_goal_x, _ = get_goal_positions(team_id)
    return 1.0 if my_goal_x > 0 else -1.0


def carry_lead_m(ticks: int, per_tick: float, ctype: str | None, params: dict,
                 *, toward_goal: bool = True) -> float:
    """Metres of goal-ward movement to subtract from dist-to-goal."""
    base = per_tick * ticks
    if ctype == "MOVE_TO" and params and toward_goal:
        tx = params.get("target_x")
        x = params.get("_x")
        if isinstance(tx, (int, float)) and isinstance(x, (int, float)):
            dir_my = params.get("_dir_my", -1.0)
            if (tx - x) * -dir_my > 0:
                return base if params.get("sprint") else base * 0.5
    if toward_goal:
        return base
    return base * 0.35


def projected_toward_goal(me_pos: dict, team_id: int, ticks: int,
                          per_tick: float) -> dict:
    """Pre-LLM forecast: carrier keeps sprinting at the opponent goal."""
    dir_my = dir_my_for(team_id)
    x, y = me_pos.get("x", 0) or 0, me_pos.get("y", 0) or 0
    step = per_tick * ticks
    return {"x": x + step * -dir_my, "y": y}


def effective_d_goal(me_pos: dict, opp_goal_x: float, lead_m: float) -> float:
    return max(0.0, dist(me_pos, {"x": opp_goal_x, "y": 0}) - lead_m)


def in_opponent_half(me_pos: dict, team_id: int, line: float = 5.0) -> bool:
    return (me_pos.get("x", 0) or 0) * -dir_my_for(team_id) > line


def in_opponent_half_soon(me_pos: dict, team_id: int, ticks: int, per_tick: float,
                          line: float = 5.0) -> bool:
    if in_opponent_half(me_pos, team_id, line):
        return True
    future = projected_toward_goal(me_pos, team_id, ticks, per_tick)
    return in_opponent_half(future, team_id, line)


def forecast_prompt_lines(me_pos: dict, team_id: int, position_label: str,
                          has_ball: bool, *, ticks: int = 2,
                          per_tick: float = 6.0, longshot_max: float = 58.0,
                          opp_half_line: float = 5.0, aim: str = "CENTER") -> list[str]:
    """Lines injected into the LLM state summary before the model decides."""
    if not has_ball or position_label not in ("MID", "FWD1", "FWD2"):
        return []
    _, opp_goal_x = get_goal_positions(team_id)
    d_now = dist(me_pos, {"x": opp_goal_x, "y": 0})
    lead = carry_lead_m(ticks, per_tick, None, {}, toward_goal=True)
    d_eff = effective_d_goal(me_pos, opp_goal_x, lead)
    future = projected_toward_goal(me_pos, team_id, ticks, per_tick)
    lines = [
        f"FORECAST (+{ticks} ticks): projected pos=({future['x']:.1f},{future['y']:.1f}) "
        f"distOppGoal now={d_now:.0f} effective={d_eff:.0f}",
    ]
    if d_eff <= longshot_max:
        from tactics import shot_power
        pwr = shot_power(d_eff, aim)
        lines.append(
            f"FORECAST ACTION: in range from anywhere on the pitch — reply SHOOT aim {aim} "
            f"power {pwr} THIS tick (code snap-shot uses geometry aim if you dribble/pass)"
        )
    elif d_now <= longshot_max + per_tick * ticks:
        lines.append(
            f"FORECAST ACTION: entering range — sprint toward goal or SHOOT when "
            f"distOppGoal effective<={longshot_max:.0f}"
        )
    return lines


def lead_for_override(cfg, me_pos: dict, opp_goal_x: float, dir_my: float,
                      ctype: str | None, params: dict) -> float:
    """Post-LLM: use the model's MOVE_TO target when available."""
    p = dict(params or {})
    p["_x"] = me_pos.get("x", 0)
    p["_dir_my"] = dir_my
    return carry_lead_m(cfg.shoot_lead_ticks, cfg.shoot_lead_per_tick,
                        ctype, p, toward_goal=True)


def projected_x_override(me_pos: dict, dir_my: float, ctype: str | None,
                         params: dict, cfg) -> float:
    x = me_pos.get("x", 0) or 0
    if ctype != "MOVE_TO" or not params:
        return x
    tx = params.get("target_x")
    if not isinstance(tx, (int, float)):
        return x
    if (tx - x) * -dir_my <= 0:
        return x
    return x + lead_for_override(cfg, me_pos, 0, dir_my, ctype, params) * -dir_my
