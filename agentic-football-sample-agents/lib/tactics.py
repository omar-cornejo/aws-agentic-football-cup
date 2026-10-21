"""Inline tactical computations — the Gateway tools' math without the round trips."""

import math

from state import _player_idx, _is_my_team, get_goal_positions, resolve_holder

GOAL_HALF_WIDTH = 5.0
_TOP_AIMS = frozenset({"TL", "TR"})


def shot_power(d_goal: float, aim: str) -> float:
    """Power rises with distance so the ball reaches the frame (gateway formula)."""
    base = min(1.0, 0.6 + d_goal / 80.0)
    if aim in _TOP_AIMS:
        if d_goal > 35:
            base *= 0.85
        elif d_goal > 20:
            base *= 0.92
    return round(max(0.55, min(1.0, base)), 2)


def _dist(a, b) -> float:
    return math.sqrt((a.get("x", 0) - b.get("x", 0)) ** 2 + (a.get("y", 0) - b.get("y", 0)) ** 2)


def _lane_perp(me_pos, opp_goal_x, opponents, aim_y):
    """Minimum perpendicular distance from any blocking opponent to the shot
    line me_pos -> (opp_goal_x, aim_y). Higher = clearer lane.
    """
    dx_goal = opp_goal_x - me_pos.get("x", 0)
    dy_goal = aim_y - me_pos.get("y", 0)
    length = math.hypot(dx_goal, dy_goal) or 1.0
    best = math.inf
    for o in opponents:
        p = o.get("position", {}) or {}
        px = p.get("x", 0) - me_pos.get("x", 0)
        py = p.get("y", 0) - me_pos.get("y", 0)
        t = (px * dx_goal + py * dy_goal) / (length ** 2)
        if not 0.05 < t < 0.98:
            continue
        perp = abs((py * dx_goal - px * dy_goal) / length)
        if perp < best:
            best = perp
    return best


def _geometry_aim_order(me_pos: dict, opp_goal_x: float) -> list[tuple[str, float]]:
    """Order goal corners by where the shooter stands relative to the frame."""
    me_y = me_pos.get("y", 0) or 0
    d_goal = _dist(me_pos, {"x": opp_goal_x, "y": 0})
    if abs(me_y) < 2.5:
        return [("CENTER", 0.0), ("TL", -4.0), ("TR", 4.0), ("BL", -4.0), ("BR", 4.0)]
    if me_y < 0:
        order = [("TR", 4.0), ("BR", 4.0), ("CENTER", 0.0), ("TL", -4.0), ("BL", -4.0)]
    else:
        order = [("TL", -4.0), ("BL", -4.0), ("CENTER", 0.0), ("TR", 4.0), ("BR", 4.0)]
    if d_goal <= 18 and abs(me_y) >= 8:
        if me_y < 0:
            order = [("BR", 4.0), ("TR", 4.0)] + [a for a in order if a[0] not in ("BR", "TR")]
        else:
            order = [("BL", -4.0), ("TL", -4.0)] + [a for a in order if a[0] not in ("BL", "TL")]
    if d_goal > 28:
        low = [a for a in order if a[0] in ("BL", "BR", "CENTER")]
        high = [a for a in order if a[0] in _TOP_AIMS]
        rest = [a for a in order if a[0] not in ("BL", "BR", "CENTER", "TL", "TR")]
        order = low + high + rest
    return order


def _aim_geometry_line(me_pos: dict, opp_goal_x: float, aim: str) -> str:
    return ("- Shot plan: CENTER lane clear -> SHOOT CENTER power 1.0; "
            "if blocked -> sprint MOVE_TO to open space (wide OR back), "
            "then SHOOT CENTER")


def _best_shot_aim(me_pos, opp_goal_x, opponents) -> tuple[str, float, float]:
    """Always CENTER — aim at the middle of the goal frame (most accurate).
    Returns center-lane clarity as the third value for block/cutback checks."""
    center_perp = _lane_perp(me_pos, opp_goal_x, opponents, 0.0)
    return "CENTER", 0.0, center_perp


def _lane_blocked(me_pos, opp_goal_x, opponents, lane_radius=None) -> tuple[bool, float]:
    """Legacy helper kept for tests: is the CENTER shot line blocked, and by
    how much should we sidestep to clear it? Adaptive lane radius — close-range
    hard shots slip past slight overlaps."""
    dx_goal = opp_goal_x - me_pos.get("x", 0)
    if abs(dx_goal) < 0.1:
        return False, 0.0
    if lane_radius is None:
        d = math.hypot(dx_goal, -me_pos.get("y", 0))
        lane_radius = 1.5 if d <= 25 else 2.5

    center_perp = _lane_perp(me_pos, opp_goal_x, opponents, 0.0)
    if center_perp >= lane_radius:
        return False, 0.0

    for off in (3, -3, 6, -6, 9, -9, 12, -12):
        if _lane_perp(me_pos, opp_goal_x, opponents, me_pos.get("y", 0) + off) >= lane_radius:
            return True, float(off)
    return True, 0.0


def _shot_line(me_pos, opp_gk_pos, opponents, opp_goal_x,
               position_label=None) -> tuple[str, bool, float]:
    """Shot probability + lane check. Also returns (blocked, y_offset) so the
    caller can add a follow-up TACTICS line telling the agent to shift laterally
    before shooting (rather than blindly firing into a defender's shins).
    """
    goal = {"x": opp_goal_x, "y": 0}
    d_goal = _dist(me_pos, goal)

    if position_label == "GK":
        nearest = min((_dist(me_pos, o.get("position", {}) or {}) for o in opponents),
                      default=math.inf)
        if nearest <= 14:
            aim, _y, _perp = _best_shot_aim(me_pos, opp_goal_x, opponents)
            pwr = shot_power(d_goal, aim)
            return (f"- Shot: BLAST (dist {d_goal:.0f}, PRESSURE) -> SHOOT NOW aim "
                    f"{aim} power {pwr} — clearance under pressure", False, 0.0)
        return (f"- Distribute: no pressure (nearest opp {nearest:.0f}m) -> "
                f"GK_DISTRIBUTE KICK to most advanced MID/FWD", False, 0.0)
    if position_label == "DEF":
        aim, _y, _perp = _best_shot_aim(me_pos, opp_goal_x, opponents)
        pwr = shot_power(d_goal, aim)
        return (f"- Shot: BLAST (dist {d_goal:.0f}) -> SHOOT NOW aim {aim} "
                f"power {pwr} — no range limit for you: worst case it's a "
                f"clearance, best case it's a goal", False, 0.0)

    angle_factor = min(1.0, math.atan2(GOAL_HALF_WIDTH, max(d_goal, 0.1)) / 0.15)
    distance_factor = max(0.0, 1.0 - d_goal / 55.0)
    if opp_gk_pos is not None:
        gk_factor = min(1.0, abs(opp_gk_pos.get("y", 0)) / 8.0) * 0.3
        gk_dist_factor = min(1.0, _dist(opp_gk_pos, goal) / 15.0) * 0.2
    else:
        gk_factor, gk_dist_factor = 0.3, 0.2

    blocker_penalty = 0.0
    for o in opponents:
        d = _dist(me_pos, o.get("position", {}))
        if d < 10:
            blocker_penalty += (10 - d) / 10.0 * 0.15
    blocker_penalty = min(blocker_penalty, 0.4)

    p = max(0.02, min(0.95, distance_factor * 0.45 + angle_factor * 0.25
                      + gk_factor + gk_dist_factor - blocker_penalty))

    lane_radius = 1.5 if d_goal <= 25 else 2.5
    aim, _aim_y, aim_perp = _best_shot_aim(me_pos, opp_goal_x, opponents)
    lane_clear = aim_perp >= lane_radius

    pwr = shot_power(d_goal, aim)
    if d_goal > 45:
        if d_goal <= 58:
            if lane_clear:
                verdict = (f"LANE CLEAR — LONG SHOT: aim CENTER power {pwr} "
                           f"(code snap-shots; no dribbling for range)")
            else:
                verdict = ("LANE BLOCKED — LONG SHOT: SHOOT CENTER power "
                           f"{pwr} (code snap-shots; no dribbling for range)")
                blocked = False
                return (f"- Shot: {round(p * 100)}% (dist {d_goal:.0f} to goal) -> {verdict}",
                        blocked, 0.0)
        else:
            verdict = "out of range: sprint carry to open wing, then shoot CENTER"
        blocked = False
    elif lane_clear:
        verdict = f"LANE CLEAR — SHOOT NOW: aim CENTER power {pwr}"
        blocked = False
    elif d_goal <= 15:
        verdict = f"POINT-BLANK — SHOOT NOW: aim CENTER power {shot_power(d_goal, 'CENTER')}"
        blocked = False
    else:
        verdict = (f"LANE BLOCKED — SHOOT NOW: aim CENTER power "
                   f"{shot_power(d_goal, 'CENTER')} (drive through traffic)")
        blocked = False
    return (f"- Shot: {round(p * 100)}% (dist {d_goal:.0f} to goal) -> {verdict}",
            blocked, 0.0)


def _pass_line(me_pos, my_id, teammates, opponents) -> str:
    """Port of gateway_tools/calculate_pass_options.py — top 2 options."""
    opp_positions = [o.get("position", {}) for o in opponents]
    options = []
    for tm in teammates:
        tid = _player_idx(tm)
        if tid == my_id:
            continue
        tm_pos = tm.get("position", {})
        pass_dist = _dist(me_pos, tm_pos)

        risk = 0.0
        if pass_dist >= 1:
            dx = tm_pos.get("x", 0) - me_pos.get("x", 0)
            dy = tm_pos.get("y", 0) - me_pos.get("y", 0)
            for op in opp_positions:
                ox = op.get("x", 0) - me_pos.get("x", 0)
                oy = op.get("y", 0) - me_pos.get("y", 0)
                t = max(0, min(1, (ox * dx + oy * dy) / (pass_dist ** 2)))
                lane = math.sqrt((op.get("x", 0) - (me_pos.get("x", 0) + t * dx)) ** 2
                                 + (op.get("y", 0) - (me_pos.get("y", 0) + t * dy)) ** 2)
                if lane < 8:
                    risk = max(risk, 1.0 - lane / 8.0)
            risk = min(risk, 0.95)

        success = max(0.05, 1.0 - risk - pass_dist / 120.0)
        ptype = "GROUND" if pass_dist < 20 else ("THROUGH" if success > 0.5 else "AERIAL")
        options.append((success, tid, ptype))

    if not options:
        return ""
    options.sort(reverse=True)
    best = ", ".join(f"P{tid} {round(s * 100)}% {t}" for s, tid, t in options[:2])
    return f"- Best passes: {best}"


def _mark_line(me_pos, opponents, ball, my_goal_x, opp_holder_idx) -> str:
    """Port of gateway_tools/get_defensive_assignment.py — top threat only."""
    my_goal = {"x": my_goal_x, "y": 0}
    ball_pos = ball.get("position", {}) or {}

    best = None
    for o in opponents:
        pos = o.get("position", {})
        oid = _player_idx(o)
        threat = max(0.0, 1.0 - _dist(pos, my_goal) / 80.0)
        d_ball = _dist(pos, ball_pos)
        threat += 0.3 if d_ball < 10 else (0.15 if d_ball < 20 else 0.0)
        has_ball = opp_holder_idx is not None and oid == opp_holder_idx
        if has_ball:
            threat += 0.4
        if best is None or threat > best[0]:
            best = (threat, oid, has_ball)

    if best is None:
        return ""
    threat, oid, has_ball = best
    tightness = "TIGHT" if threat > 0.7 else "LOOSE"
    ball_note = ", has ball" if has_ball else ""
    return f"- Top threat: P{oid} (score {threat:.2f}{ball_note}) -> MARK {tightness} or intercept"


def _space_line(me_pos, opponents, team_id) -> str:
    """Port of gateway_tools/find_open_space.py — attack zone, 5-unit grid."""
    opp_positions = [o.get("position", {}) for o in opponents]
    x_min, x_max = (15, 52) if team_id == 0 else (-52, -15)

    best_point, best_score, best_min = None, -1e9, 0.0
    for x in range(x_min, x_max + 1, 5):
        for y in range(-30, 31, 5):
            pt = {"x": x, "y": y}
            min_opp = min((_dist(pt, op) for op in opp_positions), default=999.0)
            score = min_opp - _dist(pt, me_pos) * 0.15
            if score > best_score:
                best_score, best_point, best_min = score, pt, min_opp

    if best_point is None:
        return ""
    return (f"- Open space: ({best_point['x']},{best_point['y']}) "
            f"nearest opp {best_min:.0f} away -> MOVE_TO there if not attacking the ball")


def tactics_report(game_state: dict, team_id: int, my_player_id: int, position_label: str) -> str:
    """Build a <=3-line TACTICS block. Empty string when nothing useful."""
    ball = game_state.get("ball", {}) or {}
    players = game_state.get("players", []) or []

    my_team = [p for p in players if _is_my_team(p, team_id)]
    opponents = [p for p in players if not _is_my_team(p, team_id)]
    me = next((p for p in my_team if _player_idx(p) == my_player_id), None)
    if me is None:
        return ""
    me_pos = me.get("position", {}) or {}

    my_goal_x, opp_goal_x = get_goal_positions(team_id)

    holder = resolve_holder(ball, players)
    i_have_ball = holder is me
    opp_holder_idx = _player_idx(holder) if (holder is not None and not _is_my_team(holder, team_id)) else None

    opp_gk = next((o for o in opponents if _player_idx(o) == 0), None)
    opp_gk_pos = (opp_gk or {}).get("position")

    lines = []
    if i_have_ball:
        aim, _, _ = _best_shot_aim(me_pos, opp_goal_x, opponents)
        lines.append(_aim_geometry_line(me_pos, opp_goal_x, aim))
        shot, _blocked, _y_off = _shot_line(me_pos, opp_gk_pos, opponents,
                                            opp_goal_x, position_label)
        lines.append(shot)
        pass_line = _pass_line(me_pos, my_player_id, my_team, opponents)
        if pass_line:
            lines.append(pass_line)
    elif position_label in ("GK", "DEF"):
        mark = _mark_line(me_pos, opponents, ball, my_goal_x, opp_holder_idx)
        if mark:
            lines.append(mark)
    elif opp_holder_idx is not None:
        mark = _mark_line(me_pos, opponents, ball, my_goal_x, opp_holder_idx)
        if mark:
            lines.append(mark)
    else:
        space = _space_line(me_pos, opponents, team_id)
        if space:
            lines.append(space)

    if not lines:
        return ""
    return "TACTICS (computed):\n" + "\n".join(lines)
