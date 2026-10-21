"""Post-LLM tactical overrides — prompts suggest, code enforces."""

from __future__ import annotations
from dataclasses import dataclass

from state import (get_goal_positions, dist, _player_idx, _is_my_team,
                   resolve_holder, count_opponents_in_our_half)
import math

from tactics import _best_shot_aim, _lane_blocked, _lane_perp, shot_power
from forecast import effective_d_goal as _effective_d_goal
from forecast import lead_for_override, projected_x_override

_CHASE_CMDS = ("PRESS_BALL", "INTERCEPT", "SLIDE_TACKLE")
_NEEDS_BALL_CMDS = ("PASS", "SHOOT", "GK_DISTRIBUTE")
_SET_PIECE_MODES = {"GOALKICK", "FREEKICK", "KICKOFF", "CORNER", "CORNERKICK",
                    "THROWIN", "PENALTY"}


def _is_set_piece(play_mode) -> bool:
    pm = str(play_mode or "").upper().replace("_", "").replace(" ", "")
    return pm in _SET_PIECE_MODES


@dataclass
class OverrideConfig:
    enforce_shot: bool = True
    shoot_threshold: float = 45.0
    always_blast: bool = False
    """GK/DEF: ANY open-play possession -> full-power shot at the clearest aim.
    Doubles as a clearance; no distance limit by design."""
    build_from_back: bool = False
    """Softens always_blast (iter-11, from match data): possession with a
    clear advancing lane becomes an outlet pass that feeds the attack; the
    blast remains the no-outlet fallback."""
    blast_pressure_dist: float = 10.0
    """An opponent within this of the carrier counts as pressure: the outlet
    then needs a 1.5x wider clear lane before it beats the blast."""
    counter_attack: bool = True
    longshot_max: float = 52.0
    gk_out_dist: float = 12.0
    """Opponent GK this far off his goal center counts as 'off his line'."""
    outlet_min_gain: float = 12.0
    """Outlet must be at least this much closer to the opponent goal than me."""
    outlet_lane_radius: float = 5.0
    press_bodies: int = 3
    """Opponents in our half that constitute a HIGH PRESS (counter trigger)."""
    no_chase: bool = True
    chase_radius: float = 8.0
    """MOVE_TO with a target this close to the ball counts as chasing it."""
    compact_anchor: bool = True
    anchor_slack: float = 8.0
    """Defensive MOVE_TO targets further than this from the anchor get clamped."""
    mark_radius: float = 22.0
    """MARK an opponent within this range; else hold the compact line."""
    press_radius: float = 18.0
    """Non-designated players this close to the carrier shadow-press or MARK."""
    compact_max_spread: float = 20.0
    """Pull the anchor inward when farther than this from the team centroid."""
    wing_y: float = 0.0
    """Forwards only: carrying the ball out of range down the center (|y|<8)
    is steered to this wing lane. 0 disables."""
    tackle_dist: float = 2.5
    """Designated presser this close to the carrier slides instead of
    shadowing forever."""
    gk_crowd_dist: float = 14.0
    """GK with the ball: opponent this close -> blast, never distribute."""
    intercept_radius: float = 10.0
    """Designated player (and the GK in its box) INTERCEPTs a loose ball
    inside this radius instead of jogging at it."""
    carry_cap_x: float = 44.0
    """Dribble targets clamp to |x| <= this: past the box edge the shot angle
    dies at the byline (stale-state overruns, user-observed)."""
    shoot_lead_per_tick: float = 6.0
    """Estimated metres toward goal per tick when sprinting."""
    shoot_lead_ticks: int = 2
    """How many ticks ahead to project (state + execution lag)."""
    opp_half_line: float = 5.0
    """Past this |x| from center counts as opponent half for snap-shots."""
    ev_select: bool = False
    """Ball carrier (DEF/MID/FWD) picks shoot / pass / carry by expected value
    from lib/ev_model.json (Markov xT + logistic success models)."""
    def_search: bool = False
    """Opponent ball: DEF/MID/FWD tasks (press / mark / cover) chosen by
    IDS + backtracking search over joint assignments (lib/def_search)."""
    gk_ev: bool = False
    """GK: distribute to the best-EV teammate; angle positioning without ball."""
    ev_support: bool = False
    """Teammate has the ball: off-ball DEF/MID/FWD run to the open zone with
    the highest Markov threat inside their role band (fixes anchored lines)."""
    ev_model: str = ""
    """Model file in lib/ for the EV / search features ("" = ev_model.json)."""
    ev_support_roles: str = "DEF,MID,FWD1,FWD2"
    """Which positions use the Markov off-ball runs (comma separated)."""
    ev_support_shift: float = 0.0
    """Move every role band forward (+) or back (-) by this many metres."""
    press_count: int = 1
    """How many outfield players (nearest the ball) press/intercept together.
    1 = original iter-12 behaviour."""
    lane_y: float = 0.0
    """MID only: lateral lane (+ = one side, - = the other) so two midfielders
    in a 1-2-1 do not stand on the same spot."""


def _cmd(cmd_type: str, pid: int, tid: int, params: dict, duration: int = 0) -> dict:
    return {"commandType": cmd_type, "playerId": pid, "teamId": tid,
            "parameters": params, "duration": duration}


def _between(a: float, b: float, v: float) -> float:
    lo, hi = (a, b) if a <= b else (b, a)
    return max(lo, min(hi, v))


def _push(position_label: str) -> float:
    """Metres the role's defensive band moves up the pitch (tuning line_push_<POS>)."""
    import tuning as _t
    return float(_t.get(f"line_push_{position_label}", 0.0) or 0.0)


def _anchor(position_label: str, team_id: int, ball_pos: dict,
            deep: bool = False, lane_y: float = 0.0) -> tuple[float, float]:
    """Role anchor on a compact, ball-shifted line measured from the goal line.
    Old hi=6*dir_my let DEF camp at x≈-6 (match forensics: empty back line)."""
    my_goal_x, _ = get_goal_positions(team_id)
    dir_my = 1.0 if my_goal_x > 0 else -1.0
    bx, by = ball_pos.get("x", 0), ball_pos.get("y", 0)

    def _from_goal(metres: float) -> float:
        return my_goal_x - metres * dir_my

    push = _push(position_label)
    if position_label == "DEF":
        deep_m, shallow_m = (10, 22) if deep else (12, 28)
        deep_m, shallow_m = deep_m + push, shallow_m + push
        shift_m = 10 if deep else 14
        ax = _between(_from_goal(deep_m), _from_goal(shallow_m), bx + shift_m * dir_my)
        ay = _between(-10.0, 10.0, by * 0.55)
    elif position_label == "MID":
        deep_m, shallow_m = (18, 32) if deep else (22, 38)
        deep_m, shallow_m = deep_m + push, shallow_m + push
        shift_m = 8 if deep else 10
        ax = _between(_from_goal(deep_m), _from_goal(shallow_m), bx + shift_m * dir_my)
        lim = 12.0 + abs(lane_y)
        ay = _between(-lim, lim, by * 0.55 + lane_y)
    else:
        wing = -8.0 if position_label == "FWD1" else 8.0
        deep_m, shallow_m = (35, 48) if deep else (28, 40)
        ax = _between(_from_goal(deep_m), _from_goal(shallow_m), bx - 8 * dir_my)
        return ax, wing
    return ax, ay


def _max_forward_x(position_label: str, team_id: int, ball_pos: dict,
                   deep: bool = False) -> float:
    """Shallowest (most advanced) allowed x on the defensive line for this role."""
    my_goal_x, _ = get_goal_positions(team_id)
    dir_my = 1.0 if my_goal_x > 0 else -1.0
    if position_label == "DEF":
        shallow_m = 22 if deep else 28
    elif position_label == "MID":
        shallow_m = 32 if deep else 38
    else:
        shallow_m = 48 if deep else 40
    return my_goal_x - (shallow_m + _push(position_label)) * dir_my


def _target_past_line(tx: float, position_label: str, team_id: int,
                      ball_pos: dict, deep: bool, slack: float) -> bool:
    limit = _max_forward_x(position_label, team_id, ball_pos, deep)
    my_goal_x, _ = get_goal_positions(team_id)
    if my_goal_x < 0:
        return tx > limit + slack
    return tx < limit - slack


def _support_spot(position_label: str, team_id: int, ball_pos: dict,
                  lane_y: float = 0.0) -> tuple[float, float]:
    """Where to run when a teammate has the ball (attack support).
    Forwards split wide at PENALTY-SPOT depth (iter-12: 13 units off the goal
    line instead of 7 — camping the byline left them behind the play and fed
    the user-observed byline pile-ups); stretching the keeper while staying
    attackable is what the cutback + enforce-shot rules need."""
    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    dir_my = 1.0 if my_goal_x > 0 else -1.0
    if position_label == "FWD1":
        return opp_goal_x + 13 * dir_my, -9.0
    if position_label == "FWD2":
        return opp_goal_x + 13 * dir_my, 9.0
    if position_label == "MID":
        lim = 10.0 + abs(lane_y)
        return opp_goal_x * 0.6, _between(-lim, lim, ball_pos.get("y", 0) * 0.4 + lane_y)
    return -8 * dir_my, 0.0


def _pass_lane_clear(me_pos, tgt_pos, opponents, radius: float) -> bool:
    """No opponent within `radius` of the pass segment (ignoring bodies right
    next to the passer, t<0.15 — the ball leaves too fast there)."""
    dx = tgt_pos.get("x", 0) - me_pos.get("x", 0)
    dy = tgt_pos.get("y", 0) - me_pos.get("y", 0)
    l2 = dx * dx + dy * dy
    if l2 < 1:
        return True
    for o in opponents:
        p = o.get("position", {}) or {}
        rx = p.get("x", 0) - me_pos.get("x", 0)
        ry = p.get("y", 0) - me_pos.get("y", 0)
        t = (rx * dx + ry * dy) / l2
        if t < 0.15:
            continue
        t = min(1.0, t)
        cx, cy = me_pos.get("x", 0) + t * dx, me_pos.get("y", 0) + t * dy
        if dist({"x": cx, "y": cy}, p) < radius:
            return False
    return True


def _best_outlet(cfg, players, team_id, my_player_id, me_pos, opp_goal_x,
                 opponents, lane_radius: float | None = None):
    """Most advanced teammate (MID/FWDs) meaningfully closer to goal with a
    clear pass lane. Returns player idx or None."""
    goal = {"x": opp_goal_x, "y": 0}
    my_d = dist(me_pos, goal)
    if lane_radius is None:
        lane_radius = cfg.outlet_lane_radius
    best_idx, best_d = None, None
    for p in players:
        idx = _player_idx(p)
        if not _is_my_team(p, team_id) or idx == my_player_id or idx not in (2, 3, 4):
            continue
        pos = p.get("position", {}) or {}
        d = dist(pos, goal)
        if my_d - d < cfg.outlet_min_gain:
            continue
        if not _pass_lane_clear(me_pos, pos, opponents, lane_radius):
            continue
        if best_d is None or d < best_d:
            best_idx, best_d = idx, d
    return best_idx


def _most_advanced_mate(players, team_id, my_player_id, opp_goal_x, me_pos,
                        min_gain, dir_my: float = 0):
    """Highest MID/FWD upfield for a route-one AERIAL. Ignores lane clarity.
    Iter-13: anyone already in the opponent half counts even if gain < min_gain
    (launch fired 9x in 3h while forwards took 0 shots — the gain gate was
    too strict against Total Attack's high press)."""
    goal = {"x": opp_goal_x, "y": 0}
    my_d = dist(me_pos, goal)
    best_idx, best_d = None, None
    for p in players:
        idx = _player_idx(p)
        if not _is_my_team(p, team_id) or idx == my_player_id or idx not in (2, 3, 4):
            continue
        pos = p.get("position", {}) or {}
        d = dist(pos, goal)
        in_opp_half = pos.get("x", 0) * (-dir_my) > 5 if dir_my else False
        if not in_opp_half and my_d - d < min_gain:
            continue
        if best_d is None or d < best_d:
            best_idx, best_d = idx, d
    return best_idx


def _cutback_mate(cfg, players, team_id, my_player_id, me_pos, opp_goal_x,
                  opponents):
    """Best cutback target: a MID/FWD teammate already in shooting range,
    reachable through a clear pass lane, whose own shot lane is open — the
    wide-carrier-to-penalty-spot ball that beats dribbling into the block."""
    goal = {"x": opp_goal_x, "y": 0}
    best_idx, best_perp = None, None
    for p in players:
        idx = _player_idx(p)
        if not _is_my_team(p, team_id) or idx == my_player_id or idx not in (2, 3, 4):
            continue
        pos = p.get("position", {}) or {}
        d_goal = dist(pos, goal)
        if d_goal > cfg.shoot_threshold or dist(me_pos, pos) > 30:
            continue
        if not _pass_lane_clear(me_pos, pos, opponents, 4.0):
            continue
        _aim, _y, perp = _best_shot_aim(pos, opp_goal_x, opponents)
        if perp < (1.5 if d_goal <= 25 else 2.5):
            continue
        if best_perp is None or perp > best_perp:
            best_idx, best_perp = idx, perp
    return best_idx


def _threat_score(opp, my_goal_x: float, ball_pos: dict) -> float:
    pos = opp.get("position", {}) or {}
    threat = max(0.0, 1.0 - abs(pos.get("x", 0) - my_goal_x) / 80.0)
    d_ball = dist(pos, ball_pos)
    if d_ball < 10:
        threat += 0.35
    elif d_ball < 20:
        threat += 0.15
    return threat


def _pick_mark_target(me_pos: dict, opponents: list, holder, my_goal_x: float,
                      ball_pos: dict, mark_radius: float):
    """Best man to mark among opponents within mark_radius (excludes GK)."""
    pool = [o for o in opponents if _player_idx(o) != 0 and o is not holder]
    near = [o for o in pool
            if dist(me_pos, o.get("position", {}) or {}) <= mark_radius]
    if not near:
        return None
    return max(near, key=lambda o: _threat_score(o, my_goal_x, ball_pos))


def _team_centroid(my_team: list, my_player_id: int) -> tuple[float, float]:
    mates = [p for p in my_team
             if _player_idx(p) != my_player_id and _player_idx(p) != 0]
    if not mates:
        return 0.0, 0.0
    xs = [p.get("position", {}).get("x", 0) for p in mates]
    ys = [p.get("position", {}).get("y", 0) for p in mates]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _compact_anchor(cfg, position_label: str, team_id: int, ball_pos: dict,
                    me_pos: dict, my_team: list, my_player_id: int,
                    deep: bool = False) -> tuple[float, float]:
    """Ball-shifted role anchor. DEF/MID stay goal-tied — never pulled shallow
    by a forward centroid (match data: whole block at x≈-6 when FWDs pushed up)."""
    ax, ay = _anchor(position_label, team_id, ball_pos, deep, cfg.lane_y)
    by = ball_pos.get("y", 0) or 0
    ay = ay * 0.55 + by * 0.45
    if position_label in ("DEF", "MID"):
        return ax, ay
    cx, cy = _team_centroid(my_team, my_player_id)
    if my_team and dist(me_pos, {"x": cx, "y": cy}) > cfg.compact_max_spread:
        ax = ax * 0.65 + cx * 0.35
        ay = ay * 0.65 + cy * 0.35
    return ax, ay


def _defensive_duty(cfg, position_label, team_id, my_player_id, me_pos,
                    ball_pos, opponents, holder=None, deep=False,
                    my_team=None, designated=False) -> dict:
    """Press/tackle when close; DEF/MID hold the back line unless danger is near."""
    my_goal_x, _ = get_goal_positions(team_id)
    my_team = my_team or []
    ball_third = _ball_in_our_box(ball_pos, my_goal_x, 35.0)
    hold_line = position_label in ("DEF", "MID") and not ball_third

    if holder is not None and not designated:
        hpos = holder.get("position", {}) or {}
        d_carrier = dist(me_pos, hpos)
        if d_carrier <= cfg.tackle_dist * 2.0:
            return _cmd("SLIDE_TACKLE", my_player_id, team_id,
                        {"target_player_id": -1, "sprint": True})
        if d_carrier <= cfg.press_radius * 0.45:
            return _cmd("PRESS_BALL", my_player_id, team_id,
                        {"intensity": 1.0}, duration=3)
        if not hold_line and d_carrier <= cfg.press_radius:
            return _cmd("MARK", my_player_id, team_id,
                        {"target_player_id": _player_idx(holder),
                         "tightness": "TIGHT"}, duration=3)

    if not hold_line:
        mark_r = min(cfg.mark_radius, 14.0) if position_label == "DEF" else cfg.mark_radius
        target = _pick_mark_target(me_pos, opponents, holder, my_goal_x,
                                   ball_pos, mark_r)
        if target is not None:
            tpos = target.get("position", {}) or {}
            if dist(me_pos, tpos) <= mark_r:
                return _cmd("MARK", my_player_id, team_id,
                            {"target_player_id": _player_idx(target),
                             "tightness": "TIGHT"}, duration=3)

    ax, ay = _compact_anchor(cfg, position_label, team_id, ball_pos, me_pos,
                             my_team, my_player_id, deep)
    sprint = dist(me_pos, {"x": ax, "y": ay}) > 10
    return _cmd("MOVE_TO", my_player_id, team_id,
                {"target_x": round(ax, 1), "target_y": round(ay, 1),
                 "sprint": sprint})


def _in_opponent_half(me_pos: dict, dir_my: float, line: float) -> bool:
    return (me_pos.get("x", 0) or 0) * -dir_my > line


def _in_opponent_half_soon(me_pos: dict, dir_my: float, line: float,
                           ctype: str | None, params: dict,
                           cfg: OverrideConfig) -> bool:
    if _in_opponent_half(me_pos, dir_my, line):
        return True
    future = {**me_pos, "x": projected_x_override(me_pos, dir_my, ctype, params, cfg)}
    return _in_opponent_half(future, dir_my, line)


def _ball_in_our_box(ball_pos: dict, my_goal_x: float, depth: float = 18.0) -> bool:
    """Ball within `depth` metres of our goal line (x-axis)."""
    return abs(ball_pos.get("x", 0) - my_goal_x) <= depth


def _gk_line_target(my_goal_x: float, ball_pos: dict) -> tuple[float, float]:
    return (round(my_goal_x * 0.96, 1),
            round(_between(-6.0, 6.0, ball_pos.get("y", 0)), 1))


def _gk_nearest_opp_dist(me_pos: dict, opponents: list) -> float:
    if not opponents:
        return math.inf
    return min(dist(me_pos, o.get("position", {}) or {}) for o in opponents)


def _gk_under_pressure(cfg, me_pos: dict, opponents: list) -> bool:
    return _gk_nearest_opp_dist(me_pos, opponents) <= cfg.gk_crowd_dist


def _gk_blast_shot(commands, cmd, ctype, params, my_player_id, team_id,
                   me_pos, opp_goal_x, opponents, d_goal: float, tag: str):
    aim, _, _ = _best_shot_aim(me_pos, opp_goal_x, opponents)
    return _force_shot(commands, cmd, ctype, params, my_player_id, team_id,
                       aim, tag, d_goal)


def _gk_distribute(method: str, target: int, my_player_id: int, team_id: int,
                   tag: str):
    return [_cmd("GK_DISTRIBUTE", my_player_id, team_id,
                 {"target_player_id": target, "method": method})], tag


def _gk_possession(commands, cmd, ctype, params, my_player_id, team_id, cfg,
                   me_pos, opp_goal_x, opponents, players, d_goal: float,
                   set_piece: bool, dir_my: float):
    """Conservative GK: long distribute when safe, blast only under pressure."""
    if _gk_under_pressure(cfg, me_pos, opponents):
        return _gk_blast_shot(commands, cmd, ctype, params, my_player_id,
                              team_id, me_pos, opp_goal_x, opponents,
                              d_goal, "gk-blast")
    outlet = _best_outlet(cfg, players, team_id, my_player_id, me_pos,
                          opp_goal_x, opponents)
    if outlet is not None:
        tgt = next(p for p in players
                   if _player_idx(p) == outlet and _is_my_team(p, team_id))
        tgt_pos = tgt.get("position", {}) or {}
        method = "KICK" if dist(me_pos, tgt_pos) > 10 else "THROW"
        tag = "gk-restart" if set_piece else "gk-dist"
        return _gk_distribute(method, outlet, my_player_id, team_id, tag)
    launch = _most_advanced_mate(players, team_id, my_player_id, opp_goal_x,
                                 me_pos, cfg.outlet_min_gain, dir_my)
    if launch is not None:
        tag = "gk-restart" if set_piece else "gk-launch"
        return _gk_distribute("KICK", launch, my_player_id, team_id, tag)
    tag = "gk-restart-clear" if set_piece else "gk-clear"
    return _gk_blast_shot(commands, cmd, ctype, params, my_player_id, team_id,
                          me_pos, opp_goal_x, opponents, d_goal, tag)


def _open_carry_score(pos: dict, opp_goal_x: float, opponents: list,
                      lane_radius: float) -> float:
    """How open a carry target is: CENTER lane clarity minus nearby crowd."""
    perp = _lane_perp(pos, opp_goal_x, opponents, 0.0)
    crowd = sum(1 for o in opponents
                if dist(pos, o.get("position", {}) or {}) < 8)
    return perp - crowd * 1.5


def _best_open_carry_target(me_pos: dict, opp_goal_x: float, opponents: list,
                            cfg, lane_radius: float) -> tuple[float, float]:
    """Pick open space — lateral, forward, or backward (user: 可以往后带)."""
    mx, my = me_pos.get("x", 0), me_pos.get("y", 0)
    gx = 1.0 if opp_goal_x > mx else -1.0
    wing = cfg.wing_y or (12.0 if my >= 0 else -12.0)
    _blocked, y_off = _lane_blocked(me_pos, opp_goal_x, opponents, lane_radius)

    candidates = [
        (mx, my + y_off),
        (mx - 8 * gx, my),
        (mx - 8 * gx, my + y_off),
        (mx, wing),
        (mx - 8 * gx, wing),
        (mx + 4 * gx, my + y_off),
    ]
    if y_off == 0.0:
        candidates.extend([(mx - 12 * gx, my), (mx, my + 9), (mx, my - 9)])

    best_tx, best_ty, best_score = mx, my, -math.inf
    for tx, ty in candidates:
        score = _open_carry_score({"x": tx, "y": ty}, opp_goal_x, opponents,
                                  lane_radius)
        if score > best_score:
            best_tx, best_ty, best_score = tx, ty, score
    return round(best_tx, 1), round(best_ty, 1)


def _open_lane_carry(cfg, me_pos: dict, opp_goal_x: float, opponents: list,
                     my_player_id: int, team_id: int, lane_radius: float):
    """Sprint to open space (any direction), then blast CENTER next tick."""
    tx, ty = _best_open_carry_target(me_pos, opp_goal_x, opponents, cfg,
                                     lane_radius)
    return [_cmd("MOVE_TO", my_player_id, team_id,
                 {"target_x": tx, "target_y": ty, "sprint": True})], "open-lane"


def _force_shot(commands, cmd, ctype, params, my_player_id, team_id,
                aim: str, tag: str, d_goal: float):
    """Rewrite/patch the command into a distance-scaled shot at `aim`.
    Tag is only reported when something actually changed."""
    power = shot_power(d_goal, aim)
    if ctype != "SHOOT":
        return [_cmd("SHOOT", my_player_id, team_id,
                     {"aim_location": aim, "power": power})], tag
    changed = (params.get("power") != power or params.get("aim_location") != aim)
    params["power"] = power
    params["aim_location"] = aim
    return commands, (tag if changed else None)


def _ev_model(cfg):
    import ev_policy
    return ev_policy.get_model(cfg.ev_model)


_OWN: dict[tuple[int, int], list] = {}


def _track_regain(game_state, team_id, my_pid, holder):
    """Remember when OUR team last won the ball (per player process)."""
    t = float(game_state.get("gameTime", 0) or 0)
    k = (team_id, my_pid)
    st = _OWN.get(k)
    if st is None or t < st[2]:
        st = [None, -1e9, t]
        _OWN[k] = st
    owner = None if holder is None else ("US" if _is_my_team(holder, team_id) else "THEM")
    if owner is not None:
        if owner == "US" and st[0] == "THEM":
            st[1] = t
        st[0] = owner
    st[2] = t


def seconds_since_regain(game_state, team_id, my_pid) -> float:
    st = _OWN.get((team_id, my_pid))
    if st is None:
        return 1e9
    return float(game_state.get("gameTime", 0) or 0) - st[1]


def apply_overrides(commands: list[dict], game_state: dict, team_id: int,
                    my_player_id: int, position_label: str,
                    cfg: OverrideConfig | None) -> tuple[list[dict], str | None]:
    """Rewrite the LLM's first command when it violates a hard tactical rule.
    Returns (commands, override_tag) — tag is None when nothing was changed."""
    if cfg is None or not commands:
        return commands, None

    ball = game_state.get("ball", {}) or {}
    ball_pos = ball.get("position", {}) or {}
    players = game_state.get("players", []) or []
    me = next((p for p in players
               if _player_idx(p) == my_player_id and _is_my_team(p, team_id)), None)
    if me is None:
        return commands, None
    me_pos = me.get("position", {}) or {}

    my_goal_x, opp_goal_x = get_goal_positions(team_id)
    d_goal = dist(me_pos, {"x": opp_goal_x, "y": 0})
    dir_my = 1.0 if my_goal_x > 0 else -1.0
    holder = resolve_holder(ball, players)
    i_have = holder is me
    teammate_has = holder is not None and holder is not me and _is_my_team(holder, team_id)
    opp_has = holder is not None and not _is_my_team(holder, team_id)
    opponents = [p for p in players if not _is_my_team(p, team_id)]
    _track_regain(game_state, team_id, my_player_id, holder)

    cmd = commands[0]
    ctype = cmd.get("commandType")
    params = cmd.get("parameters") or {}
    cmd["parameters"] = params

    play_mode = game_state.get("playMode")
    set_piece = _is_set_piece(play_mode)

    if my_player_id == 0 and cfg.gk_ev and not set_piece:
        import def_search
        if i_have:
            c, _info = def_search.gk_distribute(game_state, team_id,
                                                model=_ev_model(cfg))
            if c is not None:
                return [c], "gk-ev"
        elif not (holder is None and _ball_in_our_box(ball_pos, my_goal_x)
                  and dist(me_pos, ball_pos) <= 8.0):
            gx, gy = def_search.gk_position(game_state, team_id)
            return [_cmd("MOVE_TO", my_player_id, team_id,
                         {"target_x": round(gx, 1), "target_y": round(gy, 1),
                          "sprint": dist(me_pos, {"x": gx, "y": gy}) > 3})], "gk-angle"
    if cfg.ev_support and teammate_has and my_player_id != 0 and not set_piece:
        import ev_policy
        c, _info = ev_policy.support_spot(
            game_state, team_id, my_player_id, position_label, model=_ev_model(cfg),
            params={"roles": tuple(r.strip() for r in cfg.ev_support_roles.split(",")),
                    "line_shift": cfg.ev_support_shift})
        if c is not None:
            return [c], "ev-support"
    if cfg.def_search and opp_has and my_player_id != 0 and not set_piece:
        import def_search
        c, _info = def_search.assign(game_state, team_id, my_player_id, model=_ev_model(cfg))
        if c is not None:
            return [c], "dsearch"

    if my_player_id == 0 and i_have:
        return _gk_possession(commands, cmd, ctype, params, my_player_id,
                              team_id, cfg, me_pos, opp_goal_x, opponents,
                              players, d_goal, set_piece, dir_my)

    if cfg.ev_select and i_have and not set_piece:
        import ev_policy
        ev_cmd, info = ev_policy.choose(game_state, team_id, my_player_id, position_label,
                                        model=_ev_model(cfg))
        if ev_cmd is not None:
            return [ev_cmd], "ev"

    if cfg.always_blast and i_have:
        if not set_piece:
            if cfg.build_from_back and opponents:
                pressed = min(dist(o.get("position", {}) or {}, me_pos)
                              for o in opponents) <= cfg.blast_pressure_dist
                lane = cfg.outlet_lane_radius * (1.5 if pressed else 1.0)
                outlet = _best_outlet(cfg, players, team_id, my_player_id,
                                      me_pos, opp_goal_x, opponents, lane_radius=lane)
                if outlet is not None:
                    tgt = next(p for p in players
                               if _player_idx(p) == outlet and _is_my_team(p, team_id))
                    long_ball = dist(me_pos, tgt.get("position", {}) or {}) > 45
                    return [_cmd("PASS", my_player_id, team_id,
                                 {"target_player_id": outlet,
                                  "type": "AERIAL" if long_ball else "THROUGH"})], "build"
                launch = _most_advanced_mate(players, team_id, my_player_id,
                                             opp_goal_x, me_pos,
                                             cfg.outlet_min_gain, dir_my)
                if launch is not None:
                    return [_cmd("PASS", my_player_id, team_id,
                                 {"target_player_id": launch, "type": "AERIAL"})], "launch"
                if not pressed and my_player_id != 0:
                    return [_cmd("MOVE_TO", my_player_id, team_id,
                                 {"target_x": round(me_pos.get("x", 0) - 14 * dir_my, 1),
                                  "target_y": 10.0 if me_pos.get("y", 0) >= 0 else -10.0,
                                  "sprint": True})], "carry"
            aim, _, _ = _best_shot_aim(me_pos, opp_goal_x, opponents)
            return _force_shot(commands, cmd, ctype, params, my_player_id, team_id,
                               aim, "blast", d_goal)

    if my_player_id == 0:
        if not i_have and not _is_set_piece(game_state.get("playMode")):
            in_box = _ball_in_our_box(ball_pos, my_goal_x)
            gk_smother_dist = min(cfg.intercept_radius, 8.0)
            d_ball = dist(me_pos, ball_pos)
            lx, ly = _gk_line_target(my_goal_x, ball_pos)
            if in_box and holder is None and d_ball <= gk_smother_dist:
                return [_cmd("INTERCEPT", my_player_id, team_id,
                             {"aggressive": d_ball <= 5.0}, duration=2)], "gk-smother"
            tag = "gk-cover" if ctype in _NEEDS_BALL_CMDS else "gk-line"
            return [_cmd("MOVE_TO", my_player_id, team_id,
                         {"target_x": lx, "target_y": ly, "sprint": False})], tag
        return commands, None

    if i_have:
        lane_radius = 1.5 if d_goal <= 25 else 2.5
        _, _, center_perp = _best_shot_aim(me_pos, opp_goal_x, opponents)
        center_clear = center_perp >= lane_radius
        point_blank = d_goal <= 15
        lead_m = lead_for_override(cfg, me_pos, opp_goal_x, dir_my, ctype, params)
        effective_d = _effective_d_goal(me_pos, opp_goal_x, lead_m)
        in_shoot_window = (d_goal <= cfg.shoot_threshold
                           or effective_d <= cfg.shoot_threshold)

        if cfg.enforce_shot and in_shoot_window:
            return _force_shot(commands, cmd, ctype, params, my_player_id, team_id,
                               "CENTER",
                               "shoot" if ctype != "SHOOT" else "aim", d_goal)

        if (cfg.enforce_shot and position_label in ("MID", "FWD1", "FWD2")
                and effective_d <= cfg.longshot_max
                and not in_shoot_window):
            in_opp_half = me_pos.get("x", 0) * -dir_my > cfg.opp_half_line
            if in_opp_half or ctype == "SHOOT" or center_clear:
                return _force_shot(commands, cmd, ctype, params, my_player_id,
                                   team_id, "CENTER",
                                   "snap-shot" if ctype != "SHOOT" else "aim",
                                   d_goal)
            if ctype in ("MOVE_TO", "PASS"):
                return _open_lane_carry(cfg, me_pos, opp_goal_x, opponents,
                                        my_player_id, team_id, lane_radius)

        if cfg.counter_attack:
            pressure = count_opponents_in_our_half(game_state, team_id)
            in_shoot_range = effective_d <= cfg.longshot_max
            panic_blast = ctype == "SHOOT" and not in_shoot_range
            pressed_carry = (pressure >= cfg.press_bodies and ctype == "MOVE_TO"
                             and not in_shoot_range)
            if panic_blast or pressed_carry:
                outlet = _best_outlet(cfg, players, team_id, my_player_id,
                                      me_pos, opp_goal_x, opponents)
                if outlet is not None:
                    return [_cmd("PASS", my_player_id, team_id,
                                 {"target_player_id": outlet,
                                  "type": "THROUGH"})], "counter"
                if panic_blast:
                    ty = cfg.wing_y or (10.0 if me_pos.get("y", 0) >= 0 else -10.0)
                    tx = me_pos.get("x", 0) - 12 * dir_my
                    return [_cmd("MOVE_TO", my_player_id, team_id,
                                 {"target_x": round(tx, 1), "target_y": ty,
                                  "sprint": True})], "counter"

        if ctype == "MOVE_TO":
            tx, ty = params.get("target_x"), params.get("target_y")
            tag = None
            toward_opp = isinstance(tx, (int, float)) and (tx - me_pos.get("x", 0)) * -dir_my > 0
            if cfg.wing_y and isinstance(ty, (int, float)) and abs(ty) < 8 and toward_opp:
                ty = cfg.wing_y
                params["target_y"] = ty
                tag = "wing"
            if isinstance(tx, (int, float)) and tx * -dir_my > cfg.carry_cap_x:
                tx = cfg.carry_cap_x * -dir_my
                params["target_x"] = tx
                tag = tag or "cap"
            if (params.get("sprint") and isinstance(tx, (int, float))
                    and isinstance(ty, (int, float))
                    and dist(me_pos, {"x": tx, "y": ty}) <= 12):
                params["sprint"] = False
            return commands, tag
        return commands, None

    if teammate_has:
        support = ctype in _NEEDS_BALL_CMDS or ctype in _CHASE_CMDS
        if not support and position_label in ("MID", "FWD1", "FWD2"):
            if ctype in ("MARK", "FOLLOW_PLAYER", "SET_STANCE"):
                support = True
            elif ctype == "MOVE_TO":
                tx = params.get("target_x")
                advancing = (isinstance(tx, (int, float))
                             and (tx - me_pos.get("x", 0)) * -dir_my > 2)
                support = not advancing
        if support:
            sx, sy = _support_spot(position_label, team_id, ball_pos, cfg.lane_y)
            return [_cmd("MOVE_TO", my_player_id, team_id,
                         {"target_x": round(sx, 1), "target_y": round(sy, 1),
                          "sprint": True})], "support"
        return commands, None

    my_team = [p for p in players if _is_my_team(p, team_id)]
    closest = min(my_team, key=lambda p: dist(p.get("position", {}) or {}, ball_pos),
                  default=None)
    designated = closest is not None and _player_idx(closest) == my_player_id
    if cfg.press_count > 1 and not designated:
        outfield = sorted((p for p in my_team if _player_idx(p) != 0),
                          key=lambda p: dist(p.get("position", {}) or {}, ball_pos))
        designated = my_player_id in [_player_idx(p) for p in outfield[:cfg.press_count]]
    pressure = count_opponents_in_our_half(game_state, team_id)
    deep = pressure >= cfg.press_bodies
    ball_in_our_half = ball_pos.get("x", 0) * dir_my > 0

    def _duty(tag: str):
        return [_defensive_duty(cfg, position_label, team_id, my_player_id,
                                me_pos, ball_pos, opponents, holder, deep,
                                my_team, designated)], tag

    if ctype in _NEEDS_BALL_CMDS:
        if designated:
            return [_cmd("PRESS_BALL", my_player_id, team_id,
                         {"intensity": 1.0}, duration=3)], "phantom"
        return _duty("phantom")

    if designated:
        if opp_has:
            hpos = holder.get("position", {}) or {}
            if dist(me_pos, hpos) <= cfg.tackle_dist * 2.0 and ctype != "SLIDE_TACKLE":
                return [_cmd("SLIDE_TACKLE", my_player_id, team_id,
                             {"target_player_id": -1, "sprint": True})], "tackle"
            if ctype == "PRESS_BALL" and dist(me_pos, hpos) <= cfg.tackle_dist * 2.5:
                return [_cmd("SLIDE_TACKLE", my_player_id, team_id,
                             {"target_player_id": -1, "sprint": True})], "tackle"
            if (position_label in ("DEF", "MID")
                    and not _ball_in_our_box(ball_pos, my_goal_x, 40.0)
                    and dist(me_pos, hpos) > cfg.press_radius * 0.6):
                return _duty("press-cap")
        elif holder is None and ctype != "INTERCEPT":
            if dist(me_pos, ball_pos) <= cfg.intercept_radius:
                if position_label in ("FWD1", "FWD2"):
                    if abs(me_pos.get("x", 0) - my_goal_x) > 28:
                        return _duty("no-chase")
                return [_cmd("INTERCEPT", my_player_id, team_id,
                             {"aggressive": True}, duration=2)], "intercept"
        if ctype == "MOVE_TO":
            tx = params.get("target_x")
            if (isinstance(tx, (int, float))
                    and _target_past_line(tx, position_label, team_id, ball_pos,
                                          deep, cfg.anchor_slack)):
                return _duty("press-cap")
        return commands, None

    chasing = ctype in _CHASE_CMDS
    if not chasing and ctype == "MOVE_TO":
        tx, ty = params.get("target_x"), params.get("target_y")
        if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
            chasing = dist({"x": tx, "y": ty}, ball_pos) < cfg.chase_radius

    if not designated:
        near_opp = min(opponents, key=lambda o: dist(me_pos, o.get("position", {}) or {}),
                       default=None)
        if near_opp is not None:
            d_opp = dist(me_pos, near_opp.get("position", {}) or {})
            if d_opp <= cfg.mark_radius and ctype in ("MOVE_TO", "SET_STANCE", "FOLLOW_PLAYER"):
                return _duty("mark-near")

    if cfg.no_chase and chasing:
        return _duty("no-chase")

    if cfg.compact_anchor and ctype in ("MOVE_TO", "MARK", "SET_STANCE", "FOLLOW_PLAYER",
                                        "PRESS_BALL", "INTERCEPT"):
        if ctype == "MARK" and not designated and position_label in ("DEF", "MID"):
            if not _ball_in_our_box(ball_pos, my_goal_x, 35.0):
                return _duty("hold-line")
        if ctype in ("MOVE_TO", "MARK", "PRESS_BALL", "INTERCEPT"):
            tx = params.get("target_x")
            if isinstance(tx, (int, float)):
                if _target_past_line(tx, position_label, team_id, ball_pos,
                                     deep, cfg.anchor_slack):
                    return _duty("anchor")
            if ctype == "MOVE_TO":
                ax, ay = _compact_anchor(cfg, position_label, team_id, ball_pos, me_pos,
                                         my_team, my_player_id, deep)
                tx, ty = params.get("target_x"), params.get("target_y")
                bad_target = not (isinstance(tx, (int, float)) and isinstance(ty, (int, float)))
                if bad_target or dist({"x": tx, "y": ty}, {"x": ax, "y": ay}) > cfg.anchor_slack:
                    return _duty("anchor")
            elif not designated:
                return _duty("hold-line")

    return commands, None
