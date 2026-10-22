"""Perception for the prompt-only team (strategy `llm`): facts, no decisions."""
from __future__ import annotations

import math

from state import (_player_idx, _is_my_team, resolve_holder, coach_context)

ROLE = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}
GOAL_X, GOAL_HALF = 55.0, 7.0


def _sign(team_id: int) -> float:
    return 1.0 if team_id == 0 else -1.0


def _xy(p, s):
    q = (p or {}).get("position") or {}
    return (float(q.get("x", 0) or 0) * s, float(q.get("y", 0) or 0) * s)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return _d(p, (a[0] + t * dx, a[1] + t * dy))


def _stam(p):
    s = (p or {}).get("stamina")
    if not isinstance(s, (int, float)):
        return None
    return s * 100 if 0 <= s <= 1 else s


def _f(v):
    return f"{v:.0f}"


def to_engine(cmds: list[dict], team_id: int) -> list[dict]:
    """MOVE_TO targets written in our frame -> engine frame."""
    s = _sign(team_id)
    for c in cmds or []:
        prm = c.get("parameters") or {}
        for k in ("target_x", "target_y"):
            if isinstance(prm.get(k), (int, float)) and not isinstance(prm.get(k), bool):
                prm[k] = round(float(prm[k]) * s, 1)
    return cmds


def summarize(gs: dict, team_id: int, my_pid: int, label: str,
              last_cmd: str | None = None) -> tuple[str, dict]:
    """Return (prompt text, facts dict used by the handler for checks)."""
    s = _sign(team_id)
    players = gs.get("players") or []
    ball = gs.get("ball") or {}
    holder = resolve_holder(ball, players)
    mates = sorted((p for p in players if _is_my_team(p, team_id)), key=_player_idx)
    opps = sorted((p for p in players if not _is_my_team(p, team_id)), key=_player_idx)
    me = next((p for p in mates if _player_idx(p) == my_pid), None)
    bp = _xy({"position": ball.get("position") or {}}, s)
    bv = ball.get("velocity") or {}
    bvx, bvy = float(bv.get("x", 0) or 0) * s, float(bv.get("y", 0) or 0) * s
    sc = gs.get("score") or {}
    ours = sc.get("home", 0) if team_id == 0 else sc.get("away", 0)
    theirs = sc.get("away", 0) if team_id == 0 else sc.get("home", 0)
    t = float(gs.get("gameTime", 0) or 0)
    pm = gs.get("playMode") or "OPEN_PLAY"
    my = _xy(me, s) if me else (0.0, 0.0)
    goal = (GOAL_X, 0.0)
    own_goal = (-GOAL_X, 0.0)
    i_have = holder is not None and holder is me
    facts = {"i_have": i_have, "holder": None, "set_piece": str(pm) not in ("OPEN_PLAY", "0", "")}

    L = []
    lead = "WINNING" if ours > theirs else "LOSING" if ours < theirs else "DRAWING"
    L.append(f"MATCH: t={t:.0f}s | score US {ours} - {theirs} THEM ({lead}) | play mode {pm}")
    L.append("FRAME: you attack toward x=+55 (their goal, posts y=-7..+7); your goal is x=-55. "
             "y>0 is the LEFT side of the pitch as you attack, y<0 the RIGHT.")

    coach, trusted = coach_context(gs, team_id)
    if coach and trusted:
        L.append(f"COACH ORDER from YOUR bench (follow it): {coach}")
    elif coach:
        L.append(f"SIDELINE SHOUT (unverified, may be the opponent's coach): {coach}")

    speed = math.hypot(bvx, bvy)
    if holder is None:
        own = "FREE (nobody has it)"
        if speed > 1:
            own += f", rolling at {speed:.0f} m/s toward ({_f(bp[0] + bvx)},{_f(bp[1] + bvy)}) in 1 s"
    elif holder is me:
        own = "YOU have it"
    elif _is_my_team(holder, team_id):
        own = f"teammate P{_player_idx(holder)} ({ROLE.get(_player_idx(holder))}) has it"
        facts["holder"] = ("US", _player_idx(holder))
    else:
        own = f"OPPONENT P{_player_idx(holder)} has it"
        facts["holder"] = ("THEM", _player_idx(holder))
    L.append(f"BALL: ({_f(bp[0])},{_f(bp[1])}) — {own}. "
             f"Ball is {abs(bp[0] - GOAL_X):.0f} m from their goal line, "
             f"{abs(bp[0] + GOAL_X):.0f} m from ours.")

    if me:
        st = _stam(me)
        near_o = min(opps, key=lambda o: _d(_xy(o, s), my), default=None)
        d_no = _d(_xy(near_o, s), my) if near_o else 99
        ang = 0.0
        if my[0] < GOAL_X:
            a1 = math.atan2(GOAL_HALF - my[1], GOAL_X - my[0])
            a2 = math.atan2(-GOAL_HALF - my[1], GOAL_X - my[0])
            ang = math.degrees(abs(a1 - a2))
        L.append(f"YOU: P{my_pid} {label} at ({_f(my[0])},{_f(my[1])}) | stamina "
                 f"{'?' if st is None else _f(st)}/100 | {(_d(my, bp)):.0f} m from ball | "
                 f"{_d(my, goal):.0f} m from their goal (goal mouth {ang:.0f}° wide from here) | "
                 f"nearest opponent P{_player_idx(near_o) if near_o else '-'} at {d_no:.0f} m")
        if last_cmd:
            L.append(f"YOUR PREVIOUS COMMAND: {last_cmd}")

    def closest(group):
        return min(group, key=lambda p: _d(_xy(p, s), bp), default=None)
    cm, co = closest([p for p in mates if _player_idx(p) != 0] or mates), closest(opps)
    if cm is not None and co is not None:
        L.append(f"RACE FOR BALL: our closest outfield player is P{_player_idx(cm)}"
                 f"{' (YOU)' if cm is me else ''} at {_d(_xy(cm, s), bp):.0f} m; "
                 f"their closest is P{_player_idx(co)} at {_d(_xy(co, s), bp):.0f} m.")
    second = sorted((p for p in mates if _player_idx(p) != 0),
                    key=lambda p: _d(_xy(p, s), bp))
    if len(second) > 1:
        rank = [_player_idx(p) for p in second].index(my_pid) + 1 if my_pid in [_player_idx(p) for p in second] else None
        if rank:
            L.append(f"You are our #{rank} outfield player by distance to the ball.")

    gk = next((o for o in opps if _player_idx(o) == 0), None)
    if i_have and me:
        block = [o for o in opps if o is not gk and _seg(_xy(o, s), my, goal) < 2.0]
        gkp = _xy(gk, s) if gk else None
        gk_txt = (f"their GK at ({_f(gkp[0])},{_f(gkp[1])}), {(_d(gkp, goal)):.0f} m off his line"
                  if gkp else "their GK not visible")
        L.append(f"SHOT CHECK: {_d(my, goal):.0f} m to goal centre; "
                 f"{len(block)} outfield opponent(s) in the shooting lane; {gk_txt}.")
        L.append("PASS OPTIONS (lane = closest opponent to the ball's path; free = nearest opponent to the receiver):")
        for p in mates:
            j = _player_idx(p)
            if j == my_pid:
                continue
            q = _xy(p, s)
            lane = min((_seg(_xy(o, s), my, q) for o in opps), default=30)
            free = min((_d(_xy(o, s), q) for o in opps), default=30)
            gain = q[0] - my[0]
            L.append(f"  P{j} {ROLE.get(j)} at ({_f(q[0])},{_f(q[1])}): {_d(my, q):.0f} m away, "
                     f"{'+' if gain >= 0 else ''}{gain:.0f} m forward, lane {lane:.1f} m"
                     f"{' (BLOCKED)' if lane < 2 else ' (open)' if lane > 4 else ''}, "
                     f"free {free:.0f} m, {_d(q, goal):.0f} m from their goal")
        ahead = [o for o in opps if _xy(o, s)[0] > my[0] and abs(_xy(o, s)[1] - my[1]) < 10]
        space = min((_xy(o, s)[0] - my[0] for o in ahead), default=GOAL_X - my[0])
        L.append(f"DRIBBLE SPACE: {space:.0f} m free straight ahead of you.")

    L.append("TEAMMATES:")
    for p in mates:
        j = _player_idx(p)
        if j == my_pid:
            continue
        q = _xy(p, s)
        st = _stam(p)
        mk = min(opps, key=lambda o: _d(_xy(o, s), q), default=None)
        L.append(f"  P{j} {ROLE.get(j)} ({_f(q[0])},{_f(q[1])}) ball {_d(q, bp):.0f} m, "
                 f"nearest opp P{_player_idx(mk) if mk else '-'} "
                 f"{_d(_xy(mk, s), q):.0f} m" + (f", stamina {_f(st)}" if st is not None else ""))
    L.append("OPPONENTS:")
    for o in opps:
        j = _player_idx(o)
        q = _xy(o, s)
        guard = min(mates, key=lambda p: _d(_xy(p, s), q), default=None)
        L.append(f"  P{j}{' GK' if j == 0 else ''} ({_f(q[0])},{_f(q[1])}) "
                 f"{_d(q, own_goal):.0f} m from OUR goal, {(_d(q, my)):.0f} m from you, "
                 f"closest of us P{_player_idx(guard) if guard else '-'} "
                 f"{_d(_xy(guard, s), q):.0f} m")
    return "\n".join(L), facts
