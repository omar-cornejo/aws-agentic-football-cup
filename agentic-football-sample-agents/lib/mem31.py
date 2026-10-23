"""mem3.1: mem3 plus lane-gated CENTER shots, Markov + constraints carrier, lane block and loose-ball rules."""
from __future__ import annotations

import math

import tuning as tuning_mod
from state import _is_my_team, _player_idx

GOAL_CLOSE = ((20.0, 0.55), (30.0, 0.45))


def on() -> bool:
    return bool(tuning_mod.get("mem31", False))


def _p(key, default):
    return float(tuning_mod.get(key, default))


def _xy(p):
    q = p.get("position") or {}
    return float(q.get("x", 0) or 0), float(q.get("y", 0) or 0)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return _d(p, (a[0] + t * dx, a[1] + t * dy))


class Geo:
    """Metric helpers on top of mem3._Ctx (which already holds the state in metres)."""

    def __init__(self, ctx):
        self.c = ctx
        self.me = _xy(ctx.me) if ctx.me is not None else (0.0, 0.0)
        self.goal = (ctx.opp_goal_x, 0.0)
        self.own_goal = (ctx.my_goal_x, 0.0)
        opps = ctx.opps
        self.keeper = min(opps, key=lambda o: _d(_xy(o), self.goal), default=None)
        self.field = [_xy(o) for o in opps if o is not self.keeper]
        self.all_opps = [_xy(o) for o in opps]

    def lane_clear(self, a, b, r, upto=None, pts=None):
        """No opponent within r metres of segment a-b (only its first `upto` m)."""
        if upto is not None:
            L = _d(a, b)
            if L > upto > 0:
                f = upto / L
                b = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        return all(_seg(o, a, b) > r for o in (self.field if pts is None else pts))

    def shot_ok(self, frm=None):
        """(ok, why). Close: always. Deep blast: first 25 m clear. Else full lane clear."""
        frm = frm or self.me
        d = _d(frm, self.goal)
        r = _p("m31_lane_r", 2.5)
        if d <= _p("m31_free_shot_m", 30):
            return True, "close"
        if d >= _p("m31_deep_m", 75):
            ok = self.lane_clear(frm, self.goal, r + 0.5, upto=25.0)
            return ok, "deep-clear" if ok else "deep-blocked"
        ok = self.lane_clear(frm, self.goal, r)
        return ok, "lane-clear" if ok else "lane-blocked"

    def p_close(self, xy):
        d = _d(xy, self.goal)
        for lim, p in GOAL_CLOSE:
            if d <= lim:
                return p
        return 0.0

    def pass_point(self, idx, kind):
        q = _xy(self.c.mates[idx])
        if kind == "THROUGH":
            return (max(-54.0, min(54.0, q[0] + 6 * self.c.att)), q[1])
        return q

    def intercept_margin(self, tgt):
        """Seconds the ball beats the fastest opponent to any point of the lane.
        Real-engine ball: a full shot crosses ~100 m in 2 s, so passes are taken
        at `m31_ball_speed` (22 m/s) instead of the simulator's 15 m/s."""
        vb = _p("m31_ball_speed", 22.0)
        vo = _p("m31_opp_speed", 7.0)
        L = _d(self.me, tgt) or 1.0
        best = 9.0
        for k in range(1, 11):
            t = k / 10
            pt = (self.me[0] + (tgt[0] - self.me[0]) * t, self.me[1] + (tgt[1] - self.me[1]) * t)
            to = min((max(0.0, _d(o, pt) - 1.2) / vo for o in self.all_opps), default=9.0)
            best = min(best, to - L * t / vb)
        return best

    def pass_ok(self, idx, kind="GROUND", margin=None):
        if idx not in self.c.mates or idx in (0, self.c.pid):
            return False
        tgt = self.pass_point(idx, kind)
        if _d(self.me, tgt) < 4:
            return False
        m = self.intercept_margin(tgt)
        recv = min((_d(o, _xy(self.c.mates[idx])) for o in self.all_opps), default=30.0)
        return m >= (_p("m31_pass_margin", 0.15) if margin is None else margin) and \
            recv > _p("m31_recv_free_m", 2.0)

    def carry_target(self, dy=0.0, step=None):
        step = _p("m31_carry_m", 10) if step is None else step
        tx = max(-52.0, min(52.0, self.me[0] + step * self.c.att))
        ty = max(-27.0, min(27.0, self.me[1] + dy))
        return tx, ty

    def carry_ok(self, t):
        if not self.lane_clear(self.me, t, 2.5, pts=self.all_opps):
            return False
        return min((_d(o, t) for o in self.all_opps), default=30.0) > 3.0

    def dribble_at_goal(self):
        """Carry toward the goal centre, sidestepping the nearest blocker."""
        gx, gy = self.goal
        L = _d(self.me, self.goal) or 1.0
        step = min(_p("m31_carry_m", 10), L)
        base = (self.me[0] + (gx - self.me[0]) / L * step, self.me[1] + (gy - self.me[1]) / L * step)
        best = None
        for off in (0.0, -7.0, 7.0, -12.0, 12.0):
            t = (base[0], max(-27.0, min(27.0, base[1] + off)))
            if self.carry_ok(t):
                best = t
                break
        if best is None:
            near = min(self.all_opps, key=lambda o: _d(o, self.me), default=None)
            side = 7.0 if near is None or near[1] <= self.me[1] else -7.0
            best = (base[0], max(-27.0, min(27.0, self.me[1] + side)))
        return best


def _shoot(ctx):
    return ctx.cmd("SHOOT", {"aim_location": "CENTER", "power": 1.0})


def _move(ctx, t, sprint=True):
    return ctx.cmd("MOVE_TO", {"target_x": round(t[0], 1), "target_y": round(t[1], 1),
                               "sprint": sprint})


def _pass(ctx, g, idx, kind):
    ptype = "AERIAL" if _d(g.me, g.pass_point(idx, kind)) > 35 else kind
    return ctx.cmd("PASS", {"target_player_id": idx, "type": ptype})


def assist_option(ctx, g, min_gain=8.0):
    """Safe pass to a free teammate in the shoot zone with his own lane clear."""
    best = None
    d_me = _d(g.me, g.goal)
    for idx, p in ctx.mates.items():
        if idx in (0, ctx.pid):
            continue
        for kind in ("GROUND", "THROUGH"):
            t = g.pass_point(idx, kind)
            pc = g.p_close(t)
            if pc <= 0 or _d(t, g.goal) > d_me - min_gain:
                continue
            if not g.pass_ok(idx, kind, margin=_p("m31_assist_margin", 0.2)):
                continue
            if not g.shot_ok(t)[0]:
                continue
            v = 0.9 * pc
            if best is None or v > best[0]:
                best = (v, idx, kind)
    return best


def coach_shoot(ctx):
    """«shoot»/«attack» with the ball: (command, tag)."""
    g = Geo(ctx)
    d = _d(g.me, g.goal)
    if d <= _p("m31_coach_shoot_m", 35) or g.shot_ok()[0]:
        return _shoot(ctx), "coach-shoot"
    a = assist_option(ctx, g)
    if a is not None:
        return _pass(ctx, g, a[1], a[2]), "coach-shoot-pass"
    return _move(ctx, g.dribble_at_goal()), "coach-shoot-carry"


def carrier(ctx, game_state, team_id, pid, label):
    """Markov EV ranking + metric constraints. (command, tag) or (None, None)."""
    import ev_policy
    model = ev_policy.get_model(str(tuning_mod.get("ev_model", "ev_model_f3.json") or ""))
    ec = ev_policy._Ctx(game_state, team_id, pid, model)
    if not ec.ok:
        return None, None
    g = Geo(ctx)
    opts = list(ec.options(biased=True))
    a = assist_option(ctx, g)
    if a is not None:
        opts.append((a[0], "assist", _pass(ctx, g, a[1], a[2])))
    opts.sort(key=lambda o: -o[0])
    for ev, lab, cmd in opts:
        if lab == "shoot":
            ok, why = g.shot_ok()
            if ok:
                return _shoot(ctx), f"m31-shot-{why}"
        elif lab == "assist":
            return cmd, "m31-assist"
        elif lab.startswith("pass"):
            prm = cmd.get("parameters") or {}
            idx = prm.get("target_player_id")
            kind = "THROUGH" if prm.get("type") == "THROUGH" or "THROUGH" in lab else "GROUND"
            if g.pass_ok(idx, kind):
                return _pass(ctx, g, idx, kind), "m31-pass"
        elif lab.startswith("carry"):
            try:
                dy = float(lab[5:])
            except ValueError:
                dy = 0.0
            for ddy in (math.copysign(8.0, dy) if dy else 0.0, 0.0, -8.0, 8.0):
                t = g.carry_target(ddy)
                if g.carry_ok(t):
                    return _move(ctx, t), "m31-carry"
    if _d(g.me, g.goal) >= _p("m31_deep_m", 75):
        return _shoot(ctx), "m31-clear"
    near = min((_d(o, g.me) for o in g.all_opps), default=30.0)
    if near > 4.0:
        return _move(ctx, g.dribble_at_goal()), "m31-shield"
    ev, lab, cmd = max(ec.options(biased=True), key=lambda o: o[0])
    if lab == "shoot":
        return _shoot(ctx), "m31-ev"
    return None, None


def _nearest_two(ctx, point):
    out = sorted(((_d(_xy(p), point), i) for i, p in ctx.mates.items() if i != 0))
    return [i for _, i in out[:2]]


def defence(ctx):
    """Opponent has the ball: (command, tag) or (None, None) to let dsearch decide."""
    g = Geo(ctx)
    carrier_xy = _xy(ctx.holder)
    dc = _d(carrier_xy, g.own_goal)
    two = _nearest_two(ctx, carrier_xy)
    if dc <= _p("m31_double_m", 30) and len(two) == 2:
        if ctx.pid == two[0]:
            return ctx.press_or_tackle(), "m31-press"
        if ctx.pid == two[1]:
            L = dc or 1.0
            t = (carrier_xy[0] + (g.own_goal[0] - carrier_xy[0]) / L * 3.0,
                 carrier_xy[1] + (g.own_goal[1] - carrier_xy[1]) / L * 3.0)
            return _move(ctx, t), "m31-double"
        return None, None
    if ctx.pid == 1 and dc >= _p("m31_block_from_m", 35) and two and two[0] != 1:
        danger = [o for o in ctx.opps if o is not ctx.holder and o is not g.keeper
                  and _d(_xy(o), g.own_goal) < 22.0]
        if not danger:
            k = max(16.0, min(29.0, dc * 0.5))
            t = (g.own_goal[0] + (carrier_xy[0] - g.own_goal[0]) / dc * k,
                 g.own_goal[1] + (carrier_xy[1] - g.own_goal[1]) / dc * k)
            floor = _p("def_floor_m", -26)
            if t[0] * ctx.att < floor and carrier_xy[0] * ctx.att > floor + 6:
                t = (floor * ctx.att, t[1] * 0.6)
            return _move(ctx, t, sprint=_d(g.me, t) > 4), "m31-block"
    return None, None


def loose(ctx):
    """Nobody holds the ball: our outfield player nearest to where the ball
    will be in ~0.8 s goes to win it (a pass in flight -> the receiver)."""
    v = ((ctx.gs.get("ball") or {}).get("velocity")) or {}
    px = ctx.ball[0] + 0.8 * float(v.get("x", 0) or 0)
    py = ctx.ball[1] + 0.8 * float(v.get("y", 0) or 0)
    out = [(i, p) for i, p in ctx.mates.items() if i != 0]
    if not out:
        return None, None
    first = min(out, key=lambda ip: (_d(_xy(ip[1]), (px, py)), ip[0]))[0]
    if ctx.pid == first:
        return ctx.cmd("INTERCEPT", {"aggressive": True}, duration=2), "m31-loose"
    return None, None


def decide(ctx, game_state, team_id, pid, label):
    """Called by mem3.decide after bench orders and the shoot zone."""
    if ctx.set_piece or pid == 0:
        return None, None
    flag = lambda k: bool(tuning_mod.get(k, True))      # noqa: E731  (ablation switches)
    if ctx.i_have:
        return carrier(ctx, game_state, team_id, pid, label) if flag("m31_carrier") else (None, None)
    if ctx.opp_has:
        return defence(ctx) if flag("m31_defence") else (None, None)
    if ctx.holder is None:
        return loose(ctx) if flag("m31_loose") else (None, None)
    return None, None


def force_center(commands):
    """Every SHOOT goes CENTER (TL/TR: 0 goals in 24 real shots). Tag or None."""
    if not on() or not commands:
        return None
    c = commands[0]
    if c.get("commandType") != "SHOOT":
        return None
    prm = c.setdefault("parameters", {})
    if prm.get("aim_location") != "CENTER" or prm.get("power", 1.0) != 1.0:
        prm["aim_location"], prm["power"] = "CENTER", 1.0
        return "aim-center"
    return None


def log_fields(game_state, team_id, pid, command):
    """Extra DECISION fields for the next Markov refit from real matches:
    op = opponent positions (state units) on carrier ticks, ln = shot lane."""
    if not on():
        return {}
    try:
        players = game_state.get("players") or []
        out = {"op": [[round(_xy(p)[0], 2), round(_xy(p)[1], 2)]
                      for p in players if not _is_my_team(p, team_id)]}
        if (command or {}).get("commandType") == "SHOOT":
            import mem3
            ctx = mem3._Ctx(game_state, team_id, pid)
            if ctx.me is not None:
                out["ln"] = Geo(ctx).shot_ok()[1]
        return out
    except Exception:
        return {}


_COMMON = ("\n\nMEM3.1 OFF-BALL RULES (they win over older rules when they conflict). "
           "Code already decides for the ball carrier, the presser and bench orders; "
           "you decide where to stand.\n")
PROMPTS = {
    "GK": "- Stand on the straight line from the ball to the centre of our goal, on your line. "
          "Long shots are the main danger in this engine.\n",
    "DEF": "- Teammate has the ball: stay behind the ball as the last outfield player, "
           "central (|y|<=6), never deeper than x = my_goal_x*0.47.\n"
           "- Opponent has the ball: stand on the straight line from the ball to the centre "
           "of our goal (block the long shot). MARK only an opponent inside our half.\n"
           "- Loose ball and you are the nearest: INTERCEPT.\n",
    "MID": "- Teammate has the ball: offer a safe pass, 10-15 m from the carrier, central "
           "(|y|<=7), ahead of the DEF. Never MARK while we have the ball.\n"
           "- Opponent has the ball: screen the line from the ball to our goal; MARK only in our half.\n"
           "- Loose ball and you are the nearest: INTERCEPT.\n",
    "FWD1": "- Teammate has the ball or the ball is loose: NEVER MARK. Run into space on the "
            "LEFT channel ahead of the ball: MOVE_TO x = opp_goal_x*0.7, y = -10, sprint — "
            "be the pass target between their defenders and their goal.\n"
            "- Restart (KICK_OFF / RESETTING): MOVE_TO x = opp_goal_x*0.15, y = -10.\n"
            "- Opponent has the ball in their half: MARK the opponent nearest to you.\n"
            "- Keep 8 m or more from the other striker.\n",
    "FWD2": "- Teammate has the ball or the ball is loose: NEVER MARK. Run into space on the "
            "RIGHT channel ahead of the ball: MOVE_TO x = opp_goal_x*0.7, y = 10, sprint — "
            "be the pass target between their defenders and their goal.\n"
            "- Restart (KICK_OFF / RESETTING): MOVE_TO x = opp_goal_x*0.15, y = 10.\n"
            "- Opponent has the ball in their half: MARK the opponent nearest to you.\n"
            "- Keep 8 m or more from the other striker.\n",
}


def prompt_addendum(label: str) -> str:
    if not on() or not tuning_mod.get("m31_prompts", True):
        return ""
    body = PROMPTS.get(label)
    return _COMMON + body if body else ""
