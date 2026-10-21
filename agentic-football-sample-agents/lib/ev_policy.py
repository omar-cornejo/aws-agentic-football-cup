"""Expected-value action choice for the ball carrier (Markov / xT model)."""
from __future__ import annotations

import json
import math
import os

_PATH = os.path.join(os.path.dirname(__file__), "ev_model.json")

HALF_X, HALF_Y = 55.0, 30.0
GRID_X, GRID_Y = 12, 6
GOAL_HALF = 7.0


_CACHE = {}


def get_model(name="ev_model.json"):
    """Load (and cache) a model file from lib/. Returns None if missing."""
    if not name:
        name = "ev_model.json"
    if name not in _CACHE:
        try:
            with open(os.path.join(os.path.dirname(__file__), name), encoding="utf-8") as f:
                _CACHE[name] = json.load(f)
        except (OSError, json.JSONDecodeError):
            _CACHE[name] = None
    return _CACHE[name]


MODEL = get_model()
CHOICE_PARAMS = {"shot_bias": 1.0, "chase_bias": 1.0, "carry_len": 12.0,
                 "carry_dys": (0.0, -9.0, 9.0)}
SUPPORT_PARAMS = {"line_shift": 0.0, "lane_bonus": 1.25,
                  "roles": ("DEF", "MID", "FWD1", "FWD2")}
import tuning as _tun
LOOKAHEAD = {"on": bool(_tun.get("ev_lookahead", False)), "gamma": 0.8}
TRANSITION = {"on": bool(_tun.get("ev_transition", False)), "s": 4.0, "shift": 8.0}


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return _d(p, (a[0] + t * dx, a[1] + t * dy))


def cell(x, y, attack_dir):
    """Cell index in the attacking frame (attack toward +x)."""
    ax, ay = x * attack_dir, y * attack_dir
    i = int((ax + HALF_X) / (2 * HALF_X) * GRID_X)
    j = int((ay + HALF_Y) / (2 * HALF_Y) * GRID_Y)
    return min(GRID_X - 1, max(0, i)) * GRID_Y + min(GRID_Y - 1, max(0, j))


def f_shot(me, opp_goal_x, opponents, gk):
    dist = _d(me, (opp_goal_x, 0))
    ang = abs(math.atan2(me[1], abs(opp_goal_x - me[0]) + 1e-6))
    block = sum(1 for o in opponents if o is not gk and _seg(o, me, (opp_goal_x, 0)) < 2.0)
    gk_off = _d(gk, (opp_goal_x, 0)) if gk else 10.0
    return [1.0, dist / 10, (dist / 10) ** 2, ang, block, min(gk_off, 20) / 10]


def f_pass(me, target, opponents):
    length = _d(me, target)
    lane = min((_seg(o, me, target) for o in opponents), default=30.0)
    recv = min((_d(o, target) for o in opponents), default=30.0)
    return [1.0, length / 10, min(lane, 15) / 5, min(recv, 15) / 5]


def f_keep(me, opponents):
    near = min((_d(o, me) for o in opponents), default=30.0)
    crowd = sum(1 for o in opponents if _d(o, me) < 6)
    return [1.0, min(near, 15) / 5, crowd]


OPP_SPEED = 7.0
BALL_SPEED = 15.0


def f_shot2(me, opp_goal_x, opponents, gk):
    base = f_shot(me, opp_goal_x, opponents, gk)
    press = sum(1 for o in opponents if o is not gk and _d(o, me) < 3.0)
    a1 = math.atan2(GOAL_HALF - me[1], abs(opp_goal_x - me[0]) + 1e-6)
    a2 = math.atan2(-GOAL_HALF - me[1], abs(opp_goal_x - me[0]) + 1e-6)
    opening = abs(a1 - a2)
    gk_cover = 0.0
    if gk is not None:
        to_gk = math.atan2(gk[1] - me[1], abs(gk[0] - me[0]) + 1e-6)
        mid = (a1 + a2) / 2
        gk_cover = max(0.0, 1.0 - abs(to_gk - mid) / (opening / 2 + 1e-6))
    return base + [press, opening, gk_cover]


def intercept_margin(me, target, opponents):
    """Seconds the ball beats the fastest opponent to any point of the lane
    (negative = an opponent gets there first)."""
    best = 9.0
    L = _d(me, target) or 1.0
    for k in range(1, 9):
        t = k / 8
        pt = (me[0] + (target[0] - me[0]) * t, me[1] + (target[1] - me[1]) * t)
        tb = L * t / BALL_SPEED
        to = min((max(0.0, _d(o, pt) - 1.2) / OPP_SPEED for o in opponents), default=9.0)
        best = min(best, to - tb)
    return best


def f_pass2(me, target, opponents):
    base = f_pass(me, target, opponents)
    margin = max(-2.0, min(2.0, intercept_margin(me, target, opponents)))
    crowd = sum(1 for o in opponents if _d(o, target) < 4.0)
    return base + [margin, crowd]


def f_keep2(me, opponents, opp_vel=None):
    base = f_keep(me, opponents)
    vel = opp_vel or [(0.0, 0.0)] * len(opponents)
    proj = [(o[0] + v[0], o[1] + v[1]) for o, v in zip(opponents, vel)]
    near1 = min((_d(o, me) for o in proj), default=30.0)
    return base + [min(near1, 15) / 5]


def f_pass3(me, target, opponents):
    """Robust pass model: only lane clearance and interception margin.
    Length/receiver features overfit to the rivals seen in training."""
    x = f_pass2(me, target, opponents)
    return [x[0], x[2], x[4]]


FEATS = {1: (f_shot, f_pass, f_keep), 2: (f_shot2, f_pass2, f_keep2),
         3: (f_shot2, f_pass3, f_keep2)}


def _logit(w, x):
    z = sum(a * b for a, b in zip(w, x))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


class _Ctx:
    """Positions seen from one player's side."""

    def __init__(self, game_state, team_id, my_pid, m):
        from state import _player_idx, _is_my_team, get_goal_positions
        players = game_state.get("players") or []

        def pos(p):
            q = p.get("position") or {}
            return (float(q.get("x", 0) or 0), float(q.get("y", 0) or 0))

        me_p = next((p for p in players
                     if _player_idx(p) == my_pid and _is_my_team(p, team_id)), None)
        self.ok = me_p is not None and m is not None
        if not self.ok:
            return
        self.m, self.pid, self.team = m, my_pid, team_id
        self.gs = game_state
        self.me = pos(me_p)
        self.mates = {_player_idx(p): pos(p) for p in players
                      if _is_my_team(p, team_id) and _player_idx(p) != my_pid}
        self.opps = {_player_idx(p): pos(p) for p in players if not _is_my_team(p, team_id)}
        self.opp_list = list(self.opps.values())

        def vel(p):
            q = p.get("velocity") or {}
            return (float(q.get("x", 0) or 0), float(q.get("y", 0) or 0))

        self.opp_vel = [vel(p) for p in players if not _is_my_team(p, team_id)]
        self.fv = int(m.get("fv", 1))
        _, self.opp_goal_x = get_goal_positions(team_id)
        self.att = 1.0 if self.opp_goal_x > 0 else -1.0
        sc = game_state.get("score") or {}
        diff = (sc.get("home", 0) or 0) - (sc.get("away", 0) or 0)
        self.winning = (diff if team_id == 0 else -diff) > 0

    def V(self, x, y):
        return self.m["xt"][cell(x, y, self.att)]

    def _keep_feats(self):
        if self.fv == 2:
            return f_keep2(self.me, self.opp_list, self.opp_vel)
        return f_keep(self.me, self.opp_list)

    def ev_shoot(self):
        fs = FEATS[self.fv][0]
        return _logit(self.m["shot"], fs(self.me, self.opp_goal_x, self.opp_list,
                                         self.opps.get(0)))

    def pass_target(self, idx, kind):
        tp = self.mates.get(idx)
        if tp is None:
            return None
        if kind == "THROUGH":
            return (max(-HALF_X + 1, min(HALF_X - 1, tp[0] + 6 * self.att)), tp[1])
        return tp

    def ev_pass(self, idx, kind="GROUND"):
        tgt = self.pass_target(idx, kind)
        if tgt is None or _d(self.me, tgt) < 1:
            return 0.0
        fp = FEATS[self.fv][1]
        v = self.V(*tgt)
        if LOOKAHEAD["on"]:
            fs = FEATS[self.fv][0]
            others = [o for o in self.opp_list]
            p_next = _logit(self.m["shot"], fs(tgt, self.opp_goal_x, others, self.opps.get(0)))
            v = max(v, LOOKAHEAD["gamma"] * p_next)
        return _logit(self.m["pass"], fp(self.me, tgt, self.opp_list)) * v

    def ev_carry(self, tx, ty):
        p_keep = _logit(self.m["keep"], self._keep_feats())
        lane = min((_seg(o, self.me, (tx, ty)) for o in self.opp_list), default=30.0)
        return p_keep * (1.0 if lane > 3 else 0.6) * self.V(tx, ty)

    def ev_hold(self):
        """Doing nothing useful with the ball: keep it for a tick, no progress."""
        return _logit(self.m["keep"], self._keep_feats()) * self.V(*self.me)

    def options(self, biased=False):
        opts = []
        gk = self.opps.get(0)
        aim = "CENTER"
        if gk is not None and abs(gk[1]) > 1.0:
            aim = "TL" if (gk[1] * self.att) < 0 else "TR"
        cp = dict(CHOICE_PARAMS)
        if biased:
            import tuning as _t
            cp["shot_bias"] = float(_t.get("ev_shot_bias", cp["shot_bias"]))
        bias = cp["shot_bias"] * (1.0 if self.winning else cp["chase_bias"]) if biased else 1.0
        self.mem_why = None
        if biased:
            import match_memory
            d_goal = math.hypot(self.me[0] - self.opp_goal_x, self.me[1])
            mult, self.mem_why = match_memory.long_shot_multiplier(
                self.gs, self.team, self.pid, d_goal)
            bias *= mult
            if d_goal > 45:
                import memory_v2
                pm, why = memory_v2.long_shot_prior_mult(self.team, self.pid)
                if why:
                    bias *= pm
                    self.mem_why = (self.mem_why + "; " if self.mem_why else "") + why
        opts.append((self.ev_shoot() * bias, "shoot",
                     {"commandType": "SHOOT", "parameters": {"aim_location": aim, "power": 1.0}}))
        for idx in self.mates:
            if idx == 0:
                continue
            for kind in ("GROUND", "THROUGH"):
                tgt = self.pass_target(idx, kind)
                length = _d(self.me, tgt)
                if length < 4:
                    continue
                ptype = "AERIAL" if length > 35 else kind
                opts.append((self.ev_pass(idx, kind), f"pass{idx}-{ptype}",
                             {"commandType": "PASS",
                              "parameters": {"target_player_id": idx, "type": ptype}}))
        for dy in cp["carry_dys"]:
            tx = max(-HALF_X + 3, min(HALF_X - 3, self.me[0] + cp["carry_len"] * self.att))
            ty = max(-HALF_Y + 3, min(HALF_Y - 3, self.me[1] + dy))
            opts.append((self.ev_carry(tx, ty), f"carry{int(dy):+d}",
                         {"commandType": "MOVE_TO", "parameters": {
                             "target_x": round(tx, 1), "target_y": round(ty, 1), "sprint": True}}))
        return opts

    def ev_of(self, cmd):
        """EV of an arbitrary command (LLM's, a rule's, or ours)."""
        ct = (cmd or {}).get("commandType")
        prm = (cmd or {}).get("parameters") or {}
        if ct == "SHOOT":
            return self.ev_shoot(), "shoot"
        if ct in ("PASS", "GK_DISTRIBUTE"):
            idx = prm.get("target_player_id")
            kind = "THROUGH" if prm.get("type") == "THROUGH" else "GROUND"
            return self.ev_pass(idx, kind), f"pass{idx}"
        if ct == "MOVE_TO":
            tx, ty = prm.get("target_x"), prm.get("target_y")
            if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
                return self.ev_carry(tx, ty), "carry"
        return self.ev_hold(), "hold"


def choose(game_state, team_id, my_pid, position_label="MID", model=None):
    """Return (command, info) for the carrier, or (None, None) without a model."""
    ctx = _Ctx(game_state, team_id, my_pid, model or MODEL)
    if not ctx.ok:
        return None, None
    ev, label, cmd = max(ctx.options(biased=True), key=lambda o: o[0])
    cmd["playerId"] = my_pid
    cmd["teamId"] = team_id
    cmd["duration"] = 0
    return cmd, {"ev": round(ev, 3), "choice": label, "mem": ctx.mem_why}


def evaluate(game_state, team_id, my_pid, command, model=None):
    """Critic: how good was this carrier decision compared with the best option?"""
    ctx = _Ctx(game_state, team_id, my_pid, model or MODEL)
    if not ctx.ok:
        return None
    ev, kind = ctx.ev_of(command)
    best, label, _ = max(ctx.options(), key=lambda o: o[0])
    best = max(best, ev)
    return {"ev": round(ev, 4), "best": round(best, 4), "best_choice": label,
            "regret": round(best - ev, 4), "kind": kind, "v_here": round(ctx.V(*ctx.me), 4)}


SUPPORT_ROLE = {
    "DEF": (-26.0, -8.0),
    "MID": (-10.0, 14.0),
    "FWD1": (0.0, 48.0),
    "FWD2": (0.0, 48.0),
}


def support_spot(game_state, team_id, my_pid, position_label, model=None, params=None):
    """Teammate has the ball: run to the open zone with the highest threat V."""
    p = dict(SUPPORT_PARAMS)
    p.update(params or {})
    if position_label not in SUPPORT_ROLE or position_label not in p["roles"]:
        return None, None
    ctx = _Ctx(game_state, team_id, my_pid, model or MODEL)
    if not ctx.ok:
        return None, None
    b = (game_state.get("ball") or {}).get("position") or {}
    ball = (float(b.get("x", 0) or 0), float(b.get("y", 0) or 0))
    att = ctx.att
    bxa = ball[0] * att
    lo, hi = SUPPORT_ROLE[position_label]
    shift = p.get("line_shift", 0.0)
    if TRANSITION["on"] and position_label in ("FWD1", "FWD2"):
        import overrides as _ov
        if _ov.seconds_since_regain(game_state, team_id, my_pid) <= TRANSITION["s"]:
            shift += TRANSITION["shift"]
    lo, hi = bxa + lo + shift, bxa + hi + shift
    lo, hi = max(-50.0, lo), min(48.0, hi)
    if position_label == "DEF":
        lo, hi = min(lo, 25.0), min(hi, 30.0)
    side = {"FWD1": -1.0, "FWD2": 1.0}.get(position_label, 0.0)
    best, best_v = None, -1.0
    xs = [lo + (hi - lo) * k / 5 for k in range(6)] if hi > lo else [lo]
    for xa in xs:
        for ya in (-20, -12, -6, 0, 6, 12, 20):
            if side and ya * side < -6:
                continue
            x, y = xa * att, ya * att
            near_o = min((_d(o, (x, y)) for o in ctx.opp_list), default=30.0)
            near_m = min((_d(q, (x, y)) for i, q in ctx.mates.items() if i != 0), default=30.0)
            openness = 1.0 if near_o > 6 else 0.7 if near_o > 3 else 0.4
            spacing = 1.0 if near_m > 8 else 0.6
            lane = min((_seg(o, ball, (x, y)) for o in ctx.opp_list), default=30.0)
            lane_b = p.get("lane_bonus", 1.25) if lane > 3 else 1.0
            reach = 1.0 if _d(ctx.me, (x, y)) < 22 else 0.8
            v = ctx.V(x, y) * openness * spacing * lane_b * reach
            if v > best_v:
                best, best_v = (x, y), v
    if best is None:
        return None, None
    cmd = {"commandType": "MOVE_TO", "playerId": my_pid, "teamId": team_id, "duration": 0,
           "parameters": {"target_x": round(best[0], 1), "target_y": round(best[1], 1),
                          "sprint": _d(ctx.me, best) > 4}}
    return cmd, {"v": round(best_v, 4)}
