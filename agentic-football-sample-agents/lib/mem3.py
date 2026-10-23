"""mem3: mem2 plus bench orders first, a shoot zone, Nova 2 Lite on field players and DEF/MID positioning."""
from __future__ import annotations

import difflib
import math

import tuning as tuning_mod
import units
import mem31
from state import (_is_my_team, _player_idx, coach_context, get_goal_positions,
                   resolve_holder)

ORDERS = {
    "shoot": ("shoot", "shot", "goal", "gol", "chuta", "tira", "dispara", "remata"),
    "attack": ("attack", "ataca", "ataque", "arriba"),
    "mark": ("mark", "marca", "marcar", "man"),
    "defend": ("defend", "defense", "defiende", "defensa", "atras"),
    "pass": ("pass", "pasa", "pase", "toca"),
    "press": ("press", "presiona", "tackle", "segada", "entra"),
}
_WORDS = {w: k for k, ws in ORDERS.items() for w in ws}
SHOOT_ORDERS = ("shoot", "attack")
BOX_M = 16.5


def on() -> bool:
    return bool(tuning_mod.get("mem3", False))


def _p(key, default):
    return float(tuning_mod.get(key, default))


def parse_order(text: str) -> str | None:
    """Newest message -> canonical order, tolerant to typos. None if unknown."""
    if not text:
        return None
    newest = text.split("|")[-1].strip().lower()
    for word in newest.replace(",", " ").split():
        if word in _WORDS:
            return _WORDS[word]
        hit = difflib.get_close_matches(word, list(_WORDS), 1, 0.6)
        if hit:
            return _WORDS[hit[0]]
    return None


def current_order(game_state: dict, team_id: int) -> str | None:
    text, trusted = coach_context(game_state, team_id)
    return parse_order(text) if text and trusted else None


def _xy(p):
    q = p.get("position") or {}
    return float(q.get("x", 0) or 0), float(q.get("y", 0) or 0)


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    if L == 0:
        return _d(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return _d(p, (ax + t * dx, ay + t * dy))


class _Ctx:
    """The match in metres, from our attacking point of view."""

    def __init__(self, game_state, team_id, pid):
        gs, _ = units.to_field(game_state, force=True)
        self.gs, self.team, self.pid = gs, team_id, pid
        players = gs.get("players") or []
        self.mates = {_player_idx(p): p for p in players if _is_my_team(p, team_id)}
        self.opps = [p for p in players if not _is_my_team(p, team_id)]
        self.me = self.mates.get(pid)
        ball = gs.get("ball") or {}
        b = ball.get("position") or {}
        self.ball = (float(b.get("x", 0) or 0), float(b.get("y", 0) or 0))
        self.holder = resolve_holder(ball, players)
        self.my_goal_x, self.opp_goal_x = get_goal_positions(team_id)
        self.att = 1.0 if self.opp_goal_x > 0 else -1.0
        self.set_piece = str(gs.get("playMode") or "OPEN_PLAY").upper() not in (
            "OPEN_PLAY", "", "0")

    @property
    def i_have(self):
        return self.me is not None and self.holder is self.me

    @property
    def mate_has(self):
        return (self.holder is not None and self.holder is not self.me
                and _is_my_team(self.holder, self.team))

    @property
    def opp_has(self):
        return self.holder is not None and not _is_my_team(self.holder, self.team)

    def d_goal(self, xy=None):
        return _d(xy or _xy(self.me), (self.opp_goal_x, 0.0))

    def cmd(self, ctype, params, duration=0):
        return {"commandType": ctype, "playerId": self.pid, "teamId": self.team,
                "parameters": params, "duration": duration}

    def shoot(self):
        """CENTER at full power, as the rest of the team (lib/tactics: most accurate)."""
        return self.cmd("SHOOT", {"aim_location": "CENTER", "power": 1.0})

    def best_pass(self, min_gain=0.0, lane_m=3.0):
        """Open teammate furthest up the pitch (not the GK), or None."""
        me = _xy(self.me)
        best, best_key = None, None
        for i, p in self.mates.items():
            if i in (0, self.pid):
                continue
            q = _xy(p)
            gain = (q[0] - me[0]) * self.att
            if gain < min_gain or _d(me, q) > 45:
                continue
            lane = min((_seg_dist(_xy(o), me, q) for o in self.opps), default=99.0)
            if lane < lane_m:
                continue
            key = (gain, -_d(me, q))
            if best_key is None or key > best_key:
                best, best_key = (i, q), key
        if best is None:
            return None
        i, q = best
        kind = "THROUGH" if _d(me, q) > 20 else "GROUND"
        return self.cmd("PASS", {"target_player_id": i, "type": kind})

    def press_or_tackle(self):
        if self.holder is None:
            return self.cmd("INTERCEPT", {"aggressive": True}, duration=2)
        hx = _xy(self.holder)
        if _d(_xy(self.me), hx) <= _p("mem3_tackle_m", 3.5):
            return self.cmd("SLIDE_TACKLE", {"target_player_id": -1, "sprint": True})
        return self.cmd("PRESS_BALL", {"intensity": 1.0}, duration=2)

    def presser_id(self):
        """Our outfield player nearest the ball (same answer on every agent)."""
        out = [(i, p) for i, p in self.mates.items() if i != 0]
        if not out:
            return None
        return min(out, key=lambda ip: (_d(_xy(ip[1]), self.ball), ip[0]))[0]

    def man_marks(self, roles):
        """Greedy, deterministic pairing of our players (ids in roles) with
        their outfield players (no keeper, no carrier): {our id: their id}."""
        keeper = min(self.opps, key=lambda o: _d(_xy(o), (self.opp_goal_x, 0.0)), default=None)
        free = [o for o in self.opps if o is not keeper and o is not self.holder]
        pairs = sorted(((_d(_xy(self.mates[i]), _xy(o)), i, _player_idx(o))
                        for i in roles if i in self.mates for o in free))
        out, used = {}, set()
        for _, i, j in pairs:
            if i in out or j in used:
                continue
            out[i] = j
            used.add(j)
        return out


ROLE_IDS = {"DEF": 1, "MID": 2, "FWD1": 3, "FWD2": 4}


def def_hold(ctx: _Ctx, c: dict | None):
    """mem3.3: while the opponent has the ball upfield, the DEF does not go
    past `def_ceiling_m` (attacking frame, -55 = our goal line). A MARK, press
    or MOVE_TO beyond it becomes a MOVE_TO onto the ball-to-our-goal line at
    the ceiling. A DEF already within tackle range of the carrier keeps his
    tackle/press. Off unless tuning has def_ceiling_m. Returns the new
    command, or None when nothing changes."""
    ceil = tuning_mod.get("def_ceiling_m", None)
    if ceil is None or not c or ctx.me is None or not ctx.opp_has or ctx.set_piece:
        return None
    ceil = float(ceil)
    a = ctx.att
    bx = ctx.ball[0] * a
    if bx <= ceil + 6:
        return None
    ct = c.get("commandType")
    prm = c.get("parameters") or {}
    me = _xy(ctx.me)
    if ct in ("SLIDE_TACKLE", "PRESS_BALL") and _d(me, _xy(ctx.holder)) <= 4.0:
        return None
    if ct == "MOVE_TO":
        tx = prm.get("target_x")
        if not isinstance(tx, (int, float)) or tx * a <= ceil:
            return None
    elif ct == "MARK":
        tgt = next((p for i, p in ((_player_idx(o), o) for o in ctx.opps)
                    if i == prm.get("target_player_id")), None)
        if tgt is not None and _xy(tgt)[0] * a <= ceil:
            return None
    elif ct not in ("PRESS_BALL", "SLIDE_TACKLE", "INTERCEPT"):
        return None
    goal = (ctx.my_goal_x, 0.0)
    k = (ceil + 55.0) / max(1.0, bx + 55.0)
    tx = goal[0] + (ctx.ball[0] - goal[0]) * k
    ty = max(-12.0, min(12.0, ctx.ball[1] * k))
    return ctx.cmd("MOVE_TO", {"target_x": round(tx, 1), "target_y": round(ty, 1),
                               "sprint": _d(me, (tx, ty)) > 6})


def coach_command(ctx: _Ctx, order: str, label: str):
    """The bench order turned into this player's command, or None."""
    if ctx.me is None or ctx.pid == 0 or ctx.set_piece:
        return None
    if ctx.i_have:
        if order in SHOOT_ORDERS and mem31.on():
            return mem31.coach_shoot(ctx)
        if order in SHOOT_ORDERS and ctx.d_goal() <= _p("coach_shoot_max_m", 60):
            return ctx.shoot()
        if order == "pass":
            return ctx.best_pass(min_gain=-5.0) or ctx.best_pass(min_gain=-30.0, lane_m=1.5)
        return None
    if ctx.mate_has:
        return None
    if order in ("mark", "defend", "press"):
        presser = ctx.presser_id()
        if ctx.pid == presser:
            return ctx.press_or_tackle()
        if order == "press":
            return None
        roles = [1, 2, 3, 4] if order == "mark" else [1, 2]
        marks = ctx.man_marks([i for i in roles if i != presser])
        j = marks.get(ctx.pid)
        if j is not None:
            return ctx.cmd("MARK", {"target_player_id": j, "tightness": "TIGHT"})
    return None


def decide(game_state, team_id, pid, label):
    """(command, tag) decided in code for mem3, or (None, None) to fall through."""
    if not on():
        return None, None
    try:
        ctx = _Ctx(game_state, team_id, pid)
        if ctx.me is None or pid == 0:
            return None, None
        order = current_order(game_state, team_id)
        if order:
            c = coach_command(ctx, order, label)
            if isinstance(c, tuple):
                return c
            if c is not None:
                if label == "DEF":
                    h = def_hold(ctx, c)
                    if h is not None:
                        return h, "def-hold"
                return c, f"coach-{order}"
        if ctx.i_have and not ctx.set_piece:
            d = ctx.d_goal()
            if d <= BOX_M or d <= _p("shoot_early_m", 28):
                return ctx.shoot(), "shoot-zone"
            if label == "MID" and tuning_mod.get("mid_pass_first", True):
                c = ctx.best_pass(min_gain=_p("mid_pass_gain_m", 6))
                if c is not None:
                    return c, "mid-pass"
        if mem31.on():
            c, tag = mem31.decide(ctx, game_state, team_id, pid, label)
            if c is not None:
                return c, tag
    except Exception:
        return None, None
    return None, None


def shape(commands, game_state, team_id, pid, label):
    """Clamp DEF depth and MID width on MOVE_TO. Returns a tag when changed.
    mem3.1: also every SHOOT goes CENTER (lib/mem31.force_center)."""
    if not on() or not commands:
        return None
    aim = mem31.force_center(commands)
    if aim or label not in ("DEF", "MID"):
        return aim
    c = commands[0]
    if label == "DEF":
        try:
            h = def_hold(_Ctx(game_state, team_id, pid), c)
        except Exception:
            h = None
        if h is not None:
            commands[0] = h
            return "def-hold"
    if c.get("commandType") != "MOVE_TO":
        return None
    prm = c.get("parameters") or {}
    tx, ty = prm.get("target_x"), prm.get("target_y")
    if not isinstance(tx, (int, float)) or not isinstance(ty, (int, float)):
        return None
    try:
        ctx = _Ctx(game_state, team_id, pid)
        if ctx.me is None or ctx.i_have:
            return None
        tag = None
        if label == "DEF":
            floor = _p("def_floor_m", -26)
            ball_att = ctx.ball[0] * ctx.att
            if ball_att > floor + 6 and tx * ctx.att < floor:
                prm["target_x"] = round(floor * ctx.att, 1)
                tag = "def-up"
        if label == "MID":
            lim = _p("mid_max_y", 7)
            if abs(ty) > lim:
                prm["target_y"] = round(math.copysign(lim, ty), 1)
                tag = "mid-centre"
        return tag
    except Exception:
        return None
