"""Approximate 5v5 match engine for local testing (no AWS, no LLM)."""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import ev_policy as evp  # noqa: E402

HALF_X, HALF_Y = 55.0, 30.0
GOAL_HALF = 7.0
DT = 0.1
TICK_S = 2.0
LATENCY_S = 0.6
FRICTION = 0.95
SHOT_FRICTION = 0.97
JOG, SPRINT = 4.5, 7.0
GK_REFLEX_SPEED = 6.5
DRIBBLE_FACTOR = 0.85

FORMATION = {0: (-52, 0), 1: (-30, 0), 2: (-14, 0), 3: (-4, -12), 4: (-4, 12)}
LABELS = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _d(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


@dataclass
class Player:
    team: int
    idx: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    stamina: float = 1.0
    sprinting: bool = False
    stunned: float = 0.0
    cmd: dict = field(default_factory=dict)
    cmd_used: bool = False
    st: dict = field(default_factory=lambda: dict(
        passes=0, passes_ok=0, shots=0, goals=0, steals=0, losses=0, xt=0.0, ball_ticks=0))
    heat: list = field(default_factory=lambda: [0] * (12 * 6))

    @property
    def goal_x(self):
        return -HALF_X if self.team == 0 else HALF_X


@dataclass
class TeamStats:
    goals: int = 0
    shots: int = 0
    on_target: int = 0
    passes: int = 0
    passes_ok: int = 0
    steals: int = 0
    possession_s: float = 0.0
    territory_s: float = 0.0
    final_third_s: float = 0.0


class Match:
    def __init__(self, policy_home, policy_away, duration=240.0, seed=0, log=None):
        self.rng = random.Random(seed)
        self.policies = (policy_home, policy_away)
        self.duration = duration
        self.t = 0.0
        self.tick = 0
        self.players = [Player(t, i, 0, 0) for t in (0, 1) for i in range(5)]
        self.stats = (TeamStats(), TeamStats())
        self.bx = self.by = self.bvx = self.bvy = 0.0
        self.holder: Player | None = None
        self.immune = 0.0
        self.flight = None
        self.last_touch = 0
        self.pending = []
        self.log = log
        self.open_shot = self.open_pass = None
        self.tick_rec = None
        self.pass_from = self.shot_from = None
        self.xt_prev = None
        self.goal_by = None
        self._kickoff(0)

    def _kickoff(self, team_with_ball):
        for p in self.players:
            fx, fy = FORMATION[p.idx]
            p.x, p.y = (fx, fy) if p.team == 0 else (-fx, -fy)
            p.vx = p.vy = 0.0
            p.stunned = 0.0
            p.cmd, p.cmd_used = {}, False
        mid = self.players[team_with_ball * 5 + 2]
        mid.x, mid.y = (-1.0 if team_with_ball == 0 else 1.0), 0.0
        self._give(mid)
        self.pending = []

    def _give(self, p: Player):
        if self.pass_from is not None:
            if p.team == self.pass_from.team:
                self.pass_from.st["passes_ok"] += 1
            else:
                self.pass_from.st["losses"] += 1
            self.pass_from = None
        self.shot_from = None
        if self.open_pass is not None:
            self.open_pass["y"] = int(p.team == self.open_pass_team)
            self.open_pass = None
        if self.open_shot is not None:
            self.open_shot["y"] = 0
            self.open_shot = None
        self.holder, self.flight = p, None
        self.bx, self.by, self.bvx, self.bvy = p.x, p.y, 0.0, 0.0
        self.last_touch = p.team
        self.immune = 0.7

    def game_state(self):
        return {
            "tick": self.tick, "gameTime": self.t, "playMode": "OPEN_PLAY",
            "score": {"home": self.stats[0].goals, "away": self.stats[1].goals},
            "ball": {
                "position": {"x": self.bx, "y": self.by, "z": 0},
                "velocity": {"x": self.bvx, "y": self.bvy, "z": 0},
                "isFree": self.holder is None,
                "possessionAgentId": (f"agentId_{self.holder.idx}" if self.holder else None),
            },
            "players": [{
                "agentId": f"agentId_{p.idx}", "teamCode": "home" if p.team == 0 else "away",
                "position": {"x": p.x, "y": p.y}, "velocity": {"x": p.vx, "y": p.vy},
                "stamina": p.stamina, "isSprinting": p.sprinting,
            } for p in self.players],
            "teamChat": [],
        }

    def run(self):
        next_tick = 0.0
        while self.t < self.duration:
            if self.t >= next_tick - 1e-9:
                self._decide()
                next_tick += TICK_S
            while self.pending and self.pending[0][0] <= self.t + 1e-9:
                _, batch = self.pending.pop(0)
                for p, cmd in batch:
                    p.cmd, p.cmd_used = cmd, False
            self._step()
            self.t += DT
        return self.stats

    def _decide(self):
        self._xt_credit()
        if self.log is not None:
            self._tick_log()
        gs = self.game_state()
        by_time = {}
        for p in self.players:
            try:
                cmd = self.policies[p.team](gs, p.team, p.idx) or {}
            except Exception as e:
                cmd = {"commandType": "SET_STANCE", "parameters": {}, "_error": str(e)}
            lat = cmd.get("_latency", LATENCY_S)
            by_time.setdefault(round(self.t + lat, 3), []).append((p, cmd))
        self.pending = sorted(self.pending + list(by_time.items()), key=lambda e: e[0])
        self.tick += 1

    def _V(self, p):
        m = evp.MODEL
        return m["xt"][evp.cell(p.x, p.y, 1.0 if p.team == 0 else -1.0)] if m else 0.0

    def _xt_credit(self):
        """Credit the carrier at the start of each tick with the change in
        threat (V) until the next tick: + for progress, - for losing it."""
        prev = self.xt_prev
        if prev is not None:
            p, v0 = prev
            h = self.holder
            if self.goal_by == p.team:
                p.st["xt"] += 1.0 - v0
            elif self.goal_by is not None:
                p.st["xt"] -= v0
            elif h is not None and h.team == p.team:
                p.st["xt"] += self._V(h) - v0
            elif h is not None:
                p.st["xt"] -= v0
        self.goal_by = None
        for p in self.players:
            p.heat[evp.cell(p.x, p.y, 1.0 if p.team == 0 else -1.0)] += 1
        h = self.holder
        self.xt_prev = (h, self._V(h)) if h is not None else None
        if h is not None:
            h.st["ball_ticks"] += 1

    def _opps_xy(self, team):
        return [(o.x, o.y) for o in self.players if o.team != team]

    def _tick_log(self):
        """Close the previous tick's samples and open new ones."""
        r = self.tick_rec
        if r is not None:
            h = self.holder
            same_team = h is not None and h.team == r["team"]
            nxt = evp.cell(h.x, h.y, 1.0 if h.team == 0 else -1.0) if same_team else None
            self.log["xt"].append((r["cell"], int(r["shot"]), int(r["goal"]), nxt))
            if not r["shot"] and not r["passed"]:
                r["keep"]["y"] = int(same_team)
                self.log["keep"].append(r["keep"])
        h = self.holder
        if h is None:
            self.tick_rec = None
            return
        self.tick_rec = {"team": h.team, "cell": evp.cell(h.x, h.y, 1.0 if h.team == 0 else -1.0),
                         "keep": {"v1": evp.f_keep((h.x, h.y), self._opps_xy(h.team)),
                                  "v2": evp.f_keep2((h.x, h.y), self._opps_xy(h.team),
                                                    [(o.vx, o.vy) for o in self.players
                                                     if o.team != h.team])},
                         "shot": False, "goal": False, "passed": False}

    def _step(self):
        if self.holder:
            self.stats[self.holder.team].possession_s += DT
        if self.bx > 0:
            self.stats[0].territory_s += DT
            if self.bx > HALF_X / 3:
                self.stats[0].final_third_s += DT
        elif self.bx < 0:
            self.stats[1].territory_s += DT
            if self.bx < -HALF_X / 3:
                self.stats[1].final_third_s += DT
        self.immune = max(0.0, self.immune - DT)
        for p in self.players:
            p.stunned = max(0.0, p.stunned - DT)
            self._act(p)
        self._gk_reflex()
        self._move_players()
        self._move_ball()
        self._contests()
        self._boundaries()

    def _opp(self, p: Player, idx):
        return self.players[(1 - p.team) * 5 + int(idx) % 5]

    def _mate(self, p: Player, idx):
        return self.players[p.team * 5 + int(idx) % 5]

    def _act(self, p: Player):
        """Turn the player's current command into a movement target / ball action."""
        p.tx, p.ty, p.run = p.x, p.y, False
        if p.stunned > 0:
            return
        f = self.flight
        if f and f["kind"] == "pass" and f["to"] is p:
            p.tx, p.ty, p.run = f["tx"], f["ty"], True
            return
        c = p.cmd or {}
        ct = c.get("commandType")
        prm = c.get("parameters") or {}
        has = self.holder is p

        if ct in ("PASS", "GK_DISTRIBUTE") and has and not p.cmd_used:
            p.cmd_used = True
            tgt = self._mate(p, prm.get("target_player_id", 3))
            if tgt is not p:
                kind = prm.get("type") or ("AERIAL" if prm.get("method") == "KICK" else "GROUND")
                self._pass(p, tgt, kind)
            return
        if ct == "SHOOT" and has and not p.cmd_used:
            p.cmd_used = True
            self._shoot(p, prm.get("aim_location", "CENTER"), prm.get("power", 1.0))
            return
        if ct == "MOVE_TO":
            tx, ty = prm.get("target_x"), prm.get("target_y")
            if isinstance(tx, (int, float)) and isinstance(ty, (int, float)):
                p.tx, p.ty, p.run = tx, ty, bool(prm.get("sprint"))
        elif ct in ("PRESS_BALL", "INTERCEPT"):
            lead = 0.4 if ct == "INTERCEPT" else 0.0
            p.tx, p.ty = self.bx + self.bvx * lead, self.by + self.bvy * lead
            p.run = (prm.get("intensity", 1.0) or 0) > 0.6 or ct == "INTERCEPT"
        elif ct in ("MARK", "FOLLOW_PLAYER"):
            o = self._opp(p, prm.get("target_player_id", 0))
            gap = 2.0 if prm.get("tightness", "TIGHT") == "TIGHT" else 4.0
            gx = p.goal_x
            d = _d(o.x, o.y, gx, 0) or 1.0
            p.tx, p.ty = o.x + (gx - o.x) / d * gap, o.y + (0 - o.y) / d * gap
            p.run = _d(p.x, p.y, p.tx, p.ty) > 6
        elif ct == "SLIDE_TACKLE":
            h = self.holder
            if h is not None and h.team != p.team:
                p.tx, p.ty, p.run = h.x, h.y, True
                if not p.cmd_used and _d(p.x, p.y, h.x, h.y) <= 2.6 and self.immune <= 0:
                    p.cmd_used = True
                    if self.rng.random() < 0.45:
                        self._steal(p)
                    else:
                        p.stunned = 1.2

    def _gk_reflex(self):
        """Keepers react to a shot on their goal regardless of their command."""
        f = self.flight
        if not f or f["kind"] != "shot":
            return
        gk = self.players[(1 - f["team"]) * 5]
        gx = gk.goal_x
        if (gx - self.bx) * self.bvx <= 0 or gk.stunned > 0:
            return
        tline = (gx - self.bx) / self.bvx if self.bvx else 99
        yline = _clamp(self.by + self.bvy * tline, -GOAL_HALF - 1, GOAL_HALF + 1)
        gk.tx, gk.ty, gk.run = gx - math.copysign(1.0, gx), yline, True
        gk.reflex = True

    def _move_players(self):
        for p in self.players:
            if p.stunned > 0:
                p.vx = p.vy = 0.0
                continue
            dx, dy = p.tx - p.x, p.ty - p.y
            dist = math.hypot(dx, dy)
            fast = p.run and p.stamina > 0.05
            speed = (GK_REFLEX_SPEED if getattr(p, "reflex", False) else (SPRINT if fast else JOG))
            speed *= 0.7 + 0.3 * p.stamina
            if self.holder is p:
                speed *= DRIBBLE_FACTOR
            step = min(dist, speed * DT)
            if dist > 1e-6:
                p.vx, p.vy = dx / dist * step / DT, dy / dist * step / DT
                p.x = _clamp(p.x + dx / dist * step, -HALF_X - 2, HALF_X + 2)
                p.y = _clamp(p.y + dy / dist * step, -HALF_Y - 1, HALF_Y + 1)
            else:
                p.vx = p.vy = 0.0
            p.sprinting = fast and dist > 0.5
            p.stamina = _clamp(p.stamina + (-0.012 if p.sprinting else 0.006) * DT, 0, 1)
            p.reflex = False

    def _move_ball(self):
        if self.holder:
            self.bx, self.by = self.holder.x, self.holder.y
            self.bvx, self.bvy = self.holder.vx, self.holder.vy
            return
        self.bx += self.bvx * DT
        self.by += self.bvy * DT
        fr = SHOT_FRICTION if (self.flight and self.flight["kind"] == "shot") else FRICTION
        self.bvx *= fr
        self.bvy *= fr
        if self.flight:
            self.flight["travel"] += math.hypot(self.bvx, self.bvy) * DT / fr

    def _release(self, p, tx, ty, err_deg, extra=1.1):
        d = _d(p.x, p.y, tx, ty)
        ang = math.atan2(ty - p.y, tx - p.x) + math.radians(self.rng.gauss(0, err_deg))
        v0 = d * (1 - FRICTION) / DT * extra
        self.holder = None
        self.bvx, self.bvy = math.cos(ang) * v0, math.sin(ang) * v0
        self.last_touch = p.team
        return d

    def _pass(self, p, tgt, kind):
        st = self.stats[p.team]
        st.passes += 1
        p.st["passes"] += 1
        self.pass_from = p
        opp_gx = -p.goal_x
        tx, ty = tgt.x, tgt.y
        if kind == "THROUGH":
            tx += math.copysign(6.0, opp_gx)
        tx, ty = _clamp(tx, -HALF_X + 1, HALF_X - 1), _clamp(ty, -HALF_Y + 1, HALF_Y - 1)
        aerial = kind == "AERIAL"
        if self.log is not None:
            rec = {"v1": evp.f_pass((p.x, p.y), (tx, ty), self._opps_xy(p.team)),
                   "v2": evp.f_pass2((p.x, p.y), (tx, ty), self._opps_xy(p.team)), "y": None}
            self.log["pass"].append(rec)
            self.open_pass, self.open_pass_team = rec, p.team
            if self.tick_rec and self.tick_rec["team"] == p.team:
                self.tick_rec["passed"] = True
        d = self._release(p, tx, ty, 5.0 if aerial else 3.0)
        self.flight = {"kind": "pass", "team": p.team, "from": p, "to": tgt,
                       "aerial": aerial, "plan": d, "travel": 0.0, "tx": tx, "ty": ty,
                       "tried": set()}

    def _shoot(self, p, aim, power):
        st = self.stats[p.team]
        st.shots += 1
        p.st["shots"] += 1
        gx = -p.goal_x
        if self.log is not None:
            gk = self.players[(1 - p.team) * 5]
            opps = self._opps_xy(p.team)
            rec = {"v1": evp.f_shot((p.x, p.y), gx, opps, (gk.x, gk.y)),
                   "v2": evp.f_shot2((p.x, p.y), gx, opps, (gk.x, gk.y)), "y": None}
            self.log["shot"].append(rec)
            if self.tick_rec and self.tick_rec["team"] == p.team:
                self.tick_rec["shot"] = True
        side = 1.0 if p.team == 0 else -1.0
        aim = str(aim or "CENTER").upper()
        off = 0.0 if aim == "CENTER" else (0.7 * GOAL_HALF * (side if "L" in aim else -side))
        d = _d(p.x, p.y, gx, off)
        power = _clamp(float(power or 1.0), 0.3, 1.0)
        err = 1.5 + 0.12 * d
        ang = math.atan2(off - p.y, gx - p.x) + math.radians(self.rng.gauss(0, err))
        v0 = 28.0 * power
        self.holder = None
        self.bvx, self.bvy = math.cos(ang) * v0, math.sin(ang) * v0
        self.last_touch = p.team
        t_line = (gx - p.x) / self.bvx if self.bvx else 99
        y_line = p.y + self.bvy * t_line
        reach = v0 * SHOT_FRICTION / (1 - SHOT_FRICTION) * DT
        if t_line > 0 and abs(y_line) <= GOAL_HALF and reach >= abs(gx - p.x):
            st.on_target += 1
        self.flight = {"kind": "shot", "team": p.team, "from": p, "travel": 0.0, "tried": set()}
        self.shot_from = p
        if self.log is not None:
            self.open_shot = rec

    def _steal(self, p):
        self.stats[p.team].steals += 1
        p.st["steals"] += 1
        old = self.holder
        if old is not None:
            old.st["losses"] += 1
        self._give(p)
        if old:
            old.stunned = 0.4

    def _contests(self):
        h = self.holder
        if h is not None:
            if self.immune > 0:
                return
            for o in self.players:
                if o.team == h.team or o.stunned > 0:
                    continue
                if _d(o.x, o.y, h.x, h.y) > 1.2:
                    continue
                ct = (o.cmd or {}).get("commandType")
                inten = (o.cmd.get("parameters") or {}).get("intensity", 1.0) if o.cmd else 1.0
                p = 0.012 + (0.04 * (inten or 1.0) if ct == "PRESS_BALL" else 0.0)
                if o.idx == 0 and abs(h.x - o.goal_x) < 16:
                    p += 0.05
                if self.rng.random() < p:
                    self._steal(o)
                    return
            return

        speed = math.hypot(self.bvx, self.bvy)
        f = self.flight or {}
        cands = []
        for p in self.players:
            if p.stunned > 0:
                continue
            if f.get("from") is p and f.get("travel", 9) < 3:
                continue
            if f.get("kind") == "shot" and p.team == f.get("team"):
                continue
            intended = f.get("to") is p
            is_gk = p.idx == 0 and abs(self.bx - p.goal_x) < 18
            r = 2.4 if is_gk else (2.0 if intended else 1.3)
            d = _d(p.x, p.y, self.bx, self.by)
            if d > r:
                continue
            if f.get("aerial") and f["travel"] < 0.65 * f["plan"] and not is_gk:
                continue
            if f and speed >= 4.0:
                key = (p.team, p.idx)
                if key in f["tried"]:
                    continue
                f["tried"].add(key)
            if is_gk:
                reaction = min(0.25, f.get("travel", 0.0) / 150) if f else 0.0
                prob = _clamp(1.35 - speed / 22 + reaction, 0.15, 0.97)
            else:
                prob = _clamp(1.15 - speed / 20, 0.1, 0.95)
                if intended:
                    prob = max(prob, 0.85)
            cands.append((d, prob, p))
        for _, prob, p in sorted(cands, key=lambda c: c[0]):
            if self.rng.random() < prob:
                if f.get("kind") == "pass":
                    if p.team == f["team"]:
                        self.stats[p.team].passes_ok += 1
                    else:
                        self.stats[p.team].steals += 1
                        p.st["steals"] += 1
                self._give(p)
                self.immune = 0.4
                return
        if self.flight and speed < 2.0:
            self.flight = None

    def _boundaries(self):
        if abs(self.bx) > HALF_X:
            scorer = 0 if self.bx > 0 else 1
            if abs(self.by) <= GOAL_HALF:
                self.stats[scorer].goals += 1
                self.goal_by = scorer
                if self.shot_from is not None and self.shot_from.team == scorer:
                    self.shot_from.st["goals"] += 1
                self.shot_from = None
                if self.open_shot is not None:
                    self.open_shot["y"] = 1
                    self.open_shot = None
                if self.tick_rec and self.tick_rec["team"] == scorer:
                    self.tick_rec["goal"] = True
                self._kickoff(1 - scorer)
                return
            defending = 1 - scorer
            if self.last_touch == scorer:
                gk = self.players[defending * 5]
                gk.x, gk.y = gk.goal_x - math.copysign(4.0, gk.goal_x), 0.0
                self._give(gk)
            else:
                att = min((p for p in self.players if p.team == scorer and p.idx != 0),
                          key=lambda p: _d(p.x, p.y, self.bx, self.by))
                att.x = math.copysign(HALF_X - 2, self.bx)
                att.y = math.copysign(HALF_Y - 2, self.by)
                self._give(att)
        elif abs(self.by) > HALF_Y:
            team = 1 - self.last_touch
            taker = min((p for p in self.players if p.team == team and p.idx != 0),
                        key=lambda p: _d(p.x, p.y, self.bx, self.by))
            taker.x = _clamp(self.bx, -HALF_X + 1, HALF_X - 1)
            taker.y = math.copysign(HALF_Y - 1, self.by)
            self._give(taker)
