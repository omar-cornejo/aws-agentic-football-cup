"""Teams for the local simulator."""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_REGION"] = "us-east-1"
sys.path.insert(0, str(BASE / "lib"))

from test_helpers import mock_agentcore  # noqa: E402

mock_agentcore()
import test_helpers  # noqa: E402

test_helpers._FakeApp.logger.info = staticmethod(lambda msg: None)

from overrides import apply_overrides  # noqa: E402
from tuning import apply_tuning  # noqa: E402

POS = [("ai-gk", 0, "GK"), ("ai-def", 1, "DEF"), ("ai-mid", 2, "MID"),
       ("ai-fwd1", 3, "FWD1"), ("ai-fwd2", 4, "FWD2")]


def _load(team_dir: str, agent_dir: str):
    path = BASE / team_dir / agent_dir / "src" / "main.py"
    name = f"sim_{team_dir}_{agent_dir}".replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def repo_team(team_dir: str, use_overrides: bool = True, extra: dict | None = None,
              fast_ev: bool = True, critic=None):
    """extra: tuning overlay on top of lib/tuning.json, same shape
    ({"global": {...}, "DEF": {...}}). critic(gs, team_id, pid, label, command, tag)
    is called for every decision."""
    import copy
    import tuning as tuning_mod
    data = copy.deepcopy(tuning_mod._DATA)
    for sect, vals in (extra or {}).items():
        data.setdefault(sect, {}).update(vals)
        if sect == "global":
            for pos in ("GK", "DEF", "MID", "FWD1", "FWD2"):
                for k in vals:
                    data.get(pos, {}).pop(k, None)
    mods, cfgs = {}, {}
    for agent_dir, pid, label in POS:
        mod = _load(team_dir, agent_dir)
        mods[pid] = mod
        cfg = getattr(mod, "OVERRIDE_CONFIG", None) if use_overrides else None
        cfgs[pid] = apply_tuning(cfg, label, data)

    def policy(gs, team_id, pid):
        cmds = mods[pid].fallback_commands(gs, team_id, pid)
        tag = None
        if cfgs[pid] is not None:
            cmds, tag = apply_overrides(cmds, gs, team_id, pid, POS[pid][2], cfgs[pid])
        c = dict(cmds[0]) if cmds else {}
        if tag in ("ev", "dsearch", "gk-ev", "gk-angle", "ev-support") and fast_ev:
            c["_latency"] = 0.05
        if critic is not None:
            critic(gs, team_id, pid, POS[pid][2], c, tag)
        return c
    return policy


class View:
    """Convenience view of a gameState from one player's side."""

    def __init__(self, gs, team_id, pid):
        code = "home" if team_id == 0 else "away"
        self.pid = pid
        self.mine, self.opps = {}, {}
        for p in gs["players"]:
            idx = int(p["agentId"].rsplit("_", 1)[-1])
            pos = (p["position"]["x"], p["position"]["y"])
            (self.mine if p["teamCode"] == code else self.opps)[idx] = pos
        b = gs["ball"]
        self.ball = (b["position"]["x"], b["position"]["y"])
        self.me = self.mine[pid]
        self.own_x = -55.0 if team_id == 0 else 55.0
        self.opp_x = -self.own_x
        self.fwd = 1.0 if team_id == 0 else -1.0
        self.holder_team = None
        owner = b.get("possessionAgentId")
        if owner is not None:
            idx = int(owner.rsplit("_", 1)[-1])
            cands = [("me", self.mine.get(idx)), ("opp", self.opps.get(idx))]
            cands = [c for c in cands if c[1] is not None]
            side, _ = min(cands, key=lambda c: d(c[1], self.ball))
            self.holder_team = side
            self.holder_idx = idx
        self.i_have = self.holder_team == "me" and self.holder_idx == pid
        self.score_diff = (gs["score"]["home"] - gs["score"]["away"]) * (1 if team_id == 0 else -1)
        self.t = gs["gameTime"]

    def goal_dist(self, pos=None):
        pos = pos or self.me
        return d(pos, (self.opp_x, 0))

    def nearest_mate_to_ball(self, outfield=True):
        c = {i: p for i, p in self.mine.items() if not (outfield and i == 0)}
        return min(c, key=lambda i: d(c[i], self.ball))

    def lane_clear(self, a, b, r=3.0):
        return all(_seg(o, a, b) > r for o in self.opps.values())

    def best_forward_pass(self, min_gain=6.0, r=3.0):
        best = None
        for i, p in self.mine.items():
            if i in (self.pid, 0):
                continue
            gain = self.goal_dist() - self.goal_dist(p)
            if gain >= min_gain and self.lane_clear(self.me, p, r):
                if best is None or self.goal_dist(p) < self.goal_dist(self.mine[best]):
                    best = i
        return best

    def most_advanced_mate(self):
        c = [i for i in self.mine if i not in (self.pid, 0)]
        return min(c, key=lambda i: self.goal_dist(self.mine[i]))

    def threat_nearest_to(self, pos):
        return min(self.opps, key=lambda i: d(self.opps[i], pos))


def d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return d(p, (ax + t * dx, ay + t * dy))


def cmd(t, **params):
    return {"commandType": t, "parameters": params, "duration": 0}


def move(x, y, sprint=False):
    return cmd("MOVE_TO", target_x=round(x, 1), target_y=round(y, 1), sprint=sprint)


def aim_for(v: View):
    """Aim at the corner furthest from the opponent keeper."""
    gk = v.opps.get(0, (v.opp_x, 0))
    return "TL" if (gk[1] * v.fwd) < 0 else "TR"


def gk_policy(v: View, line=0.95, sweep=False):
    if v.i_have:
        target = v.best_forward_pass(min_gain=10) or v.most_advanced_mate()
        return cmd("GK_DISTRIBUTE", target_player_id=target, method="KICK")
    if v.holder_team is None and d(v.ball, (v.own_x, 0)) < 18 and d(v.ball, v.me) < 9:
        return cmd("INTERCEPT", aggressive=True)
    x = v.own_x * (0.75 if sweep and abs(v.ball[0] - v.own_x) > 60 else line)
    return move(x, max(-5, min(5, v.ball[1] * 0.25)))


def defend(v: View, block_x_factor: float, pressers: int = 1):
    """Generic out-of-possession behaviour."""
    order = sorted((i for i in v.mine if i != 0), key=lambda i: d(v.mine[i], v.ball))
    if v.pid in order[:pressers]:
        if v.holder_team == "opp" and d(v.me, v.ball) < 3:
            return cmd("SLIDE_TACKLE", target_player_id=v.holder_idx)
        return cmd("PRESS_BALL", intensity=1.0) if v.holder_team == "opp" else cmd("INTERCEPT", aggressive=True)
    bx = v.own_x * block_x_factor
    by = {1: 0, 2: v.ball[1] * 0.3, 3: -10, 4: 10}[v.pid]
    tgt = v.threat_nearest_to((bx, by))
    if d(v.opps[tgt], (bx, by)) < 14:
        return cmd("MARK", target_player_id=tgt, tightness="TIGHT")
    return move(bx, by, sprint=d(v.me, (bx, by)) > 10)


def counter_bot(gs, team_id, pid):
    """Deep block, then one fast through ball to forwards who stay high."""
    v = View(gs, team_id, pid)
    if pid == 0:
        return gk_policy(v)
    if v.i_have:
        if v.goal_dist() < 35:
            return cmd("SHOOT", aim_location=aim_for(v), power=1.0)
        if pid in (1, 2):
            fwd = v.most_advanced_mate()
            long = d(v.me, v.mine[fwd]) > 35
            return cmd("PASS", target_player_id=fwd, type="AERIAL" if long else "THROUGH")
        return move(v.opp_x * 0.8, v.me[1] * 0.5, sprint=True)
    if v.holder_team == "me":
        if pid in (3, 4):
            return move(v.opp_x * 0.7, -10 if pid == 3 else 10, sprint=True)
        return move(v.own_x * 0.35, 0)
    if pid in (3, 4):
        return move(v.opp_x * 0.3, -12 if pid == 3 else 12)
    return defend(v, 0.6 if pid == 1 else 0.4, pressers=1 if abs(v.ball[0] - v.own_x) < 50 else 0)


def press_bot(gs, team_id, pid):
    """Gegenpress: two players hunt the ball everywhere, shoot early."""
    v = View(gs, team_id, pid)
    if pid == 0:
        return gk_policy(v, sweep=True)
    if v.i_have:
        if v.goal_dist() < 38:
            return cmd("SHOOT", aim_location=aim_for(v), power=1.0)
        tgt = v.best_forward_pass(min_gain=8)
        if tgt is not None:
            return cmd("PASS", target_player_id=tgt, type="THROUGH")
        return move(v.opp_x * 0.85, v.me[1], sprint=True)
    if v.holder_team == "me":
        spots = {1: (0.0, 0), 2: (0.45, 0), 3: (0.75, -10), 4: (0.75, 10)}
        fx, y = spots[pid]
        return move(v.opp_x * fx, y, sprint=True)
    return defend(v, 0.0 if pid != 1 else 0.3, pressers=2)


def possession_bot(gs, team_id, pid):
    """Short ground passes forward, shoot only from close with a clear lane."""
    v = View(gs, team_id, pid)
    if pid == 0:
        return gk_policy(v)
    if v.i_have:
        if v.goal_dist() < 30:
            return cmd("SHOOT", aim_location=aim_for(v), power=1.0)
        tgt = v.best_forward_pass(min_gain=4, r=4.0)
        if tgt is not None and d(v.me, v.mine[tgt]) < 30:
            return cmd("PASS", target_player_id=tgt, type="GROUND")
        return move(v.me[0] + 12 * v.fwd, v.me[1] * 0.8, sprint=True)
    if v.holder_team == "me":
        spots = {1: (-0.3, 0), 2: (0.3, -6), 3: (0.6, -12), 4: (0.6, 12)}
        fx, y = spots[pid]
        anchor = v.ball[0] + fx * 30 * v.fwd
        return move(max(-50, min(50, anchor)), y)
    return defend(v, 0.4 if pid != 1 else 0.6, pressers=1)


def bus_bot(gs, team_id, pid):
    """Park the bus: four players near the box, clear long to a lone striker."""
    v = View(gs, team_id, pid)
    if pid == 0:
        return gk_policy(v)
    if v.i_have:
        if v.goal_dist() < 30:
            return cmd("SHOOT", aim_location=aim_for(v), power=1.0)
        if pid == 3:
            return move(v.opp_x * 0.8, 0, sprint=True)
        return cmd("PASS", target_player_id=3, type="AERIAL")
    if pid == 3:
        return move(v.opp_x * 0.05, 0) if v.holder_team != "me" else move(v.opp_x * 0.7, 0, sprint=True)
    return defend(v, 0.72, pressers=1 if d(v.ball, (v.own_x, 0)) < 30 else 0)


def markov_bot(gs, team_id, pid):
    """Only the Markov/EV model, no hand-written tactics: the carrier plays the
    best EV option, teammates run to the open zone with the highest V,
    defenders press (nearest) or mark."""
    import ev_policy
    v = View(gs, team_id, pid)
    if pid == 0:
        return gk_policy(v)
    if v.i_have:
        c, _ = ev_policy.choose(gs, team_id, pid, model=ev_policy.get_model(RIVAL_MODEL))
        return c or move(v.opp_x, 0, True)
    if v.holder_team == "me":
        m = ev_policy.get_model(RIVAL_MODEL)
        best, best_val = v.me, -1.0
        for dx in (-6, 0, 6, 12):
            for dy in (-10, -5, 0, 5, 10):
                x = max(-52, min(52, v.me[0] + dx * v.fwd))
                y = max(-27, min(27, v.me[1] + dy))
                openness = 1.0 if min(d(o, (x, y)) for o in v.opps.values()) > 5 else 0.5
                mates = min((d(q, (x, y)) for i, q in v.mine.items() if i != pid), default=99)
                val = m["xt"][ev_policy.cell(x, y, v.fwd)] * openness * (1.0 if mates > 6 else 0.6)
                if val > best_val:
                    best, best_val = (x, y), val
        return move(best[0], best[1], sprint=True)
    return defend(v, 0.45 if pid != 1 else 0.65, pressers=1)


def _ev_carrier(bot):
    """Wrap a style bot so its ball carrier plays the EV/Markov choice."""
    def policy(gs, team_id, pid):
        import ev_policy
        v = View(gs, team_id, pid)
        if v.i_have and pid != 0:
            c, _ = ev_policy.choose(gs, team_id, pid, model=ev_policy.get_model(RIVAL_MODEL))
            if c:
                return c
        return bot(gs, team_id, pid)
    return policy


def markov_search_bot(gs, team_id, pid):
    """Markov attack + searched defence, no hand-written tactics."""
    import def_search
    if pid != 0:
        import ev_policy
        c, _ = def_search.assign(gs, team_id, pid, budget=0.015,
                                 model=ev_policy.get_model(RIVAL_MODEL))
        if c:
            return c
    return markov_bot(gs, team_id, pid)


AGG = "ai-team-strands-extremely-aggressive"
RIVAL_MODEL = "ev_model_v1.json"
VARIANTS = {
    "nuestro: base": {},
    "nuestro: EV Markov (rápido)": {"global": {"ev_select": True}},
    "nuestro: EV Markov (latencia LLM)": {"global": {"ev_select": True}},
    "nuestro: base + búsqueda defensiva": {"global": {"def_search": True}},
    "nuestro: EV + búsqueda defensiva": {"global": {"ev_select": True, "def_search": True}},
    "nuestro: EV + portero EV": {"global": {"ev_select": True, "gk_ev": True}},
    "nuestro: EV + búsqueda + portero": {"global": {"ev_select": True, "def_search": True,
                                                    "gk_ev": True}},
    "nuestro: v1 + apoyo FWD": {"global": {"ev_select": True, "def_search": True, "gk_ev": True,
                                           "ev_support": True, "ev_support_roles": "FWD1,FWD2"}},
}

TEAMS = {name: (lambda extra=extra, name=name: repo_team(AGG, True, extra, fast_ev="latencia" not in name))
         for name, extra in VARIANTS.items()}
TEAMS.update({
    "agresivo sin overrides": lambda: repo_team(AGG, False),
    "contraataque": lambda: counter_bot,
    "presión alta": lambda: press_bot,
    "posesión": lambda: possession_bot,
    "autobús": lambda: bus_bot,
    "solo Markov": lambda: markov_bot,
    "presión + EV": lambda: _ev_carrier(press_bot),
    "contraataque + EV": lambda: _ev_carrier(counter_bot),
    "Markov + búsqueda": lambda: markov_search_bot,
    "espejo v1": lambda: repo_team(AGG, True, {"global": {"ev_select": True, "def_search": True,
                                                          "gk_ev": True,
                                                          "ev_model": RIVAL_MODEL}}),
})

TRAIN_RIVALS = ["agresivo sin overrides", "presión alta", "posesión", "presión + EV"]
TEST_RIVALS = ["contraataque", "autobús", "solo Markov", "contraataque + EV",
               "Markov + búsqueda", "espejo v1"]
TRAIN_SEEDS = 0
TEST_SEEDS = 500_000
OURS = list(VARIANTS)
RIVALS = TRAIN_RIVALS + TEST_RIVALS
