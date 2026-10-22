"""Memory v2: a match notebook in process + AgentCore Memory in the background."""
from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time

import tuning as tuning_mod


def enabled() -> bool:
    return bool(tuning_mod.get("memory_v2", False))


def _remote() -> bool:
    return enabled() and bool(os.environ.get("MEMORY_ID"))


def strategy_name() -> str:
    return (os.environ.get("STRATEGY") or str(tuning_mod.get("strategy", "") or "")
            or "default")


def _safe(s: str) -> str:
    """Memory ids: letters, digits, '-', '_' only, <= 100 chars."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", s)[:100]


def _idx(p):
    from state import _player_idx
    return _player_idx(p)


def _mine(p, team_id):
    from state import _is_my_team
    return _is_my_team(p, team_id)


def _pos(p):
    q = (p or {}).get("position") or {}
    return float(q.get("x", 0) or 0), float(q.get("y", 0) or 0)


def _band(d):
    if d is None:
        return "?"
    return "close" if d <= 25 else "mid" if d <= 45 else "long"


def _side(y, attack_dir):
    """Left/centre/right seen from the team that attacks toward attack_dir."""
    ya = y * attack_dir
    return "LEFT" if ya > 8 else "RIGHT" if ya < -8 else "CENTRE"


def _new_stats():
    return {"pass": [0, 0], "shot": {"close": [0, 0], "mid": [0, 0], "long": [0, 0]},
            "carry": [0, 0], "win": [0, 0], "gf": 0, "ga": 0, "ga_side": {},
            "ticks": 0, "own_ticks": 0, "opp_ticks": 0}


class Notebook:
    """What happened to ONE player in ONE match. Pure Python, no I/O."""

    def __init__(self, match_key: str, team_id: int, pid: int, label: str):
        self.key, self.team, self.pid, self.label = match_key, team_id, pid, label
        self.s = _new_stats()
        self.pending = []
        self.recent = []
        self.goals = []
        self.carriers = {}
        self.opp_side = {"LEFT": 0, "CENTRE": 0, "RIGHT": 0}
        self.last_score = None
        self.last_holder = None
        self.prev_ball = None
        self.lessons = None
        self.lessons_n = 0
        self.prior = None
        self.t = 0.0
        self.my_dg = None

    def observe(self, gs: dict, team_id: int):
        from state import resolve_holder, get_goal_positions
        import units
        gs, _ = units.to_field(gs, force=True)
        self.t = float(gs.get("gameTime", 0) or 0)
        players = gs.get("players") or []
        ball = gs.get("ball") or {}
        holder = resolve_holder(ball, players)
        sc = gs.get("score") or {}
        ours = sc.get("home", 0) if team_id == 0 else sc.get("away", 0)
        theirs = sc.get("away", 0) if team_id == 0 else sc.get("home", 0)
        my_goal_x, _ = get_goal_positions(team_id)
        opp_attack = 1.0 if my_goal_x > 0 else -1.0
        our_attack = -opp_attack
        self.s["ticks"] += 1
        h = None
        if holder is not None:
            h = ("US" if _mine(holder, team_id) else "THEM", _idx(holder))
            self.s["own_ticks" if h[0] == "US" else "opp_ticks"] += 1
            if h[0] == "THEM":
                self.carriers[h[1]] = self.carriers.get(h[1], 0) + 1
                bx, by = _pos(holder)
                if bx * opp_attack > 0:
                    self.opp_side[_side(by, our_attack)] += 1
        if self.last_score is not None:
            if ours > self.last_score[0]:
                self.s["gf"] += 1
                self.goals.append((round(self.t), "US", None, None))
                self._resolve_goal(True)
            if theirs > self.last_score[1]:
                self.s["ga"] += 1
                side = _side(self.prev_ball[1], our_attack) if self.prev_ball else "?"
                self.s["ga_side"][side] = self.s["ga_side"].get(side, 0) + 1
                top = max(self.carriers, key=self.carriers.get) if self.carriers else None
                self.goals.append((round(self.t), "THEM", side, top))
                self._resolve_goal(False)
        self.last_score = (ours, theirs)
        self._resolve(h)
        if h is not None:
            self.last_holder = h
        bp = (ball.get("position") or {})
        self.prev_ball = (float(bp.get("x", 0) or 0), float(bp.get("y", 0) or 0))
        me = next((p for p in players if _idx(p) == self.pid and _mine(p, team_id)), None)
        if me is not None:
            _, opp_goal_x = get_goal_positions(team_id)
            mx, my = _pos(me)
            self.my_dg = round(math.hypot(mx - opp_goal_x, my))

    def _resolve_goal(self, for_us: bool):
        for p in list(self.pending):
            if p[0] == "shot":
                band = p[2]
                if for_us and self.t - p[1] <= 6:
                    self.s["shot"][band][1] += 1
                    self._done(p, "GOAL")
                else:
                    self._done(p, "missed")

    def _resolve(self, h):
        """Turn pending actions into outcomes from who holds the ball now."""
        for p in list(self.pending):
            kind, t0 = p[0], p[1]
            age = self.t - t0
            if kind == "shot":
                if age > 6:
                    self._done(p, "missed")
                continue
            if h is None:
                if age > 6:
                    self.pending.remove(p)
                continue
            if kind == "pass":
                if h[0] == "US" and h[1] != self.pid:
                    self.s["pass"][0] += 1
                    self._done(p, f"reached P{h[1]}")
                elif h[0] == "THEM":
                    self.s["pass"][1] += 1
                    self._done(p, "intercepted")
                elif age > 6:
                    self.pending.remove(p)
            elif kind == "carry" and age >= 1.5:
                if h[0] == "US":
                    self.s["carry"][0] += 1
                    self._done(p, "kept")
                else:
                    self.s["carry"][1] += 1
                    self._done(p, "lost")
            elif kind == "win":
                if h[0] == "US":
                    self.s["win"][0] += 1
                    self._done(p, "won it back")
                elif age > 6:
                    self.s["win"][1] += 1
                    self._done(p, "not won")

    def _done(self, p, outcome):
        if p in self.pending:
            self.pending.remove(p)
        self.recent.append(f"{round(p[1])}s {p[3]} -> {outcome}")
        self.recent = self.recent[-5:]

    def note(self, cmd: dict, has_ball: bool, d_goal):
        ct = (cmd or {}).get("commandType")
        prm = (cmd or {}).get("parameters") or {}
        if any(p[1] == self.t for p in self.pending):
            return
        if has_ball and ct == "SHOOT":
            band = _band(d_goal)
            self.s["shot"][band][0] += 1
            self.pending.append(["shot", self.t, band, f"SHOOT from {d_goal} m"])
        elif has_ball and ct in ("PASS", "GK_DISTRIBUTE"):
            self.pending.append(["pass", self.t, None,
                                 f"{ct} to P{prm.get('target_player_id')}"])
        elif has_ball and ct == "MOVE_TO":
            self.pending.append(["carry", self.t, None, "dribble"])
        elif not has_ball and ct in ("PRESS_BALL", "SLIDE_TACKLE", "INTERCEPT") \
                and self.last_holder and self.last_holder[0] == "THEM":
            self.pending.append(["win", self.t, None, ct])

    def block(self) -> str:
        s = self.s
        lines = [f"MATCH MEMORY (what happened so far, t={self.t:.0f}s):"]
        if self.goals:
            g = "; ".join(f"{t}s {'we scored' if who == 'US' else 'THEY scored'}"
                          + (f" (down our {side} flank, P{top} main carrier)"
                             if who == "THEM" else "")
                          for t, who, side, top in self.goals[-4:])
            lines.append(f"- Goals: {g}.")
        own, opp = s["own_ticks"], s["opp_ticks"]
        if own + opp >= 5:
            lines.append(f"- Team possession {100 * own // max(1, own + opp)} %.")
        acts = []
        if sum(s["pass"]):
            acts.append(f"passes {s['pass'][0]}/{sum(s['pass'])} reached a teammate")
        shots = [(b, v) for b, v in s["shot"].items() if v[0]]
        if shots:
            acts.append("shots " + ", ".join(f"{b} {v[1]} goal(s) in {v[0]}" for b, v in shots))
        if sum(s["carry"]):
            acts.append(f"dribbles kept {s['carry'][0]}/{sum(s['carry'])}")
        if sum(s["win"]):
            acts.append(f"ball won back {s['win'][0]}/{sum(s['win'])} presses")
        if acts:
            lines.append("- You: " + "; ".join(acts) + ".")
        if self.recent:
            lines.append("- Your last actions: " + "; ".join(self.recent[-3:]) + ".")
        if self.carriers:
            top = max(self.carriers, key=self.carriers.get)
            side = max(self.opp_side, key=self.opp_side.get)
            if sum(self.opp_side.values()) >= 4:
                lines.append(f"- Opponent danger: P{top} carries most; they attack "
                             f"mostly down our {side} flank.")
            else:
                lines.append(f"- Opponent danger: P{top} carries most.")
        if self.lessons:
            lines.append(f"LESSONS FROM YOUR LAST {self.lessons_n} MATCHES:")
            lines.extend(f"- {x}" for x in self.lessons)
        return "\n".join(lines) if len(lines) > 1 else ""

    def tally(self) -> str:
        """Compact counters for the DECISION log line."""
        s = self.s
        sh = sum(v[0] for v in s["shot"].values())
        sg = sum(v[1] for v in s["shot"].values())
        return f"p{s['pass'][0]}/{sum(s['pass'])} s{sg}/{sh} c{s['carry'][0]}/{sum(s['carry'])}"


def lessons_from(snapshots: list[dict]) -> list[str]:
    """Turn totals of previous matches into a few actionable lines."""
    if not snapshots:
        return []
    tot = _new_stats()
    res = {"W": 0, "D": 0, "L": 0}
    for sn in snapshots:
        st = sn.get("s") or {}
        for k in ("pass", "carry", "win"):
            for i in (0, 1):
                tot[k][i] += (st.get(k) or [0, 0])[i]
        for b in ("close", "mid", "long"):
            for i in (0, 1):
                tot["shot"][b][i] += ((st.get("shot") or {}).get(b) or [0, 0])[i]
        tot["gf"] += st.get("gf", 0)
        tot["ga"] += st.get("ga", 0)
        for side, n in (st.get("ga_side") or {}).items():
            tot["ga_side"][side] = tot["ga_side"].get(side, 0) + n
        gf, ga = st.get("gf", 0), st.get("ga", 0)
        res["W" if gf > ga else "D" if gf == ga else "L"] += 1
    out = [f"Results W{res['W']} D{res['D']} L{res['L']}, goals {tot['gf']}-{tot['ga']}."]
    if tot["ga_side"]:
        side = max(tot["ga_side"], key=tot["ga_side"].get)
        if tot["ga_side"][side] >= 2:
            out.append(f"most goals conceded came down our {side} flank -> cover it.")
    for b, label in (("long", "long shots (>45 m)"), ("mid", "shots 25-45 m"),
                     ("close", "shots inside 25 m")):
        n, g = tot["shot"][b]
        if n >= 4:
            hint = ""
            if b == "long" and g / n < 0.08:
                hint = " -> rarely works: pass or carry closer first"
            elif g / n >= 0.2:
                hint = " -> works: keep taking them"
            out.append(f"{label}: {g} goal(s) in {n}{hint}.")
    n = sum(tot["pass"])
    if n >= 5:
        rate = tot["pass"][0] / n
        hint = " -> passes are safe, use them" if rate >= 0.7 else \
            " -> many intercepted: only pass into clear lanes" if rate < 0.5 else ""
        out.append(f"passes {round(100 * rate)} % completed{hint}.")
    n = sum(tot["carry"])
    if n >= 5 and tot["carry"][0] / n < 0.5:
        out.append(f"dribbles kept only {round(100 * tot['carry'][0] / n)} % -> release the ball earlier.")
    return out[:6]


def prior_from(snapshots: list[dict]) -> dict:
    """Long-shot attempts and goals summed over previous matches."""
    n = g = 0
    for sn in snapshots:
        v = (((sn.get("s") or {}).get("shot") or {}).get("long") or [0, 0])
        n, g = n + v[0], g + v[1]
    return {"long_n": n, "long_g": g}


def long_shot_prior_mult(team_id: int, pid: int) -> tuple[float, str | None]:
    """Multiplier for EV(long SHOOT) from previous matches (code-first ticks)."""
    if not enabled():
        return 1.0, None
    nb = _BOOKS.get((team_id, pid))
    pr = getattr(nb, "prior", None) if nb else None
    if not pr or pr["long_n"] < 10:
        return 1.0, None
    if pr["long_g"] / pr["long_n"] < 0.08:
        return float(tuning_mod.get("mem_prior_long_mult", 0.7)), \
            f"prior {pr['long_g']}/{pr['long_n']}"
    return 1.0, None


class _Remote:
    """Background writer/reader for AgentCore Memory. Never blocks a tick."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue(maxsize=60)
        self.ok = 0
        self.fail = 0
        self.dropped = 0
        self.last_error = ""
        self.read_ms = None
        self._client = None
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def client(self):
        if self._client is None:
            from bedrock_agentcore.memory import MemoryClient
            self._client = MemoryClient(
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        return self._client

    def put(self, job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            job = self.q.get()
            try:
                job(self)
                self.ok += 1
            except Exception as e:
                self.fail += 1
                self.last_error = str(e)[:200]


_REMOTE: _Remote | None = None


def _remote_q() -> _Remote:
    global _REMOTE
    if _REMOTE is None:
        _REMOTE = _Remote()
    return _REMOTE


def memory_strategy() -> str:
    """Whose lessons to read and extend (tuning memory_strategy; mem3 -> mem2)."""
    return str(tuning_mod.get("memory_strategy", "") or "") or strategy_name()


def _actor(label):
    team = os.environ.get("TEAM_ID", "neptune")
    return _safe(f"{team}-{memory_strategy()}-{label}")


def _ledger_session(label):
    return _safe(f"ledger-{memory_strategy()}-{label}")


def _write_event(session, actor, text):
    def job(r: _Remote):
        r.client().create_event(memory_id=os.environ["MEMORY_ID"], actor_id=actor,
                                session_id=session, messages=[(text, "OTHER")])
    return job


def _load_lessons(nb: Notebook):
    def job(r: _Remote):
        t0 = time.perf_counter()
        events = r.client().list_events(memory_id=os.environ["MEMORY_ID"],
                                        actor_id=_actor(nb.label),
                                        session_id=_ledger_session(nb.label),
                                        max_results=100)
        r.read_ms = round((time.perf_counter() - t0) * 1000)
        last = {}
        for ev in events:
            for pl in ev.get("payload") or []:
                txt = ((pl.get("conversational") or {}).get("content") or {}).get("text")
                try:
                    sn = json.loads(txt)
                except (TypeError, ValueError):
                    continue
                if sn.get("m") and sn.get("m") != nb.key:
                    last[sn["m"]] = sn
        snaps = sorted(last.values(), key=lambda s: s.get("ts", 0))[-6:]
        nb.lessons = lessons_from(snaps)
        nb.lessons_n = len(snaps)
        nb.prior = prior_from(snaps)
    return job


_BOOKS: dict[tuple[int, int], Notebook] = {}
_LAST_SNAPSHOT: dict[tuple[int, int], tuple[float, int]] = {}


def notebook(match_key: str, team_id: int, pid: int, label: str) -> Notebook:
    """The notebook of this player in this match (new match -> new notebook)."""
    k = (team_id, pid)
    nb = _BOOKS.get(k)
    if nb is None or nb.key != match_key:
        nb = Notebook(match_key, team_id, pid, label)
        _BOOKS[k] = nb
        if _remote():
            _remote_q().put(_load_lessons(nb))
    return nb


def match_key_from(session_id, game_state) -> str:
    raw = str(session_id or game_state.get("matchId") or "local")
    return re.sub(r"-pos\d$", "", raw)


def after_decision(nb: Notebook, cmd: dict, has_ball: bool, d_goal, source: str):
    """Record the decision; ship events to AgentCore Memory in the background.
    d_goal from the notebook (metres) wins over the caller's (maybe engine units)."""
    if nb.my_dg is not None:
        d_goal = nb.my_dg
    nb.note(cmd, has_ball, d_goal)
    if not _remote():
        return
    r = _remote_q()
    ct = (cmd or {}).get("commandType")
    if source == "llm" or has_ball:
        text = json.dumps({"t": round(nb.t), "c": ct, "hb": int(bool(has_ball)),
                           "dg": d_goal, "src": source}, separators=(",", ":"))
        r.put(_write_event(_safe(f"{nb.key}-{nb.label}"), _actor(nb.label), text))
    k = (nb.team, nb.pid)
    goals = nb.s["gf"] + nb.s["ga"]
    last_t, last_goals = _LAST_SNAPSHOT.get(k, (-1e9, -1))
    if nb.t - last_t >= 30 or goals != last_goals:
        _LAST_SNAPSHOT[k] = (nb.t, goals)
        snap = json.dumps({"m": nb.key, "ts": time.time(), "t": round(nb.t), "s": nb.s},
                          separators=(",", ":"))
        r.put(_write_event(_ledger_session(nb.label), _actor(nb.label), snap))


def log_fields(nb: Notebook | None) -> dict:
    """Memory counters for the DECISION line (mw ok / mf failed / mq queued)."""
    out = {}
    if nb is not None:
        out["nb"] = nb.tally()
        if nb.lessons is not None:
            out["ml"] = nb.lessons_n
    if _REMOTE is not None:
        out["mw"] = _REMOTE.ok
        if _REMOTE.fail:
            out["mf"] = _REMOTE.fail
            out["me"] = _REMOTE.last_error[:80]
        if _REMOTE.dropped:
            out["md"] = _REMOTE.dropped
        out["mq"] = _REMOTE.q.qsize()
        if _REMOTE.read_ms is not None:
            out["mr"] = _REMOTE.read_ms
    return out
