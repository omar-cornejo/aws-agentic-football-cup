"""In-process match memory: zero network calls, zero added latency."""
from __future__ import annotations

import math

import tuning as tuning_mod

_STATE: dict[tuple[int, int], dict] = {}


def _cfg(key: str, default):
    return tuning_mod.get(key, default)


def _scores(game_state: dict, team_id: int) -> tuple[int, int]:
    sc = game_state.get("score") or {}
    home, away = int(sc.get("home", 0) or 0), int(sc.get("away", 0) or 0)
    return (home, away) if team_id == 0 else (away, home)


def _slot(game_state: dict, team_id: int, pid: int) -> dict:
    key = (team_id, pid)
    t = float(game_state.get("gameTime", 0) or 0)
    us, them = _scores(game_state, team_id)
    st = _STATE.get(key)
    new_match = (st is None or t + 10 < st["last_t"]
                 or (us == 0 and them == 0 and (st["us"] or st["them"]) and t < 15))
    if new_match:
        st = {"last_t": t, "us": us, "them": them, "pending": None,
              "shots": 0, "goals": 0, "misses": 0, "kickoff_t": t}
        _STATE[key] = st
    return st


def observe(game_state: dict, team_id: int, pid: int) -> None:
    """Call once per tick before deciding: resolves the pending long shot."""
    st = _slot(game_state, team_id, pid)
    t = float(game_state.get("gameTime", 0) or 0)
    us, them = _scores(game_state, team_id)
    pend = st["pending"]
    if pend is not None:
        if us > st["us"]:
            st["goals"] += 1
            st["misses"] = 0
            st["pending"] = None
        elif them > st["them"] or t - pend >= float(_cfg("mem_window_s", 5.0)):
            st["misses"] += 1
            st["pending"] = None
    if us != st["us"] or them != st["them"]:
        st["kickoff_t"] = t
    st["last_t"], st["us"], st["them"] = t, us, them


def record(commands: list, game_state: dict, team_id: int, pid: int,
           dist_to_goal: float | None, has_ball: bool) -> None:
    """Call with the command actually sent: remembers an open-play long shot."""
    if not commands or not has_ball or dist_to_goal is None:
        return
    if commands[0].get("commandType") != "SHOOT":
        return
    pm = game_state.get("playMode")
    if pm not in (None, "", 0, "OPEN_PLAY"):
        return
    if dist_to_goal < float(_cfg("mem_long_m", 45.0)):
        return
    st = _slot(game_state, team_id, pid)
    if st["pending"] is None:
        st["pending"] = float(game_state.get("gameTime", 0) or 0)
        st["shots"] += 1


def long_shot_multiplier(game_state: dict, team_id: int, pid: int,
                         dist_to_goal: float) -> tuple[float, str | None]:
    """(multiplier for EV(SHOOT), reason) — 1.0 when nothing applies."""
    if dist_to_goal < float(_cfg("mem_long_m", 45.0)):
        return 1.0, None
    mult, why = 1.0, None
    st = _STATE.get((team_id, pid))
    t = float(game_state.get("gameTime", 0) or 0)
    if st is not None and t - st.get("kickoff_t", -1e9) < float(_cfg("kickoff_no_long_s", 8.0)):
        return 0.0, "kickoff"
    us, them = _scores(game_state, team_id)
    if us > them:
        mult *= float(_cfg("lead_long_shot_mult", 0.6))
        why = "lead"
    if (st is not None and st["goals"] == 0
            and st["misses"] >= int(_cfg("mem_miss_after", 3))):
        mult *= float(_cfg("mem_miss_mult", 0.5))
        why = "mem" if why is None else why + "+mem"
    return mult, why


def snapshot(team_id: int, pid: int) -> dict:
    st = _STATE.get((team_id, pid)) or {}
    return {k: st.get(k) for k in ("shots", "goals", "misses")}


def reset() -> None:
    _STATE.clear()


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
