"""Regression tests for the EV/Markov carrier, the defensive search and the
EV keeper (lib/ev_policy.py, lib/def_search.py). No AWS.
"""
import copy
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from test_helpers import GAME_STATE  # noqa: E402
import def_search  # noqa: E402
import ev_policy  # noqa: E402
import tuning  # noqa: E402
tuning._DATA.setdefault("global", {})["ev_shot_bias"] = 1.0
from parsing import VALID_COMMANDS  # noqa: E402


def state(holder_team, holder_idx, ball):
    gs = copy.deepcopy(GAME_STATE)
    for p in gs["players"]:
        if p["teamCode"] == holder_team and p["agentId"].endswith(f"_{holder_idx}"):
            p["position"] = dict(ball)
    gs["ball"]["position"] = dict(ball)
    gs["ball"]["possessionAgentId"] = f"agentId_{holder_idx}"
    return gs


def main():
    assert ev_policy.MODEL is not None, "lib/ev_model.json missing: run sim/fit_ev.py"
    V = ev_policy.MODEL["xt"]
    att = lambda x, y: V[ev_policy.cell(x, y, 1.0)]  # noqa: E731
    assert att(48, 0) > att(0, 0) > att(-48, 0), "threat must grow toward the opponent goal"

    for pid, label in ((1, "DEF"), (2, "MID"), (3, "FWD1"), (4, "FWD2")):
        gs = state("home", pid, {"x": 20, "y": 3})
        t = time.perf_counter()
        cmd, info = ev_policy.choose(gs, 0, pid, label)
        ms = (time.perf_counter() - t) * 1000
        assert cmd["commandType"] in VALID_COMMANDS and cmd["playerId"] == pid, cmd
        assert ms < 20, f"carrier decision took {ms:.1f} ms"
        crit = ev_policy.evaluate(gs, 0, pid, cmd)
        assert crit["regret"] < 1e-6, crit
    gs = state("home", 3, {"x": 46, "y": 0})
    for p in gs["players"]:
        if p["teamCode"] == "away" and not p["agentId"].endswith("_0"):
            p["position"] = {"x": 0, "y": 25}
    assert ev_policy.evaluate(gs, 0, 3, {"commandType": "SHOOT", "parameters": {}})["ev"] > 0.3
    print("  EV carrier: valid, < 20 ms, sensible values")

    gs = state("away", 3, {"x": -20, "y": 3})
    cmd, _ = ev_policy.choose(gs, 1, 3, "FWD1")
    assert cmd["teamId"] == 1
    if cmd["commandType"] == "MOVE_TO":
        assert cmd["parameters"]["target_x"] < -20, "away carrier must carry toward -x"
    print("  EV carrier: away side mirrored")

    gs = state("away", 4, {"x": -25, "y": 5})
    tasks, ms_max = {}, 0.0
    for pid in (1, 2, 3, 4):
        t = time.perf_counter()
        cmd, info = def_search.assign(gs, 0, pid)
        ms_max = max(ms_max, (time.perf_counter() - t) * 1000)
        assert cmd["commandType"] in VALID_COMMANDS, cmd
        tasks[pid] = info["task"]
    assert len(set(tasks.values())) == 4, f"tasks must be distinct: {tasks}"
    assert ms_max < 120, f"search took {ms_max:.0f} ms"
    assert def_search.assign(state("home", 2, {"x": 0, "y": 0}), 0, 1) == (None, None), \
        "search must stay off when we have the ball"
    print(f"  defensive search: consistent plan {tasks}, max {ms_max:.0f} ms")

    import time as _time
    real = _time.perf_counter
    gs = state("away", 3, {"x": -20, "y": 5})
    fast = def_search.assign(gs, 0, 2)[0]
    tick = [0.0]

    def slow_clock():
        tick[0] += 0.002
        return real() + tick[0]
    def_search.time.perf_counter = slow_clock
    try:
        slow = def_search.assign(gs, 0, 2)[0]
    finally:
        def_search.time.perf_counter = real
    assert fast == slow, f"plan depends on host speed: {fast} vs {slow}"
    print("  defensive search: same plan regardless of host speed")

    gs = state("home", 0, {"x": -50, "y": 0})
    cmd, info = def_search.gk_distribute(gs, 0)
    assert cmd["commandType"] == "GK_DISTRIBUTE" and cmd["parameters"]["target_player_id"] != 0
    gx, gy = def_search.gk_position(state("away", 3, {"x": -30, "y": 15}), 0)
    assert -55 < gx < -50 and gy > 0, (gx, gy)
    print("  keeper: distributes to a teammate, shades toward the ball")
    print("EV / SEARCH TESTS OK")


if __name__ == "__main__":
    main()
