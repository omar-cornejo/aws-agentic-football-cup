"""mem3.1: lane-gated CENTER shots, Markov + constraints carrier, double team,
lane block, loose ball, prompt addenda — and mem3 left exactly as it was.
"""
import asyncio
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from test_helpers import mock_agentcore  # noqa: E402

mock_agentcore()
import tuning  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M31 = json.load(open(os.path.join(HERE, "lib", "tuning.mem31.json")))
MEM3 = json.load(open(os.path.join(HERE, "lib", "tuning.mem3.json")))
M32 = json.load(open(os.path.join(HERE, "lib", "tuning.mem32.json")))
M33 = json.load(open(os.path.join(HERE, "lib", "tuning.mem33.json")))
tuning._DATA = copy.deepcopy(M31)

import agent_base  # noqa: E402
import match_memory  # noqa: E402
import mem3  # noqa: E402
import mem31  # noqa: E402
from fallback import build_fallback, FWD1_CONFIG, DEF_CONFIG  # noqa: E402
from overrides import OverrideConfig  # noqa: E402

SX, SY = 55.0 / 7.0, 30.0 / 3.5
FAIL = []


def check(name, ok, info=""):
    print(f"  {'ok ' if ok else 'FAIL'} {name}" + (f"   ({info})" if not ok else ""))
    if not ok:
        FAIL.append(name)


def state(home, away, ball, holder=None, chat=None, mode="OPEN_PLAY", vel=(0, 0)):
    players = []
    for code, team in (("home", home), ("away", away)):
        for i, (x, y) in team.items():
            players.append({"agentId": f"agentId_{i}", "teamCode": code,
                            "position": {"x": x / SX, "y": y / SY},
                            "velocity": {"x": 0, "y": 0}, "stamina": 0.9})
    b = {"position": {"x": ball[0] / SX, "y": ball[1] / SY, "z": 0},
         "velocity": {"x": vel[0] / SX, "y": vel[1] / SY, "z": 0}, "isFree": holder is None}
    if holder:
        b["possessionAgentId"] = f"agentId_{holder[1]}"
        b["possessionTeamCode"] = holder[0]
    return {"tick": 100, "gameTime": 60, "playMode": mode, "score": {"home": 0, "away": 0},
            "ball": b, "players": players,
            "teamChat": [{"senderTeamCode": "home", "message": m, "timestamp": 0}
                         for m in (chat or [])]}


HOME = {0: (-52, 0), 1: (-25, 0), 2: (-5, 0), 3: (15, -8), 4: (15, 8)}
AWAY = {0: (52, 0), 1: (20, 0), 2: (5, 10), 3: (-15, -6), 4: (-20, 6)}
LABEL = {1: "DEF", 2: "MID", 3: "FWD1", 4: "FWD2"}


def decide(gs, pid):
    return mem3.decide(gs, 0, pid, LABEL[pid])


def test_aim():
    print("every SHOOT goes CENTER")
    cmds = [{"commandType": "SHOOT", "parameters": {"aim_location": "TL", "power": 0.8}}]
    tag = mem3.shape(cmds, state(HOME, AWAY, (40, 0), ("home", 4)), 0, 4, "FWD2")
    check("TL -> CENTER, power 1", tag == "aim-center" and cmds[0]["parameters"] ==
          {"aim_location": "CENTER", "power": 1.0}, cmds)
    home = dict(HOME)
    home[4] = (30, 3)
    away = dict(AWAY)
    away[0] = (52, 5)
    c, t = decide(state(home, away, (30, 3), ("home", 4)), 4)
    check("shoot zone shot is CENTER", c["commandType"] == "SHOOT"
          and c["parameters"]["aim_location"] == "CENTER", (c, t))


def test_coach_shoot():
    print("«shoot» order: close or clear lane -> shoot; blocked and far -> carry/pass")
    home = dict(HOME)
    home[3] = (25, -2)
    away = dict(AWAY)
    away[1] = (35, -1)
    c, t = decide(state(home, away, (25, -2), ("home", 3), chat=["sooht"]), 3)
    check("30 m -> shoot", t == "coach-shoot" and c["commandType"] == "SHOOT", (c, t))
    home[3] = (5, 0)
    away = {0: (52, 0), 1: (20, 15), 2: (5, 15), 3: (-15, -6), 4: (-20, 6)}
    c, t = decide(state(home, away, (5, 0), ("home", 3), chat=["shoot"]), 3)
    check("50 m clear lane -> shoot", t == "coach-shoot", (c, t))
    away[1] = (15, 0.5)
    c, t = decide(state(home, away, (5, 0), ("home", 3), chat=["shoot"]), 3)
    check("50 m blocked -> no blind shot", t in ("coach-shoot-carry", "coach-shoot-pass")
          and c["commandType"] != "SHOOT", (c, t))
    if c["commandType"] == "MOVE_TO":
        check("carry goes toward goal", c["parameters"]["target_x"] > 5, c)


def test_assist():
    print("«shoot» far and blocked, striker free in the box -> pass to him")
    home = dict(HOME)
    home[3] = (5, -15)
    home[4] = (35, 12)
    away = {0: (52, 0), 1: (30, -7.5), 2: (0, 25), 3: (-15, -6), 4: (-20, 6)}
    c, t = decide(state(home, away, (5, -15), ("home", 3), chat=["shoot"]), 3)
    check("pass to FWD2", t == "coach-shoot-pass" and c["parameters"]["target_player_id"] == 4,
          (c, t))
    c, t = decide(state(home, away, (5, -15), ("home", 3)), 3)
    check("same without order: Markov passes to him", t in ("m31-assist", "m31-pass")
          and c["parameters"]["target_player_id"] == 4, (c, t))


def test_carrier_constraints():
    print("carrier: blocked shots pruned, deep blast only with the first 25 m clear")
    home = dict(HOME)
    home[1] = (-45, 0)
    away = {0: (52, 0), 1: (30, 0), 2: (5, 10), 3: (-15, -12), 4: (-20, 12)}
    c, t = decide(state(home, away, (-45, 0), ("home", 1)), 1)
    check("deep, first 25 m clear -> blast allowed", c is None or t != "m31-clear" or
          c["commandType"] == "SHOOT", (c, t))
    if c and c["commandType"] == "SHOOT":
        check("blast is lane-gated", t in ("m31-shot-deep-clear", "m31-clear"), t)
    away[3] = (-38, 0.5)
    c, t = decide(state(home, away, (-45, 0), ("home", 1)), 1)
    check("opponent on the line -> no deep-clear shot", t != "m31-shot-deep-clear", (c, t))
    home[2] = (-5, 0)
    away = {0: (52, 0), 1: (15, 0.5), 2: (5, 10), 3: (-15, -12), 4: (-20, 12)}
    c, t = decide(state(home, away, (-5, 0), ("home", 2)), 2)
    check("60 m blocked: MID does not shoot", c is None or c["commandType"] != "SHOOT", (c, t))


def test_double_team():
    print("carrier 20 m from our goal: nearest presses, second closes goal-side")
    home = {0: (-52, 0), 1: (-33, 2), 2: (-28, -4), 3: (0, -8), 4: (0, 8)}
    away = {0: (52, 0), 1: (20, 0), 2: (5, 10), 3: (-36, 0), 4: (-20, 6)}
    gs = state(home, away, (-36, 0), ("away", 3))
    c1, t1 = decide(gs, 1)
    c2, t2 = decide(gs, 2)
    check("DEF presses", t1 == "m31-press" and c1["commandType"] in
          ("PRESS_BALL", "SLIDE_TACKLE"), (c1, t1))
    check("MID doubles goal-side", t2 == "m31-double" and c2["parameters"]["target_x"] < -36,
          (c2, t2))
    c3, t3 = decide(gs, 3)
    check("FWD left to the search", t3 is None, t3)


def test_lane_block():
    print("carrier in midfield: DEF stands on the ball-goal line")
    home = {0: (-52, 0), 1: (-30, -8), 2: (-2, 6), 3: (10, -8), 4: (10, 8)}
    away = {0: (52, 0), 1: (20, 0), 2: (0, 10), 3: (25, -6), 4: (30, 6)}
    gs = state(home, away, (0, 10), ("away", 2))
    c, t = decide(gs, 1)
    ok = t == "m31-block"
    check("DEF blocks", ok, (c, t))
    if ok:
        x, y = c["parameters"]["target_x"], c["parameters"]["target_y"]
        check("on the line, 16-29 m from goal", -40 <= x <= -26 and 0 < y < 10, (x, y))
    away[4] = (-40, 6)
    c, t = decide(state(home, away, (0, 10), ("away", 2)), 1)
    check("danger near goal -> no block", t != "m31-block", t)


def test_loose():
    print("loose ball: the player nearest to where it goes intercepts")
    gs = state(HOME, AWAY, (-8, 0), None)
    c2, t2 = decide(gs, 2)
    c1, t1 = decide(gs, 1)
    check("MID intercepts", t2 == "m31-loose" and c2["commandType"] == "INTERCEPT", (c2, t2))
    check("DEF does not", t1 is None, t1)
    gs = state(HOME, AWAY, (-5, 0), None, vel=(20, -10))
    c3, t3 = decide(gs, 3)
    check("receiver goes for a ball in flight", t3 == "m31-loose", (c3, t3))


def test_prompts():
    print("prompt addenda")
    check("FWD1 told not to mark with the ball", "NEVER MARK" in mem31.prompt_addendum("FWD1"))
    check("DEF lane block", "straight line" in mem31.prompt_addendum("DEF"))


def test_mem3_untouched():
    print("mem3 tuning: nothing of mem3.1 runs")
    tuning._DATA = copy.deepcopy(MEM3)
    try:
        check("no addendum", mem31.prompt_addendum("FWD1") == "")
        gs = state(HOME, AWAY, (-8, 0), None)
        check("loose ball left to the LLM", mem3.decide(gs, 0, 2, "MID") == (None, None))
        cmds = [{"commandType": "SHOOT", "parameters": {"aim_location": "TL", "power": 1.0}}]
        mem3.shape(cmds, gs, 0, 4, "FWD2")
        check("aim untouched", cmds[0]["parameters"]["aim_location"] == "TL", cmds)
        home = dict(HOME)
        home[3] = (5, 0)
        away = dict(AWAY)
        away[1] = (15, 0.5)
        c, t = mem3.decide(state(home, away, (5, 0), ("home", 3), chat=["shoot"]), 0, 3, "FWD1")
        check("mem3 still shoots at 50 m on order", t == "coach-shoot", (c, t))
        check("no op/ln log fields", mem31.log_fields(gs, 0, 2, {}) == {})
    finally:
        tuning._DATA = copy.deepcopy(M31)


def test_mem32():
    print("mem3.2 = mem3 + bench «shoot» of 3.1 + CENTER + shoot zone 35 m")
    tuning._DATA = copy.deepcopy(M32)
    try:
        check("no prompt addendum", mem31.prompt_addendum("FWD1") == "")
        check("loose ball as mem3", decide(state(HOME, AWAY, (-8, 0), None), 2) == (None, None))
        home = {0: (-52, 0), 1: (-33, 2), 2: (-28, -4), 3: (0, -8), 4: (0, 8)}
        away = {0: (52, 0), 1: (20, 0), 2: (5, 10), 3: (-36, 0), 4: (-20, 6)}
        c, t = decide(state(home, away, (-36, 0), ("away", 3)), 2)
        check("defence as mem3 (no double team)", t is None, t)
        home = dict(HOME)
        home[1] = (-45, 0)
        away = {0: (52, 0), 1: (30, 0), 2: (5, 10), 3: (-38, 0.5), 4: (-20, 12)}
        c, t = decide(state(home, away, (-45, 0), ("home", 1)), 1)
        check("carrier left to mem3 Markov (long shots kept)", t is None, (c, t))
        home = dict(HOME)
        home[4] = (21, 3)
        c, t = decide(state(home, AWAY, (21, 3), ("home", 4)), 4)
        check("34 m -> shoot zone, CENTER", t == "shoot-zone"
              and c["parameters"]["aim_location"] == "CENTER", (c, t))
        home[3] = (5, 0)
        away = dict(AWAY)
        away[1] = (15, 0.5)
        c, t = decide(state(home, away, (5, 0), ("home", 3), chat=["shoot"]), 3)
        check("«shoot» 50 m blocked -> carry/pass (3.1 rule)", t in ("coach-shoot-carry",
              "coach-shoot-pass"), (c, t))
        cmds = [{"commandType": "SHOOT", "parameters": {"aim_location": "TR", "power": 1.0}}]
        mem3.shape(cmds, state(HOME, AWAY, (0, 0), ("home", 4)), 0, 4, "FWD2")
        check("EV/LLM shots go CENTER", cmds[0]["parameters"]["aim_location"] == "CENTER", cmds)
    finally:
        tuning._DATA = copy.deepcopy(M31)


def test_mem33():
    print("mem3.3 = mem3.2 + DEF not past -20 m while they have the ball upfield")
    tuning._DATA = copy.deepcopy(M33)
    try:
        home = dict(HOME)
        home[1] = (-5, 0)
        away = {0: (52, 0), 1: (25, 0), 2: (10, 5), 3: (-10, -6), 4: (2, 6)}
        gs = state(home, away, (25, 0), ("away", 1))
        cmds = [{"commandType": "MARK", "parameters": {"target_player_id": 4}}]
        t = mem3.shape(cmds, gs, 0, 1, "DEF")
        c = cmds[0]
        check("LLM MARK upfield -> def-hold MOVE_TO at -20 m", t == "def-hold"
              and c["commandType"] == "MOVE_TO" and abs(c["parameters"]["target_x"] + 20) < 0.2, (t, c))
        c, t = decide(state(home, away, (25, 0), ("away", 1), chat=["mark"]), 1)
        check("bench «mark» upfield -> def-hold", t == "def-hold" and c["commandType"] == "MOVE_TO", (c, t))
        cmds = [{"commandType": "MARK", "parameters": {"target_player_id": 3}}]
        check("MARK on a striker in our zone kept", mem3.shape(
            cmds, state(home, {**away, 3: (-30, -6)}, (25, 0), ("away", 1)), 0, 1, "DEF") is None
              and cmds[0]["commandType"] == "MARK", cmds)
        gs = state(home, away, (-20, 0), ("away", 3))
        cmds = [{"commandType": "PRESS_BALL", "parameters": {"intensity": 1.0}}]
        check("ball near our third: DEF presses freely", mem3.shape(cmds, gs, 0, 1, "DEF") is None
              and cmds[0]["commandType"] == "PRESS_BALL", cmds)
        home2 = dict(home)
        home2[1] = (22, 0)
        cmds = [{"commandType": "SLIDE_TACKLE", "parameters": {"target_player_id": -1}}]
        check("DEF next to the carrier keeps the tackle", mem3.shape(
            cmds, state(home2, away, (25, 0), ("away", 1)), 0, 1, "DEF") is None, cmds)
        cmds = [{"commandType": "MARK", "parameters": {"target_player_id": 4}}]
        check("MID untouched", mem3.shape(cmds, state(home, away, (25, 0), ("away", 1)), 0, 2, "MID") is None)
        cmds = [{"commandType": "MARK", "parameters": {"target_player_id": 4}}]
        mem3.shape(cmds, state(home, away, (25, 0), ("home", 3)), 0, 1, "DEF")
        check("our ball: no hold", cmds[0]["commandType"] == "MARK", cmds)
        tuning._DATA = copy.deepcopy(M32)
        cmds = [{"commandType": "MARK", "parameters": {"target_player_id": 4}}]
        check("mem3.2 unchanged (no def_ceiling_m)", mem3.shape(
            cmds, state(home, away, (25, 0), ("away", 1)), 0, 1, "DEF") is None)
        check("mem3.3 own memory pool", M33["global"]["memory_strategy"] == "mem33")
    finally:
        tuning._DATA = copy.deepcopy(M31)


class App:
    class logger:
        info = staticmethod(lambda m: LOG.append(m))
        warn = staticmethod(lambda m: None)
        error = staticmethod(lambda m: print("ERR", m))

    def entrypoint(self, f):
        self.h = f
        return f


LOG = []


class SlowAgent:
    messages = []
    calls = 0
    system_prompt = "base"

    def __call__(self, t):
        SlowAgent.calls += 1
        return '[{"commandType":"SHOOT","playerId":3,"parameters":{"aim_location":"TR","power":1.0},"duration":0}]'


async def collect(h, pl):
    return [c async for c in h(pl, None)]


def run_handler(pid, label, cfg, gs):
    match_memory.reset()
    app = App()
    agent_base.create_invoke_handler(app, SlowAgent(), pid, label, build_fallback(cfg),
                                     fallback_cfg=cfg, override_cfg=OverrideConfig())
    out = asyncio.run(collect(app.h, {"prompt": json.dumps(
        {"gameState": gs, "teamId": 0, "myPlayers": [pid]})}))
    return json.loads(out[0])[0]


def test_handler():
    print("full agent handler")
    LOG.clear()
    home = dict(HOME)
    home[3] = (5, -15)
    home[4] = (35, 12)
    away = {0: (52, 0), 1: (30, -7.5), 2: (0, 25), 3: (-15, -6), 4: (-20, 6)}
    SlowAgent.calls = 0
    c = run_handler(3, "FWD1", FWD1_CONFIG, state(home, away, (5, -15), ("home", 3)))
    check("FWD1 assists FWD2 in code", c["commandType"] == "PASS", c)
    check("no LLM call", SlowAgent.calls == 0, SlowAgent.calls)
    dec = [json.loads(m[9:]) for m in LOG if m.startswith("DECISION ")]
    check("DECISION logs opponents", dec and len(dec[-1].get("op", [])) == 5, dec[-1:])
    c = run_handler(1, "DEF", DEF_CONFIG, state(HOME, AWAY, (-8, 0), ("home", 2)))
    check("handler runs for an off-ball DEF", c.get("commandType") is not None, c)


def main():
    for t in (test_aim, test_coach_shoot, test_assist, test_carrier_constraints,
              test_double_team, test_lane_block, test_loose, test_prompts,
              test_mem3_untouched, test_mem32, test_mem33, test_handler):
        t()
    print()
    print("FAILED: " + ", ".join(FAIL) if FAIL else "all mem3.1 tests passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
