"""mem3: bench orders first (typos ok), shoot zone, MID passes, DEF/MID shape."""
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
MEM3 = json.load(open(os.path.join(HERE, "lib", "tuning.mem3.json")))
MEM2 = json.load(open(os.path.join(HERE, "lib", "tuning.mem2.json")))
tuning._DATA = copy.deepcopy(MEM3)

import agent_base  # noqa: E402
import match_memory  # noqa: E402
import mem3  # noqa: E402
from fallback import build_fallback, MID_CONFIG, FWD2_CONFIG, DEF_CONFIG  # noqa: E402
from overrides import OverrideConfig  # noqa: E402

SX, SY = 55.0 / 7.0, 30.0 / 3.5
FAIL = []


def check(name, ok, info=""):
    print(f"  {'ok ' if ok else 'FAIL'} {name}" + (f"   ({info})" if not ok else ""))
    if not ok:
        FAIL.append(name)


def state(home, away, ball, holder=None, chat=None, mode="OPEN_PLAY"):
    """home/away: {idx: (x, y)} in metres; holder: ('home'|'away', idx). Returns ENGINE units."""
    players = []
    for code, team in (("home", home), ("away", away)):
        for i, (x, y) in team.items():
            players.append({"agentId": f"agentId_{i}", "teamCode": code,
                            "position": {"x": x / SX, "y": y / SY},
                            "velocity": {"x": 0, "y": 0}, "stamina": 0.9})
    b = {"position": {"x": ball[0] / SX, "y": ball[1] / SY, "z": 0},
         "velocity": {"x": 0, "y": 0, "z": 0}, "isFree": holder is None}
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


def test_parse():
    print("parse_order")
    cases = {"makr": "mark", "MARK": "mark", "dmark": "mark", "atack": "attack",
             "sooht": "shoot", "shoot": "shoot", "goal": "shoot", "chuta": "shoot",
             "pass": "pass", "defend": "defend", "attack | mark": "mark",
             "hola": None, "": None}
    for text, want in cases.items():
        got = mem3.parse_order(text)
        check(f"{text!r} -> {want}", got == want, got)


def test_mark_order():
    print("order «makr»: one presser, the rest man-mark different players")
    away = dict(AWAY)
    away[3] = (-22, -4)
    gs = state(HOME, away, (-22, -4), ("away", 3), chat=["makr"])
    cmds = {pid: decide(gs, pid) for pid in (1, 2, 3, 4)}
    tags = {pid: t for pid, (c, t) in cmds.items()}
    check("every outfield player obeys", all(t == "coach-mark" for t in tags.values()), tags)
    kinds = {pid: c["commandType"] for pid, (c, t) in cmds.items() if c}
    pressers = [p for p, k in kinds.items() if k in ("PRESS_BALL", "SLIDE_TACKLE")]
    check("exactly one presser (DEF, nearest)", pressers == [1], kinds)
    marked = [c["parameters"]["target_player_id"] for pid, (c, t) in cmds.items()
              if c and c["commandType"] == "MARK"]
    check("marks are distinct", len(marked) == len(set(marked)) == 3, marked)
    check("nobody marks their keeper or the carrier", not {0, 3} & set(marked), marked)


def test_tackle_close():
    print("order «mark» with the carrier 2 m away -> slide tackle")
    away = dict(AWAY)
    away[3] = (-23, 1)
    gs = state(HOME, away, (-23, 1), ("away", 3), chat=["mark"])
    c, t = decide(gs, 1)
    check("DEF slides", c and c["commandType"] == "SLIDE_TACKLE", c)


def test_loose_ball():
    print("order «mark» with a loose ball -> nearest player intercepts")
    gs = state(HOME, AWAY, (-20, 2), None, chat=["mark"])
    c, t = decide(gs, 1)
    check("DEF intercepts", c and c["commandType"] == "INTERCEPT", c)


def test_defend_order():
    print("order «defend»: DEF/MID mark, forwards keep their own play")
    gs = state(HOME, AWAY, (-15, -6), ("away", 3), chat=["defend"])
    c3, t3 = decide(gs, 3)
    c4, t4 = decide(gs, 4)
    c2, t2 = decide(gs, 2)
    check("MID obeys", t2 == "coach-defend", t2)
    check("forwards not forced to mark", t3 != "coach-defend" or c3["commandType"] != "MARK", (c3, t3))
    check("FWD2 not forced", t4 is None, t4)


def test_shoot_order():
    print("order «sooht» with the ball 40 m out -> shoot; same without order -> no")
    home = dict(HOME)
    home[3] = (15, -5)
    gs = state(home, AWAY, (15, -5), ("home", 3), chat=["sooht"])
    c, t = decide(gs, 3)
    check("shoots on order", c and c["commandType"] == "SHOOT" and t == "coach-shoot", (c, t))
    gs = state(home, AWAY, (15, -5), ("home", 3))
    c, t = decide(gs, 3)
    check("no order: not forced", t is None, (c, t))
    home[3] = (-30, 0)
    gs = state(home, AWAY, (-30, 0), ("home", 3), chat=["shoot"])
    c, t = decide(gs, 3)
    check("no blast from own half", t != "coach-shoot", t)


def test_untrusted():
    print("message from the other bench is ignored")
    home = dict(HOME)
    home[3] = (15, -5)
    gs = state(home, AWAY, (15, -5), ("home", 3))
    gs["teamChat"] = [{"senderTeamCode": "away", "message": "shoot", "timestamp": 0}]
    c, t = decide(gs, 3)
    check("ignored", t is None, t)


def test_shoot_zone():
    print("shoot zone: 25 m and inside the box -> shoot; 35 m -> no")
    for x, want in ((30, True), (45, True), (20, False)):
        home = dict(HOME)
        home[4] = (x, 3)
        gs = state(home, AWAY, (x, 3), ("home", 4))
        c, t = decide(gs, 4)
        d = round(55 - x)
        check(f"{d} m -> {'SHOOT' if want else 'free'}",
              (t == "shoot-zone") == want, (c, t))


def test_mid_pass():
    print("MID with the ball: forward pass to an open striker")
    gs = state(HOME, AWAY, (-5, 0), ("home", 2))
    c, t = decide(gs, 2)
    check("passes forward", t == "mid-pass" and c["commandType"] == "PASS"
          and c["parameters"]["target_player_id"] in (3, 4), (c, t))
    away = {0: (52, 0), 1: (5, -4), 2: (5, 4), 3: (40, 0), 4: (40, 5)}
    gs = state(HOME, away, (-5, 0), ("home", 2))
    c, t = decide(gs, 2)
    check("lanes blocked -> no forced pass", t is None, (c, t))


def test_shape():
    print("shape: DEF not deeper than -26 m with the ball upfield; MID within 7 m of centre")
    gs = state(HOME, AWAY, (10, 0), ("away", 1))
    cmds = [{"commandType": "MOVE_TO", "parameters": {"target_x": -45, "target_y": 0}}]
    tag = mem3.shape(cmds, gs, 0, 1, "DEF")
    check("DEF pushed up", tag == "def-up" and cmds[0]["parameters"]["target_x"] == -26, cmds)
    gs = state(HOME, AWAY, (-40, 0), ("away", 3))
    cmds = [{"commandType": "MOVE_TO", "parameters": {"target_x": -45, "target_y": 0}}]
    check("ball in our third: DEF may drop", mem3.shape(cmds, gs, 0, 1, "DEF") is None, cmds)
    cmds = [{"commandType": "MOVE_TO", "parameters": {"target_x": 0, "target_y": -20}}]
    tag = mem3.shape(cmds, gs, 0, 2, "MID")
    check("MID centred", tag == "mid-centre" and cmds[0]["parameters"]["target_y"] == -7, cmds)


def test_mem2_untouched():
    print("mem2 tuning: mem3 rules are off")
    tuning._DATA = copy.deepcopy(MEM2)
    try:
        home = dict(HOME)
        home[4] = (40, 3)
        gs = state(home, AWAY, (40, 3), ("home", 4), chat=["mark"])
        check("decide off", mem3.decide(gs, 0, 4, "FWD2") == (None, None))
        cmds = [{"commandType": "MOVE_TO", "parameters": {"target_x": 0, "target_y": -20}}]
        check("shape off", mem3.shape(cmds, gs, 0, 2, "MID") is None)
    finally:
        tuning._DATA = copy.deepcopy(MEM3)


class App:
    class logger:
        info = staticmethod(lambda m: None)
        warn = staticmethod(lambda m: None)
        error = staticmethod(lambda m: print("ERR", m))

    def entrypoint(self, f):
        self.h = f
        return f


class SlowAgent:
    """Fails the test if the LLM is called when mem3 already decided."""
    messages = []
    calls = 0

    def __call__(self, t):
        SlowAgent.calls += 1
        return '[{"commandType":"MOVE_TO","playerId":2,"parameters":{"target_x":0,"target_y":-25,"sprint":false},"duration":0}]'


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
    print("full agent handler (engine units in, LLM skipped when mem3 decides)")
    home = dict(HOME)
    home[4] = (32, 3)
    SlowAgent.calls = 0
    c = run_handler(4, "FWD2", FWD2_CONFIG, state(home, AWAY, (32, 3), ("home", 4)))
    check("FWD2 shoots from 23 m", c["commandType"] == "SHOOT", c)
    check("no LLM call", SlowAgent.calls == 0, SlowAgent.calls)
    c = run_handler(1, "DEF", DEF_CONFIG,
                    state(HOME, AWAY, (-15, -6), ("away", 3), chat=["makr"]))
    check("DEF obeys «makr» inside the real agent", c["commandType"] in
          ("MARK", "PRESS_BALL", "SLIDE_TACKLE"), c)
    c = run_handler(2, "MID", MID_CONFIG, state(HOME, AWAY, (20, 0), ("home", 3)))
    y = (c.get("parameters") or {}).get("target_y")
    check("MID LLM move stays central", c["commandType"] != "MOVE_TO" or abs(y) <= 7, c)


def main():
    for t in (test_parse, test_mark_order, test_tackle_close, test_loose_ball, test_defend_order,
              test_shoot_order, test_untrusted, test_shoot_zone, test_mid_pass,
              test_shape, test_mem2_untouched, test_handler):
        t()
    print()
    if FAIL:
        print(f"MEM3 FAILED: {len(FAIL)} -> {FAIL}")
        sys.exit(1)
    print("MEM3 OK")


if __name__ == "__main__":
    main()
