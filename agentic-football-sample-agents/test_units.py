"""Engine units (x +-7, y +-3.5) must give the same decisions as metres."""
import asyncio
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from test_helpers import mock_agentcore, GAME_STATE  # noqa: E402

mock_agentcore()
import agent_base  # noqa: E402
import match_memory  # noqa: E402
import units  # noqa: E402
from fallback import build_fallback, MID_CONFIG, FWD2_CONFIG, GK_CONFIG, DEF_CONFIG  # noqa: E402
from overrides import OverrideConfig  # noqa: E402

SX, SY = 55.0 / 7.0, 30.0 / 3.5


def to_engine(gs):
    g = copy.deepcopy(gs)
    for obj in [g["ball"]] + g["players"]:
        for k in ("position", "velocity"):
            v = obj.get(k)
            if isinstance(v, dict):
                v["x"] = v.get("x", 0) / SX
                v["y"] = v.get("y", 0) / SY
    return g


class App:
    class logger:
        info = staticmethod(lambda m: None)
        warn = staticmethod(lambda m: None)
        error = staticmethod(lambda m: print("ERR", m))

    def entrypoint(self, f):
        self.h = f
        return f


class Agent:
    messages = []

    def __call__(self, t):
        return '[{"commandType":"MOVE_TO","playerId":2,"parameters":{"target_x":0,"target_y":0,"sprint":false},"duration":0}]'


async def collect(h, pl):
    return [c async for c in h(pl, None)]


def decide(pid, label, cfg, gs):
    match_memory.reset()
    app = App()
    agent_base.create_invoke_handler(app, Agent(), pid, label, build_fallback(cfg),
                                     fallback_cfg=cfg, override_cfg=OverrideConfig())
    out = asyncio.run(collect(app.h, {"prompt": json.dumps(
        {"gameState": gs, "teamId": 0, "myPlayers": [pid]})}))
    return json.loads(out[0])[0]


def main():
    import tuning
    tuning._DATA.setdefault("global", {})["units_convert"] = True
    tuning._DATA["global"]["units_convert"] = False
    assert units.to_field(to_engine(GAME_STATE))[1] is False
    assert units.to_field(to_engine(GAME_STATE), force=True)[1] is True
    tuning._DATA["global"]["units_convert"] = True
    eng = to_engine(GAME_STATE)
    assert units.is_engine_units(eng) and not units.is_engine_units(GAME_STATE)
    back, conv = units.to_field(eng)
    assert conv
    for a, b in zip(back["players"], GAME_STATE["players"]):
        assert abs(a["position"]["x"] - b["position"]["x"]) < 1e-6
        assert abs(a["position"]["y"] - b["position"]["y"]) < 1e-6
    assert units.to_field(GAME_STATE) == (GAME_STATE, False)

    carrier = copy.deepcopy(GAME_STATE)
    for p in carrier["players"]:
        if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
            p["position"] = {"x": 20, "y": 3}
    carrier["ball"]["position"] = {"x": 20, "y": 3}
    carrier["ball"]["possessionAgentId"] = "agentId_4"
    cases = [(4, "FWD2", FWD2_CONFIG, carrier), (2, "MID", MID_CONFIG, GAME_STATE),
             (1, "DEF", DEF_CONFIG, GAME_STATE), (0, "GK", GK_CONFIG, GAME_STATE)]
    for pid, label, cfg, gs in cases:
        a = decide(pid, label, cfg, gs)
        b = decide(pid, label, cfg, to_engine(gs))
        assert a["commandType"] == b["commandType"], (label, a, b)
        for k, v in (a.get("parameters") or {}).items():
            w = b["parameters"].get(k)
            if isinstance(v, float):
                assert abs(v - w) < 0.5, (label, k, v, w)
            else:
                assert v == w, (label, k, v, w)
        print(f"  {label:5s} metres={a['commandType']} {a.get('parameters')}  engine={b['commandType']}")
    import stamina
    tuning._DATA["global"]["units_convert"] = False
    eng = to_engine(GAME_STATE)
    gk = [{"commandType": "MOVE_TO", "parameters": {"target_x": -42, "target_y": 8, "sprint": True}}]
    out, tag = stamina.guard(copy.deepcopy(gk), eng, 0, 0)
    assert tag == "gk-walk", (tag, out)
    far = [{"commandType": "MOVE_TO", "parameters": {"target_x": 30, "target_y": 0, "sprint": True}}]
    out, tag = stamina.guard(copy.deepcopy(far), eng, 0, 1)
    assert tag is None and out[0]["parameters"]["sprint"] is True, (tag, out)
    print("UNITS OK")


if __name__ == "__main__":
    main()
